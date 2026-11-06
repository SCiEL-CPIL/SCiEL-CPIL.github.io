"""
Strategy: Direct optimization of the 8x8 fundamental domain using Simulated Annealing.
The 32x32 cell is reconstructed from the 8x8 block via 4-fold symmetry.
Uses a moderately accurate plane-wave solver (N_pw ~ 25) to guide the search.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_symmetric_cell(fundamental_8x8):
    """
    Reconstructs a 32x32 symmetric cell from an 8x8 fundamental domain.
    The fundamental domain is the top-left 8x8 corner.
    Symmetry:
    1. Horizontal flip (left-right)
    2. Vertical flip (top-bottom)
    3. Transpose (diagonal)
    Actually, the task requires:
    cell == cell[::-1] row-wise (vertical flip)
    cell == cell[::-1] col-wise (horizontal flip)
    cell == cell.T (transpose)
    
    This implies 4-fold rotational symmetry and reflection symmetry.
    The fundamental domain for C4v symmetry in a 32x32 grid is indeed the 8x8 top-left corner.
    Let's verify:
    If we have F[i][j] for i,j in 0..7.
    Row 0..7, Col 0..7: F
    Row 0..7, Col 24..31: F[:, ::-1] (Horizontal flip of F)
    Row 24..31, Col 0..7: F[::-1, :] (Vertical flip of F)
    Row 24..31, Col 24..31: F[::-1, ::-1] (180 rotation of F)
    
    Wait, does this satisfy Transpose?
    Cell[i][j] == Cell[j][i].
    If i,j in 0..7: F[i][j] == F[j][i]. So F must be symmetric.
    If i in 0..7, j in 24..31: Cell[i][j] = F[i][31-j].
    Cell[j][i] where j is in 24..31, i in 0..7.
    j = 31 - j'. Cell[31-j'][i] = F[31-(31-j')][i] = F[j'][i].
    So we need F[i][31-j] == F[31-j][i].
    Let j' = 31-j. Since j is in 24..31, j' is in 0..7.
    So F[i][j'] == F[j'][i].
    This is just the symmetry of F again.
    
    So the condition is:
    1. F is 8x8.
    2. F is symmetric (F[i][j] == F[j][i]).
    3. The rest of the 32x32 cell is filled by mirroring.
    """
    N = 32
    cell = np.zeros((N, N), dtype=int)
    
    # Top-Left
    cell[0:8, 0:8] = fundamental_8x8
    
    # Top-Right (Horizontal Flip)
    cell[0:8, 24:32] = fundamental_8x8[:, ::-1]
    
    # Bottom-Left (Vertical Flip)
    cell[24:32, 0:8] = fundamental_8x8[::-1, :]
    
    # Bottom-Right (180 Rotation)
    cell[24:32, 24:32] = fundamental_8x8[::-1, ::-1]
    
    # Check middle rows/cols?
    # Rows 8..23 and Cols 8..23 are empty in this construction?
    # NO. The 32x32 grid is fully populated.
    # My mapping above only covers the corners.
    # Let's re-evaluate the symmetry group C4v on a 32x32 grid.
    # The center of the cell is at (15.5, 15.5).
    # The fundamental domain is the region 0 <= x < 16, 0 <= y < 16? No, 4-fold symmetry reduces area by 4.
    # 32x32 / 4 = 256 pixels. So an 8x8 block is correct size.
    # The 8x8 block corresponds to indices [0:8, 0:8].
    # 
    # Let's map indices properly.
    # Let i, j be indices in 0..31.
    # Map to fundamental domain i_f, j_f in 0..7.
    # 
    # Symmetry operations:
    # 1. Identity: (i, j) -> (i, j). If i<8, j<8, use F[i][j].
    # 2. Rot 90 CW: (i, j) -> (j, N-1-i).
    # 3. Rot 180: (i, j) -> (N-1-i, N-1-j).
    # 4. Rot 270: (i, j) -> (N-1-j, i).
    # 5. Flip X: (i, j) -> (N-1-i, j).
    # 6. Flip Y: (i, j) -> (i, N-1-j).
    # 7. Flip Diagonal: (i, j) -> (j, i).
    # 8. Flip Anti-Diag: (i, j) -> (N-1-j, N-1-i).
    
    # To ensure consistency, we define the cell value at (i,j) by mapping it to the fundamental domain.
    # We need a mapping function map(i, j) -> (i_f, j_f) such that all symmetric equivalents map to the same F index.
    
    def get_fundamental_index(i, j):
        # Bring to first quadrant relative to center?
        # Center is 15.5.
        # Let's use distance from center or just fold the indices.
        
        # Fold horizontally: if i >= 16, i = 31 - i
        # Fold vertically: if j >= 16, j = 31 - j
        # This maps to 0..15.
        
        # Now we have 16x16.
        # Apply diagonal symmetry?
        # The group is C4v.
        # If we just use the top-left 8x8, we are assuming the pattern is symmetric under 90 deg rotation AND reflection.
        # 
        # Let's try a simpler approach:
        # The 32x32 cell is composed of 16 blocks of 2x2? No.
        # Let's just iterate over all 1024 pixels and determine their value from the 8x8 fundamental block F.
        
        # For any pixel (i, j):
        # 1. If i >= 16: i = 31 - i
        # 2. If j >= 16: j = 31 - j
        # Now i, j in 0..15.
        # 3. If i > j: swap i, j (Diagonal symmetry)
        # Now i <= j.
        # 4. If i >= 8: 
        #    This part is tricky. The 16x16 quadrant is not just a copy of the 8x8.
        #    Wait, 32/4 = 8. The fundamental domain of C4v on a square grid of size N is N/4 x N/4?
        #    Yes, if N is divisible by 4. 32/4 = 8.
        #    So the 8x8 block F[0:8, 0:8] defines the whole cell IF the symmetry is applied correctly.
        
        # Let's verify the mapping for a specific point.
        # Point (0, 0) -> F[0][0]
        # Point (0, 31) -> Flip Y -> (0, 0) -> F[0][0]
        # Point (31, 0) -> Flip X -> (0, 0) -> F[0][0]
        # Point (31, 31) -> Flip X, Flip Y -> (0, 0) -> F[0][0]
        
        # Point (0, 1) -> F[0][1]
        # Point (1, 0) -> Swap -> (0, 1) -> F[0][1]
        
        # Point (0, 7) -> F[0][7]
        # Point (0, 8) -> ?
        # In a 32x32 grid, index 7 and 8 are adjacent to the center?
        # Center is 15.5.
        # Indices 0..15 are left half. 16..31 are right half.
        # So 0..7 and 8..15 are both in the left half.
        # Does F[0][0..7] cover the whole left-top 16x16 quadrant?
        # No, F is 8x8. The quadrant is 16x16.
        # So F must map to the 16x16 quadrant via some symmetry?
        # No, the fundamental domain of the *unit cell* under the *lattice* symmetry is the whole cell.
        # The *point group* symmetry (C4v) reduces the independent pixels.
        # For a 32x32 cell with C4v symmetry, the number of independent pixels is 32*32 / 8 = 128?
        # Or 32*32 / 4 = 256?
        # C4v has 8 elements.
        # If the center of symmetry is at the cell center (15.5, 15.5), then no pixel is on the axis of rotation (since N is even).
        # So orbits are size 8.
        # Total pixels 1024. 1024 / 8 = 128 independent pixels.
        # An 8x8 block has 64 pixels. So 8x8 is NOT enough.
        # We need a 16x8 block? Or 8x16?
        # Let's use a 16x8 block (top-left quarter of the top-left half?).
        # Actually, let's just use a 16x16 block and enforce symmetry within it?
        # No, 16x16 is 256 pixels. 256 / 8 = 32? No.
        
        # Let's re-read the symmetry requirement:
        # "invariant under the square symmetry group about the cell centre"
        # cell == cell[::-1] row-wise (Flip Y)
        # cell == cell[::-1] col-wise (Flip X)
        # cell == cell.T (Transpose)
        
        # This implies:
        # 1. Flip X: cell[i][j] == cell[i][31-j]
        # 2. Flip Y: cell[i][j] == cell[31-i][j]
        # 3. Transpose: cell[i][j] == cell[j][i]
        
        # From 1 and 2: cell[i][j] == cell[31-i][31-j] (180 Rot)
        # From 3 and 1: cell[i][j] == cell[j][31-i]
        
        # Let's define the independent region.
        # We can pick i in 0..15 (top half).
        # For a fixed i, j can be 0..15 (left half).
        # Due to Transpose, cell[i][j] == cell[j][i].
        # So we only need to define cell[i][j] for i <= j?
        # If i <= j, and i, j in 0..15.
        # Number of such pairs: 16*17/2 = 136.
        # But we also have Flip X/Y constraints?
        # If we define the top-left 16x16 quadrant Q, then:
        # cell[i][j] for i,j in 0..15 is Q[i][j].
        # cell[i][31-j] = Q[i][j] (Flip X)
        # cell[31-i][j] = Q[i][j] (Flip Y)
        # cell[31-i][31-j] = Q[i][j] (180)
        
        # Now check Transpose for the whole cell.
        # cell[i][j] == cell[j][i].
        # Case 1: i,j in 0..15.
        # Q[i][j] == Q[j][i]. So Q must be symmetric.
        # Case 2: i in 0..15, j in 16..31.
        # cell[i][j] = Q[i][31-j] (Flip X). Let j' = 31-j. j' in 0..15.
        # cell[j][i] = cell[31-j'][i]. Since j' in 0..15, 31-j' in 16..31.
        # cell[31-j'][i] = Q[j'][i] (Flip Y).
        # So we need Q[i][j'] == Q[j'][i].
        # This is the same symmetry condition on Q.
        
        # So the constraint is:
        # 1. Define Q (16x16).
        # 2. Q must be symmetric: Q[i][j] == Q[j][i].
        # 3. The 32x32 cell is built by mirroring Q.
        
        # Independent pixels in Q: 16*17/2 = 136.
        # This is a large search space for hill climbing, but manageable.
        
        # However, we can reduce further?
        # No, 136 bits is fine for a fast solver.
        
        # Let's implement the builder.
        Q = np.zeros((16, 16), dtype=int)
        # fundamental_8x8 is actually not enough. We need 16x16.
        # Let's change the interface: fundamental is 16x16 symmetric.
        pass

    # Revised Plan:
    # The "fundamental" object is a 16x16 matrix Q.
    # We only store the upper triangle (including diagonal) to ensure symmetry.
    # Size: 136 bits.
    
    # Let's adjust the function to take a 16x16 array (or 136 list) and build the 32x32.
    
    # For simplicity in the optimizer, let's just keep a 16x16 array Q, 
    # and enforce Q[i][j] = Q[j][i] after every update.
    
    # Build cell from Q:
    cell[0:16, 0:16] = Q
    cell[0:16, 16:32] = Q[:, ::-1]  # Flip X
    cell[16:32, 0:16] = Q[::-1, :]  # Flip Y
    cell[16:32, 16:32] = Q[::-1, ::-1] # 180 Rot
    
    # Verify Transpose:
    # cell[i][j] vs cell[j][i]
    # If i,j < 16: Q[i][j] vs Q[j][i]. Equal if Q symmetric.
    # If i < 16, j >= 16:
    #   cell[i][j] = Q[i][31-j]
    #   cell[j][i] = cell[31-(31-j)][i] = cell[31-j'][i] where j'=31-j.
    #   Since j' < 16, 31-j' >= 16.
    #   cell[31-j'][i] = Q[j'][i] (Flip Y on row 31-j' which is >= 16? No.
    #   Row index is 31-j'. Col index is i (<16).
    #   This falls in Bottom-Left block: cell[16:32, 0:16] = Q[::-1, :].
    #   So cell[31-j'][i] = Q[31-(31-j')][i] = Q[j'][i].
    #   So we need Q[i][j'] == Q[j'][i]. Equal if Q symmetric.
    
    return cell

def _compute_gap_pwf(cell, eps_high=12.0, N_pw=25, k_points=None):
    """
    Higher fidelity proxy.
    N_pw=25 corresponds to G vectors up to |G| ~ 2.something.
    """
    N = len(cell)
    cell_arr = np.array(cell, dtype=float)
    inv_eps = 1.0 / (1.0 + (eps_high - 1.0) * cell_arr)
    
    # FFT coefficients
    fft_inv_eps = np.fft.fft2(inv_eps) / (N * N)
    
    # G vectors: Circle of radius R such that count is ~ N_pw
    # For N_pw=25, R=2 gives 25 vectors (h,k in -2..2).
    # Let's use R=2.
    R = 2
    g_set = [(h, k) for h in range(-R, R+1) for k in range(-R, R+1) if h*h + k*k <= R*R]
    # This is 25 vectors.
    
    N_g = len(g_set)
    
    # K points in Irreducible BZ
    # Triangle: (0,0), (pi,0), (pi/2, pi/2)
    # Sample:
    k_pts = [
        (0.0, 0.0),
        (np.pi/4, 0.0),
        (np.pi/2, 0.0),
        (np.pi/4, np.pi/4),
        (np.pi/2, np.pi/2),
        (np.pi/3, np.pi/6), # Midpoint of hypotenuse?
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_pts:
        K = np.array([(kx + h, ky + k) for h, k in g_set])
        
        H = np.zeros((N_g, N_g), dtype=complex)
        for i in range(N_g):
            for j in range(N_g):
                Gh = g_set[i][0] - g_set[j][0]
                Gk = g_set[i][1] - g_set[j][1]
                idx_h = Gh % N
                idx_k = Gk % N
                coeff = fft_inv_eps[idx_h, idx_k]
                
                dKx = K[i][0] - K[j][0]
                dKy = K[i][1] - K[j][1]
                dist_sq = dKx*dKx + dKy*dKy
                H[i, j] = dist_sq * coeff
        
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        eigvals = np.sort(eigvals)
        
        # Bands 4 and 5 (0-indexed 3 and 4)
        if len(eigvals) >= 5:
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            if w4_sq > 1e-6 and w5_sq > 1e-6:
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
    
    if min_omega_5 == np.inf or max_omega_4 == -np.inf:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 20s safety margin
    
    # Initialize Q (16x16 symmetric)
    # Start with a random or structured pattern
    # Let's start with a "checkerboard" or "random" to explore.
    # Random initialization is often better for topology optimization.
    
    Q = np.zeros((16, 16), dtype=int)
    # Fill upper triangle randomly
    for i in range(16):
        for j in range(i, 16):
            Q[i, j] = 1 if random.random() < 0.5 else 0
            Q[j, i] = Q[i, j]
    
    best_Q = Q.copy()
    best_score = _compute_gap_pwf(_build_symmetric_cell(Q))
    
    # Simulated Annealing / Hill Climbing
    # Parameters
    T = 0.01
    cooling = 0.99
    steps_per_temp = 50
    
    current_score = best_score
    current_Q = Q.copy()
    
    # Precompute indices for upper triangle
    upper_indices = [(i, j) for i in range(16) for j in range(i, 16)]
    
    while time.time() < deadline:
        # Randomly select a pixel in the fundamental domain to flip
        idx = random.randint(0, len(upper_indices) - 1)
        i, j = upper_indices[idx]
        
        # Flip
        current_Q[i, j] = 1 - current_Q[i, j]
        current_Q[j, i] = current_Q[i, j]
        
        # Evaluate
        cell = _build_symmetric_cell(current_Q)
        new_score = _compute_gap_pwf(cell)
        
        # Accept/Reject
        if new_score > current_score or (random.random() < np.exp((new_score - current_score) / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_Q = current_Q.copy()
        else:
            # Revert
            current_Q[i, j] = 1 - current_Q[i, j]
            current_Q[j, i] = current_Q[i, j]
            
        # Cooling
        T *= cooling
        if T < 0.001:
            T = 0.01 # Reheat occasionally? Or just keep going with hill climbing.
            # Let's switch to pure hill climbing if T is low.
    
    # Final check
    if best_Q is None:
        # Fallback
        Q = np.ones((16, 16), dtype=int)
        best_Q = Q
    
    return _build_symmetric_cell(best_Q).tolist()
# EVOLVE-BLOCK-END
