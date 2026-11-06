import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s budget

    # Fundamental domain: 8x8 top-left quadrant, symmetric about diagonal
    # q[i][j] for i<=j
    q = np.zeros((8, 8), dtype=int)
    
    # Initialize with a centered square rod
    # Center of 32x32 is (15.5, 15.5)
    # Top-left 8x8 covers rows 0-7, cols 0-7
    # A square rod of size 6x6 centered would cover rows 13-18, cols 13-18
    # In the top-left quadrant (0-7), this corresponds to rows 7-7 (since 13 is in the bottom-right mirror of 18? No)
    # Let's map carefully.
    # Cell index (i,j) in 0..31.
    # Quadrant index (r,c) in 0..7.
    # r = min(i, 31-i)
    # c = min(j, 31-j)
    # If r>c, swap r,c due to q symmetry? No, q[r][c] = q[c][r].
    # We store q[r][c] for r<=c.
    
    # Let's initialize q to 0 (air).
    # Add a dielectric rod.
    # Radius 0.2 lattice constants = 3.2 pixels.
    # In the center, the rod extends from 15.5-3.2=12.3 to 15.5+3.2=18.7.
    # So rows 13-18, cols 13-18 are dielectric.
    # For a point (i,j) in the rod:
    # r = min(i, 31-i). If i=13, r=13? No, min(13, 18)=13.
    # Wait, the quadrant is 0..7.
    # i=13 -> min(13, 18) = 13. This is outside 0..7.
    # Ah, the symmetry is about the center.
    # The top-left quadrant is rows 0..15? No.
    # 32 pixels. Center is between 15 and 16.
    # Symmetry: i <-> 31-i.
    # So the unique rows are 0..15.
    # But we also have x<->y symmetry.
    # So the unique region is the triangle where i<=15, j<=15, i<=j?
    # No, the fundamental domain of C4v on a 32x32 grid (even size) is a bit tricky.
    # Let's just use the 16x16 top-left quadrant and enforce j>=i?
    # Or simpler: use the 8x8 top-left corner?
    # If I use 8x8, I am assuming the pattern repeats or mirrors in a way that 8 is the fundamental period?
    # No, 32/4 = 8.
    # The symmetry group maps the 32x32 cell to 4 quadrants of 16x16?
    # No, 32/2 = 16.
    # The symmetry i<->31-i maps 0..15 to 15..0.
    # So the unique rows are 0..15.
    # Similarly cols 0..15.
    # So the unique block is 16x16.
    # With x<->y symmetry, we only need the triangle i<=j in the 16x16 block.
    # Size: 16*17/2 = 136 bits.
    
    # Let's use the 16x16 triangle representation.
    # q[i][j] for 0<=i<=j<16.
    
    def build_cell(q_tri):
        cell = np.zeros((32, 32), dtype=int)
        for i in range(16):
            for j in range(i, 16):
                val = q_tri[i][j]
                # Symmetries:
                # (i, j)
                cell[i][j] = val
                # (i, 31-j)
                cell[i][31-j] = val
                # (31-i, j)
                cell[31-i][j] = val
                # (31-i, 31-j)
                cell[31-i][31-j] = val
                # Rotation 90: (j, 31-i)
                # This is covered by the above if we ensure consistency?
                # Let's check:
                # (j, 31-i) -> row j, col 31-i.
                # If j>=i, then (j, 31-i) is in the top-right or bottom-right?
                # j is in 0..15. 31-i is in 16..31.
                # So it's in the right half.
                # The value at (j, 31-i) should be q_tri[j][31-(31-i)]? No.
                # The value at (r, c) is determined by r' = min(r, 31-r), c' = min(c, 31-c).
                # Then if r' > c', swap r', c'.
                # Then value is q_tri[r'][c'].
                
                # Let's just fill the cell by iterating all 32x32 and mapping to the triangle.
                # This is safer.
        # Re-do build_cell properly
        cell = np.zeros((32, 32), dtype=int)
        for i in range(32):
            for j in range(32):
                r = min(i, 31-i)
                c = min(j, 31-j)
                if r > c:
                    r, c = c, r
                cell[i][j] = q_tri[r][c]
        return cell

    # Initialize q_tri with a centered square rod
    q_tri = np.zeros((16, 16), dtype=int)
    # Rod from 12 to 19 (size 8) centered?
    # Center 15.5. 15.5-3.5=12, 15.5+3.5=19.
    # So indices 12,13,14,15,16,17,18,19.
    # In the triangle (r<=c, r,c in 0..15):
    # For i in 12..15, j in 12..15:
    # r=min(i,31-i). If i=12, r=12. If i=15, r=15.
    # c=min(j,31-j).
    # We need to set q_tri[r][c]=1 for the rod.
    for i in range(12, 16):
        for j in range(12, 16):
            r = min(i, 31-i)
            c = min(j, 31-j)
            if r > c: r, c = c, r
            q_tri[r][c] = 1

    # K-points
    # Irreducible BZ: Triangle (0,0) - (0.5,0) - (0.5,0.5)
    # Generate 30 points
    k_points = []
    # Gamma to X: (k, 0) for k in [0, 0.5]
    for k in np.linspace(0, 0.5, 6):
        k_points.append((k, 0))
    # X to M: (0.5, k) for k in [0, 0.5]
    for k in np.linspace(0, 0.5, 6):
        k_points.append((0.5, k))
    # Gamma to M: (k, k) for k in [0, 0.5]
    for k in np.linspace(0, 0.5, 6):
        k_points.append((k, k))
    # Interior points
    interior = [
        (0.25, 0.1), (0.3, 0.2), (0.4, 0.3),
        (0.1, 0.1), (0.2, 0.2), (0.3, 0.1),
        (0.15, 0.05), (0.45, 0.4), (0.25, 0.25), (0.1, 0.05)
    ]
    k_points.extend(interior)
    # Ensure we have exactly 30
    k_points = k_points[:30]

    # Precompute G vectors for M=11
    M = 11
    G_vectors = []
    for i in range(-M//2, M//2 + 1):
        for j in range(-M//2, M//2 + 1):
            G_vectors.append((i, j))
    N = len(G_vectors)
    G_vectors = np.array(G_vectors)

    def evaluate(q_tri, M_eval):
        # Build cell
        cell = build_cell(q_tri)
        # Compute 1/epsilon
        inv_eps = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                if cell[i][j] == 1:
                    inv_eps[i][j] = 1.0 / eps_high
                else:
                    inv_eps[i][j] = 1.0 / eps_low
        
        # Compute Fourier coefficients of 1/epsilon
        # Use FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Map G vectors to FFT indices
        # FFT freqs: 0, 1, ..., 15, -16, ..., -1
        # Our G vectors are -5..5 for M=11.
        # We need to extract the correct coefficients.
        # fft2 gives frequencies 0..31.
        # Index for G=(i,j):
        # i_fft = i % 32
        # j_fft = j % 32
        
        coeff = np.zeros((N, N), dtype=complex)
        for idx_g, (gi, gj) in enumerate(G_vectors):
            i_fft = gi % 32
            j_fft = gj % 32
            for idx_gp, (gpi, gpj) in enumerate(G_vectors):
                igp = gi - gpi
                jgp = gj - gpj
                i_fft_gp = igp % 32
                j_fft_gp = jgp % 32
                coeff[idx_g, idx_gp] = fft_inv_eps[i_fft, j_fft_gp]
                
        # Evaluate bands
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix
            mat = np.zeros((N, N), dtype=complex)
            for i in range(N):
                ki = kx + G_vectors[i][0]
                kj = ky + G_vectors[i][1]
                for j in range(N):
                    kj_ = ky + G_vectors[j][0]
                    ki_ = kx + G_vectors[j][1]
                    # Term: (ki * kj_) + (kj * ki_) * coeff[i,j]
                    # Wait, the dot product is (k+G_i) . (k+G_j)
                    dot = ki * G_vectors[j][0] + kj * G_vectors[j][1] 
                    # Correction: (kx+Gi) * (kx+Gj) + (ky+Gj) * (ky+Gj)
                    dot = (kx + G_vectors[i][0]) * (kx + G_vectors[j][0]) + (ky + G_vectors[i][1]) * (ky + G_vectors[j][1])
                    mat[i, j] = dot * coeff[i, j]
            
            # Hermitian matrix, use eigh
            eigs = np.linalg.eigvalsh(mat)
            # Take 5 smallest
            eigs = eigs[:5]
            # omega^2 = eigs
            omegas = np.sqrt(eigs)
            
            if omegas[4] < min_omega_5:
                min_omega_5 = omegas[4]
            if omegas[3] > max_omega_4:
                max_omega_4 = omegas[3]
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Initial evaluation
    current_q = q_tri.copy()
    current_score = evaluate(current_q, M)
    best_q = current_q.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1
    cooling_rate = 0.999
    step = 0
    
    while time.time() < deadline:
        # Pick a random position in the triangle
        i = np.random.randint(0, 16)
        j = np.random.randint(i, 16)
        
        # Flip
        new_q = current_q.copy()
        new_q[i][j] = 1 - new_q[i][j]
        
        # Evaluate
        new_score = evaluate(new_q, M)
        
        # Accept
        if new_score > current_score:
            current_q = new_q
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_q = current_q.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if np.random.rand() < prob:
                    current_q = new_q
                    current_score = new_score
        
        T *= cooling_rate
        step += 1
        
        # If time is getting close, do a final high-precision check on best_q
        if time.time() > deadline - 10.0:
            break
            
    # Final verification with higher M if time permits
    if time.time() < deadline - 5.0:
        final_score = evaluate(best_q, 17)
        # If final score is better, we are happy.
        # We just return best_q.
        
    # Construct final cell
    final_cell = build_cell(best_q)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
