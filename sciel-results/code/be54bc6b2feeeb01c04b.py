"""Photonic crystal unit cell design for maximum TM band gap (bands 4-5)."""
import time
import numpy as np
from scipy.linalg import eigh

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing TM band gap (bands 4-5)."""
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Symmetry: cell must be invariant under 180° rotation and transpose
    # This means cell[i][j] = cell[size-1-i][size-1-j] = cell[j][i]
    # So we only need to define the top-left quadrant with diagonal constraint
    
    # Build symmetry mask: for each pixel (i,j), determine its orbit
    # Under D4 (square symmetry about center):
    # - 180° rotation: (i,j) -> (size-1-i, size-1-j)
    # - Transpose: (i,j) -> (j,i)
    # The fundamental domain is the top-left quadrant where i <= size//2-1 and j <= i
    # But we need to be careful about the center
    
    half = size // 2  # 16
    
    # Define the fundamental region: pixels (i,j) where i < half and j <= i
    # Actually for 32x32 with center at (15.5, 15.5):
    # 180° rotation maps (i,j) to (31-i, 31-j)
    # Transpose maps (i,j) to (j,i)
    # The fundamental domain: i in [0,15], j in [0,i] (top-left triangle including diagonal)
    # But we also need to handle the case where j > i by using transpose
    
    def get_fundamental_coords():
        """Get coordinates of the fundamental domain."""
        coords = []
        for i in range(half):
            for j in range(i + 1):
                coords.append((i, j))
        return coords
    
    def apply_symmetry(cell_fundamental):
        """Apply symmetry to generate full cell from fundamental domain."""
        cell = np.zeros((size, size), dtype=np.int8)
        for (i, j), val in zip(get_fundamental_coords(), cell_fundamental):
            # 4 symmetry operations generate 4 pixels (or fewer if on symmetry axes)
            # (i,j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
            # But also transpose: (j,i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            # The full orbit under D4 has up to 8 elements
            
            # Let's enumerate all distinct positions in the orbit
            positions = set()
            positions.add((i, j))
            positions.add((i, size - 1 - j))
            positions.add((size - 1 - i, j))
            positions.add((size - 1 - i, size - 1 - j))
            positions.add((j, i))
            positions.add((j, size - 1 - i))
            positions.add((size - 1 - j, i))
            positions.add((size - 1 - j, size - 1 - i))
            
            for (pi, pj) in positions:
                if 0 <= pi < size and 0 <= pj < size:
                    cell[pi, pj] = val
        
        return cell
    
    def cell_to_nested_list(cell):
        """Convert numpy array to nested list."""
        return [[int(cell[i, j]) for j in range(size)] for i in range(size)]
    
    # PWE solver for band gap evaluation
    def compute_band_gap(cell, n_pw=17):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses plane-wave expansion with n_pw x n_pw basis.
        """
        # Fourier coefficients of epsilon
        # epsilon(x,y) = sum_{G} epsilon_G * exp(iG·r)
        # For a pixelated structure, epsilon_G = (1/N^2) sum_{i,j} epsilon(i,j) * exp(-iG·(i,j))
        
        # Generate reciprocal lattice vectors
        # G = (m, n) * (2π/L) where L=1 (lattice constant)
        # For n_pw x n_pw, m,n range from -(n_pw-1)//2 to (n_pw-1)//2
        
        m_max = (n_pw - 1) // 2
        G_vectors = []
        for m in range(-m_max, m_max + 1):
            for n in range(-m_max, m_max + 1):
                G_vectors.append((m, n))
        n_G = len(G_vectors)
        
        # Compute Fourier coefficients
        # epsilon_G = (1/size^2) * sum_{i,j} cell[i,j] * exp(-2πi*(m*i + n*j)/size)
        # This is a 2D DFT
        
        cell_float = cell.astype(np.float64)
        # Use FFT for efficiency
        eps_fourier = np.fft.fft2(cell_float) / (size * size)
        
        # eps_fourier[m+n_max, n+n_max] = epsilon_{(m,n)}
        # But we need to reorder for our G_vectors list
        
        # Map from (m,n) to index in G_vectors
        G_to_idx = {}
        for idx, (m, n) in enumerate(G_vectors):
            G_to_idx[(m, n)] = idx
        
        # Extract Fourier coefficients in the order of G_vectors
        eps_G = np.zeros(n_G, dtype=np.complex128)
        for idx, (m, n) in enumerate(G_vectors):
            eps_G[idx] = eps_fourier[m + m_max, n + m_max]
        
        # For TM mode: (k - G)^2 * epsilon_{G-G'} * A_{G'} = omega^2 * A_G
        # The eigenvalue problem: sum_{G'} (k-G)^2 * delta_{G,G'} * eps_{G-G'} A_{G'} = omega^2 A_G
        # Actually for TM: the matrix is [(k-G)^2] * [eps_{G-G'}]
        # where eps_{G-G'} is the Fourier coefficient of 1/epsilon (for TM)
        
        # Wait, for TM polarization in 2D photonic crystals:
        # The eigenvalue equation involves epsilon^{-1} in Fourier space
        # Let me reconsider...
        
        # For TM (H field perpendicular to plane):
        # (k-G)^2 * sum_{G'} (1/epsilon)_{G-G'} * A_{G'} = omega^2 A_G
        # So we need the Fourier coefficients of 1/epsilon
        
        # epsilon = eps_low + (eps_high - eps_low) * cell
        # 1/epsilon = 1/(eps_low + (eps_high - eps_low) * cell)
        
        eps_field = eps_low + (eps_high - eps_low) * cell_float
        inv_eps_field = 1.0 / eps_field
        
        # Fourier coefficients of 1/epsilon
        inv_eps_fourier = np.fft.fft2(inv_eps_field) / (size * size)
        inv_eps_G = np.zeros(n_G, dtype=np.complex128)
        for idx, (m, n) in enumerate(G_vectors):
            inv_eps_G[idx] = inv_eps_fourier[m + m_max, n + m_max]
        
        # Now solve for several k-points in the irreducible BZ
        # Irreducible BZ for square lattice: kx >= 0, ky >= 0, kx >= ky
        # Key k-points: Gamma (0,0), M (π,0), X (π/2, π/2) in units of 2π/L
        # Let's use a set of k-points
        
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # M
            (0.5, 0.5),      # X
            (0.25, 0.0),
            (0.25, 0.25),
            (0.4, 0.1),
            (0.3, 0.3),
            (0.1, 0.1),
            (0.45, 0.05),
            (0.2, 0.4),
        ]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # k vector in units of 2π/L
            k = np.array([kx, ky]) * 2 * np.pi
            
            # Build the eigenvalue matrix
            # H[G, G'] = (k-G)^2 * inv_eps_{G-G'}
            # where G, G' are in G_vectors
            
            H = np.zeros((n_G, n_G), dtype=np.complex128)
            
            for i, Gi in enumerate(G_vectors):
                ki = k - np.array(Gi) * 2 * np.pi
                ki_sq = ki[0]**2 + ki[1]**2
                
                for j, Gj in enumerate(G_vectors):
                    # G - G'
                    dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                    if dG in G_to_idx:
                        idx_dG = G_to_idx[dG]
                    else:
                        # This shouldn't happen if G_vectors is closed under subtraction
                        # But for safety, skip
                        continue
                    H[i, j] = ki_sq * inv_eps_G[idx_dG]
            
            # Solve eigenvalue problem
            # H is Hermitian (should be), use eigh
            try:
                eigenvalues, _ = eigh(H, subset_by_index=[3, 4])  # 4th and 5th bands (0-indexed: 3,4)
                omega4 = np.sqrt(eigenvalues[0])
                omega5 = np.sqrt(eigenvalues[1])
            except Exception:
                try:
                    eigenvalues, _ = eigh(H)
                    if len(eigenvalues) < 5:
                        continue
                    omega4 = np.sqrt(eigenvalues[3])
                    omega5 = np.sqrt(eigenvalues[4])
                except Exception:
                    continue
            
            if omega4 > max_omega4:
                max_omega4 = omega4
            if omega5 < min_omega5:
                min_omega5 = omega5
        
        if max_omega4 == 0.0 or min_omega5 == float('inf'):
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap == 0:
            return 0.0
        
        return gap / midgap * 2.0  # gap-midgap ratio: 2*(gap)/(midgap*2) = gap/midgap... 
        # Actually: 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
        # = 2*gap/(2*midgap) = gap/midgap
        # So just return gap/midgap
    
    # Better gap ratio calculation
    def compute_gap_ratio(cell, n_pw=17):
        """Compute 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)."""
        size_local = size
        m_max = (n_pw - 1) // 2
        G_vectors = [(m, n) for m in range(-m_max, m_max + 1) for n in range(-m_max, m_max + 1)]
        n_G = len(G_vectors)
        G_to_idx = {(m, n): idx for idx, (m, n) in enumerate(G_vectors)}
        
        cell_float = cell.astype(np.float64)
        eps_field = eps_low + (eps_high - eps_low) * cell_float
        inv_eps_field = 1.0 / eps_field
        
        inv_eps_fourier = np.fft.fft2(inv_eps_field) / (size_local * size_local)
        inv_eps_G = np.zeros(n_G, dtype=np.complex128)
        for idx, (m, n) in enumerate(G_vectors):
            inv_eps_G[idx] = inv_eps_fourier[m + m_max, n + m_max]
        
        k_points = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
            (0.25, 0.0), (0.25, 0.25), (0.4, 0.1),
            (0.3, 0.3), (0.1, 0.1), (0.45, 0.05), (0.2, 0.4),
            (0.15, 0.35), (0.35, 0.15),
        ]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            k = np.array([kx, ky]) * 2 * np.pi
            
            H = np.zeros((n_G, n_G), dtype=np.complex128)
            for i, Gi in enumerate(G_vectors):
                ki = k - np.array(Gi) * 2 * np.pi
                ki_sq = ki[0]**2 + ki[1]**2
                for j, Gj in enumerate(G_vectors):
                    dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                    if dG in G_to_idx:
                        H[i, j] = ki_sq * inv_eps_G[G_to_idx[dG]]
            
            try:
                eigenvalues, _ = eigh(H, subset_by_index=[3, 4])
                omega4 = np.sqrt(max(eigenvalues[0], 0))
                omega5 = np.sqrt(max(eigenvalues[1], 0))
            except Exception:
                try:
                    eigenvalues, _ = eigh(H)
                    if len(eigenvalues) < 5:
                        continue
                    omega4 = np.sqrt(max(eigenvalues[3], 0))
                    omega5 = np.sqrt(max(eigenvalues[4], 0))
                except Exception:
                    continue
            
            if omega4 > max_omega4:
                max_omega4 = omega4
            if omega5 < min_omega5:
                min_omega5 = omega5
        
        if max_omega4 == 0.0 or min_omega5 == float('inf'):
            return 0.0
        
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    # Optimization: start with a few initial designs and refine
    
    fundamental_coords = get_fundamental_coords()
    n_fund = len(fundamental_coords)
    
    best_cell = None
    best_score = -float('inf')
    
    # Initial designs to try
    initial_designs = []
    
    # 1. Solid dielectric (all 1s)
    initial_designs.append([1] * n_fund)
    
    # 2. Air (all 0s)
    initial_designs.append([0] * n_fund)
    
    # 3. Circular rod
    for idx, (i, j) in enumerate(fundamental_coords):
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        initial_designs[idx] = 1 if x*x + y*y < 0.2**2 else 0
    initial_designs = [initial_designs]  # Wait, this overwrote. Let me redo.
    
    initial_designs = []
    
    # Design 1: All dielectric
    d1 = [1] * n_fund
    initial_designs.append(d1)
    
    # Design 2: All air
    d2 = [0] * n_fund
    initial_designs.append(d2)
    
    # Design 3: Circular rod
    d3 = []
    for (i, j) in fundamental_coords:
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        d3.append(1 if x*x + y*y < 0.2**2 else 0)
    initial_designs.append(d3)
    
    # Design 4: Square rod
    d4 = []
    for (i, j) in fundamental_coords:
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        d4.append(1 if abs(x) < 0.15 and abs(y) < 0.15 else 0)
    initial_designs.append(d4)
    
    # Design 5: Ring (dielectric ring)
    d5 = []
    for (i, j) in fundamental_coords:
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        r = np.sqrt(x*x + y*y)
        d5.append(1 if 0.1 < r < 0.25 else 0)
    initial_designs.append(d5)
    
    # Design 6: Checkerboard-like pattern
    d6 = []
    for (i, j) in fundamental_coords:
        d6.append(1 if (i + j) % 2 == 0 else 0)
    initial_designs.append(d6)
    
    # Design 7: Cross pattern
    d7 = []
    for (i, j) in fundamental_coords:
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        d7.append(1 if abs(x) < 0.1 or abs(y) < 0.1 else 0)
    initial_designs.append(d7)
    
    # Evaluate initial designs
    for d in initial_designs:
        if time.time() > deadline - 5:
            break
        cell = apply_symmetry(d)
        score = compute_gap_ratio(cell, n_pw=13)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search: flip random pixels in the fundamental domain
    if best_cell is not None:
        current_fund = []
        for (i, j) in fundamental_coords:
            current_fund.append(int(best_cell[i, j]))
        
        current_score = best_score
        
        # Simulated annealing
        temp = 0.1
        cooling = 0.995
        max_iters = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if temp < 0.001:
                break
            
            # Pick a random fundamental pixel to flip
            idx = np.random.randint(n_fund)
            new_fund = current_fund[:]
            new_fund[idx] = 1 - new_fund[idx]
            
            new_cell = apply_symmetry(new_fund)
            new_score = compute_gap_ratio(new_cell, n_pw=13)
            
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_fund = new_fund
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = new_cell.copy()
            
            temp *= cooling
    
    # Convert to nested list
    if best_cell is None:
        # Fallback: circular rod
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    cell[i, j] = 1
        best_cell = cell
    
    return cell_to_nested_list(best_cell)
# EVOLVE-BLOCK-END
