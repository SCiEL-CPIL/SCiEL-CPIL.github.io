import numpy as np
import time
import sys

# Constants
SIZE = 32
EPS_LOW = 1.0
EPS_HIGH = 12.0
LOWER_BAND = 4
POLARIZATION = 'TM'

# Reciprocal lattice vectors
# For a square lattice with lattice constant a=1, G = (2*pi*nx, 2*pi*ny)
# We use a cutoff such that we have ~289 vectors.
# Square box: nx, ny in [-8, 8] -> 17*17 = 289 vectors.
G_CUTOFF = 8
G_VECTORS = []
for nx in range(-G_CUTOFF, G_CUTOFF + 1):
    for ny in range(-G_CUTOFF, G_CUTOFF + 1):
        G_VECTORS.append((2 * np.pi * nx, 2 * np.pi * ny))
N_G = len(G_VECTORS)

# K-points in the irreducible Brillouin zone
# Triangle: 0 <= ky <= kx <= pi
# We generate 30 points.
K_POINTS = []
# Add high symmetry points
K_POINTS.append((0.0, 0.0))
K_POINTS.append((np.pi, 0.0))
K_POINTS.append((np.pi, np.pi))
K_POINTS.append((np.pi/2, 0.0))
K_POINTS.append((np.pi, np.pi/2))
K_POINTS.append((np.pi/2, np.pi/2))
# Add grid points
# Divide the triangle into a grid
N_GRID = 5
for i in range(N_GRID):
    for j in range(N_GRID - i):
        # Map i, j to kx, ky
        # kx ranges from 0 to pi, ky ranges from 0 to kx
        # Let's just use a simple grid in the bounding box and filter
        pass

# Better K-point generation:
def generate_k_points(n=30):
    pts = []
    # High symmetry
    pts.append((0.0, 0.0))
    pts.append((np.pi, 0.0))
    pts.append((np.pi, np.pi))
    pts.append((np.pi/2, 0.0))
    pts.append((np.pi, np.pi/2))
    pts.append((np.pi/2, np.pi/2))
    pts.append((np.pi/4, np.pi/4))
    pts.append((3*np.pi/4, np.pi/4))
    pts.append((3*np.pi/4, np.pi/2))
    pts.append((np.pi/2, np.pi/4))
    
    # Add random points in the triangle
    while len(pts) < n:
        kx = np.random.uniform(0, np.pi)
        ky = np.random.uniform(0, kx)
        pts.append((kx, ky))
    return pts[:n]

K_POINTS = generate_k_points(30)

# Pre-compute dot products for each K-point
# dot_prods[k_idx][g_idx][g_prime_idx] = (k + G) . (k + G')
DOT_PRODS = []
for k in K_POINTS:
    kp = np.array(k)
    g_array = np.array(G_VECTORS)
    # (k + G) for all G
    kg = kp[None, :] + g_array
    # Dot product matrix
    dp = np.einsum('ij,ij->i', kg, kg) # This is |k+G|^2, not (k+G).(k+G')
    # We need (k+G) . (k+G')
    # kg is (N_G, 2)
    dp_matrix = kg @ kg.T
    DOT_PRODS.append(dp_matrix)

def get_independent_indices():
    """Yields (i, j) for the fundamental domain."""
    for i in range(SIZE // 2):
        for j in range(i + 1):
            yield i, j

def apply_symmetry(cell, independent_values):
    """
    Fills the full 32x32 cell from a list of independent values.
    independent_values is a list of length 136, corresponding to the order of get_independent_indices.
    """
    full_cell = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    idx = 0
    for i, j in get_independent_indices():
        v = independent_values[idx]
        idx += 1
        # Symmetries:
        # (i, j)
        full_cell[i][j] = v
        # Vertical reflection: j -> SIZE-1-j
        full_cell[i][SIZE - 1 - j] = v
        # Horizontal reflection: i -> SIZE-1-i
        full_cell[SIZE - 1 - i][j] = v
        # 180 rotation: i -> SIZE-1-i, j -> SIZE-1-j
        full_cell[SIZE - 1 - i][SIZE - 1 - j] = v
        # Diagonal reflection: i <-> j
        full_cell[j][i] = v
        # Diagonal + Vertical: (j, SIZE-1-i)
        full_cell[j][SIZE - 1 - i] = v
        # Diagonal + Horizontal: (SIZE-1-j, i)
        full_cell[SIZE - 1 - j][i] = v
        # Diagonal + 180: (SIZE-1-j, SIZE-1-i)
        full_cell[SIZE - 1 - j][SIZE - 1 - i] = v
    return full_cell

def compute_chi(cell):
    """
    Computes the Fourier coefficients chi_G of 1/epsilon(r).
    """
    # cell is 32x32
    # r_x = (i + 0.5) / SIZE, r_y = (j + 0.5) / SIZE
    # Wait, the unit cell is [0,1] x [0,1].
    # Pixel centers are at (i+0.5)/32, (j+0.5)/32.
    
    chi = np.zeros(N_G, dtype=complex)
    eps_inv = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            eps_inv[i, j] = 1.0 / EPS_HIGH if cell[i][j] == 1 else 1.0 / EPS_LOW
    
    # Vectorized computation
    x = (np.arange(SIZE) + 0.5) / SIZE
    y = (np.arange(SIZE) + 0.5) / SIZE
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for g_idx, (gx, gy) in enumerate(G_VECTORS):
        # exp(-i (gx*X + gy*Y))
        phase = np.exp(-1j * (gx * X + gy * Y))
        chi[g_idx] = np.sum(eps_inv * phase) / (SIZE * SIZE)
        
    return chi

def compute_gap(cell):
    """
    Computes the gap between band 4 and 5.
    """
    chi = compute_chi(cell)
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for k_idx, k in enumerate(K_POINTS):
        dp_matrix = DOT_PRODS[k_idx]
        
        # Build the matrix A
        # A_{G, G'} = dp_matrix[G, G'] * chi_{G-G'}
        # We need to access chi_{G-G'} efficiently.
        # G indices are 0..N_G-1.
        # We need a mapping from G-vector difference to index.
        # Since G_VECTORS is a list of tuples, we can create a dictionary.
        
        # Pre-compute G difference indices?
        # For each G, G', we need the index of G-G' in G_VECTORS.
        # This is O(N_G^2) per k-point. 289^2 = 83521.
        # 30 k-points -> 2.5 million operations. This is fine.
        
        # To speed up, we can pre-compute the index map for G-G'.
        # But G-G' might not be in G_VECTORS if it's outside the cutoff.
        # However, if G and G' are in the box [-8,8], their difference is in [-16,16].
        # Our chi is only defined for G in [-8,8].
        # If G-G' is outside, chi_{G-G'} = 0?
        # No, the convolution sum is over ALL G.
        # But we truncated the basis.
        # The standard PWM uses the Fourier coefficients of the periodic function.
        # If we truncate the basis, we effectively ignore high-frequency components of chi.
        # This is an approximation.
        # The prompt says "289-plane-wave Galerkin method using the exact Fourier coefficients".
        # This implies we should use the exact chi for the differences that appear.
        # If G-G' is outside the cutoff, we should still include it if it's non-zero.
        # However, for a pixel map, chi decays slowly.
        # But usually, the basis set is chosen large enough.
        # Here, the difference of two vectors in [-8,8] is in [-16,16].
        # We only computed chi for [-8,8].
        # We should compute chi for the larger set [-16,16] to be safe?
        # Or assume that chi_{G} is negligible for |G| > 8?
        # For a pixel map, chi has significant high-frequency content.
        # Let's compute chi for a larger set to be accurate.
        # Max difference index is 16.
        # So we need G vectors from -16 to 16.
        # That would be 33x33 = 1089 vectors.
        # This increases the cost of compute_chi significantly.
        # But we only need chi for the differences.
        # Let's re-evaluate.
        
        # The matrix element is sum over G' of (k+G).(k+G') chi_{G-G'} C_{G'}.
        # The basis functions are e^{i(k+G)r} for G in the cutoff set.
        # The operator involves 1/epsilon, which has Fourier series over all G.
        # So we need chi_{G-G'} for all G, G' in the cutoff set.
        # G, G' in [-8,8]. G-G' in [-16,16].
        # So we need chi for G in [-16,16].
        
        # Let's modify compute_chi to take a cutoff.
        # But DOT_PRODS is pre-computed for G in [-8,8].
        # So we need chi for differences.
        
        # To avoid recomputing chi for every evaluation, we can pre-compute the index map
        # and only compute the necessary chi values.
        
        pass # Placeholder for logic update

    return (2 * (min_omega_5 - max_omega_4)) / (min_omega_5 + max_omega_4) if min_omega_5 > max_omega_4 else -1.0

# Refining the approach for chi:
# We will pre-compute the list of unique G-differences needed.
# And compute chi for those specific G-vectors.
# The set of differences is {G_i - G_j | i, j in 0..N_G-1}.
# This set is contained in [-16, 16] x [-16, 16].

DIFF_G_VECTORS = set()
for g1 in G_VECTORS:
    for g2 in G_VECTORS:
        diff = (g1[0] - g2[0], g1[1] - g2[1])
        DIFF_G_VECTORS.add(diff)

DIFF_G_LIST = list(DIFF_G_VECTORS)
N_DIFF = len(DIFF_G_LIST)

# Map difference to index in DIFF_G_LIST
DIFF_TO_IDX = {g: i for i, g in enumerate(DIFF_G_LIST)}

def compute_chi_for_diffs(cell):
    """
    Computes chi_G for G in DIFF_G_LIST.
    """
    eps_inv = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            eps_inv[i, j] = 1.0 / EPS_HIGH if cell[i][j] == 1 else 1.0 / EPS_LOW
            
    x = (np.arange(SIZE) + 0.5) / SIZE
    y = (np.arange(SIZE) + 0.5) / SIZE
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    chi_diffs = np.zeros(N_DIFF, dtype=complex)
    for idx, (gx, gy) in enumerate(DIFF_G_LIST):
        phase = np.exp(-1j * (gx * X + gy * Y))
        chi_diffs[idx] = np.sum(eps_inv * phase) / (SIZE * SIZE)
        
    return chi_diffs

def compute_gap_v2(cell):
    chi_diffs = compute_chi_for_diffs(cell)
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for k_idx, k in enumerate(K_POINTS):
        dp_matrix = DOT_PRODS[k_idx]
        
        # Build A
        A = np.zeros((N_G, N_G), dtype=complex)
        for i in range(N_G):
            g_i = G_VECTORS[i]
            for j in range(N_G):
                g_j = G_VECTORS[j]
                diff = (g_i[0] - g_j[0], g_i[1] - g_j[1])
                chi_val = chi_diffs[DIFF_TO_IDX[diff]]
                A[i, j] = dp_matrix[i, j] * chi_val
                
        # Solve eigenvalue problem
        # A is Hermitian?
        # dp_matrix is real symmetric.
        # chi_{-G} = conjugate(chi_G).
        # A_{ij} = dp_{ij} chi_{G_i - G_j}.
        # A_{ji} = dp_{ji} chi_{G_j - G_i} = dp_{ij} chi_{-(G_i - G_j)} = dp_{ij} conj(chi_{G_i - G_j}) = conj(A_{ij}).
        # So A is Hermitian.
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.sort(eigvals.real)
        
        # Bands are indexed 1, 2, 3, 4, 5...
        # eigvals[0] is band 1.
        # Band 4 is eigvals[3].
        # Band 5 is eigvals[4].
        
        omega_4 = np.sqrt(eigvals[3])
        omega_5 = np.sqrt(eigvals[4])
        
        if omega_5 < min_omega_5:
            min_omega_5 = omega_5
        if omega_4 > max_omega_4:
            max_omega_4 = omega_4
            
    if min_omega_5 <= max_omega_4:
        return -1.0
        
    return (2 * (min_omega_5 - max_omega_4)) / (min_omega_5 + max_omega_4)

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0 # 10s margin
    
    # Initial design: A centered rod
    independent_values = [0] * 136
    # Set rod radius
    # Map independent indices to coordinates
    indep_coords = list(get_independent_indices())
    for idx, (i, j) in enumerate(indep_coords):
        # Center is 15.5
        dx = i - 15.5
        dy = j - 15.5
        if dx*dx + dy*dy < 5*5: # Radius 5
            independent_values[idx] = 1
            
    current_cell = apply_symmetry(None, independent_values)
    current_score = compute_gap_v2(current_cell)
    
    best_cell = current_cell
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random independent pixel
        idx = np.random.randint(0, 136)
        
        # Flip it
        old_val = independent_values[idx]
        independent_values[idx] = 1 - old_val
        
        new_cell = apply_symmetry(None, independent_values)
        new_score = compute_gap_v2(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell
        else:
            # Revert
            independent_values[idx] = old_val
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_cell

# The function signature must be design(problem)
# The problem dict is not used in the current implementation, but we keep it.

# One issue: compute_gap_v2 is slow because of the O(N_G^2) loop in Python.
# We should vectorize the construction of A.

def compute_gap_v3(cell):
    chi_diffs = compute_chi_for_diffs(cell)
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    # Pre-compute the difference indices for all pairs (i, j)
    # diff_indices[i, j] = index in DIFF_G_LIST of G_i - G_j
    diff_indices = np.zeros((N_G, N_G), dtype=int)
    for i in range(N_G):
        g_i = G_VECTORS[i]
        for j in range(N_G):
            g_j = G_VECTORS[j]
            diff = (g_i[0] - g_j[0], g_i[1] - g_j[1])
            diff_indices[i, j] = DIFF_TO_IDX[diff]
            
    for k_idx, k in enumerate(K_POINTS):
        dp_matrix = DOT_PRODS[k_idx]
        
        # A = dp_matrix * chi_diffs[diff_indices]
        # chi_diffs is 1D, diff_indices is 2D.
        # We need to index chi_diffs with diff_indices.
        chi_matrix = chi_diffs[diff_indices]
        
        A = dp_matrix * chi_matrix
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.sort(eigvals.real)
        
        omega_4 = np.sqrt(eigvals[3])
        omega_5 = np.sqrt(eigvals[4])
        
        if omega_5 < min_omega_5:
            min_omega_5 = omega_5
        if omega_4 > max_omega_4:
            max_omega_4 = omega_4
            
    if min_omega_5 <= max_omega_4:
        return -1.0
        
    return (2 * (min_omega_5 - max_omega_4)) / (min_omega_5 + max_omega_4)

# Update design to use v3
def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0
    
    independent_values = [0] * 136
    indep_coords = list(get_independent_indices())
    for idx, (i, j) in enumerate(indep_coords):
        dx = i - 15.5
        dy = j - 15.5
        if dx*dx + dy*dy < 5*5:
            independent_values[idx] = 1
            
    current_cell = apply_symmetry(None, independent_values)
    current_score = compute_gap_v3(current_cell)
    
    best_cell = current_cell
    best_score = current_score
    
    while time.time() < deadline:
        idx = np.random.randint(0, 136)
        
        old_val = independent_values[idx]
        independent_values[idx] = 1 - old_val
        
        new_cell = apply_symmetry(None, independent_values)
        new_score = compute_gap_v3(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell
        else:
            independent_values[idx] = old_val
            
    return best_cell
