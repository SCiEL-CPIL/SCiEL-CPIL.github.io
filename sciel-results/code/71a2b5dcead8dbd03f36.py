import numpy as np
import time
import scipy.optimize as opt

# --- Precomputation for the Evaluator ---

# The evaluator uses 289 plane waves (N=8, so Gx, Gy in [-8, 8])
# We need to match this exactly for the score to be comparable.
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Diff vectors for Fourier coefficients of the 32x32 pixel map
# The pixel map is 32x32. The Fourier coefficient for difference (dx, dy) is:
# sum_{x,y} eps(x,y) * exp(-i 2pi (x*dx + y*dy)/32) / 32^2
# We need to map the G differences to pixel indices.
# G is in units of 2pi/L, where L=32 pixels? No, G is integer index.
# The basis functions are exp(i (k+G) r).
# The matrix element M_{G'G} = <k+G'| 1/eps | k+G>
# 1/eps is expanded in Fourier series: (1/eps)(r) = sum_{G} (1/eps)_G exp(i G r)
# (1/eps)_G = (1/32^2) sum_{x,y} (1/eps)(x,y) exp(-i G (x,y))
# Here G is the vector (nx, ny) in the G-basis.
# The pixel indices x,y are 0..31.
# The exponent is exp(-i 2pi (nx*x + ny*y)/32).
# This is exactly the DFT of the 32x32 array.

# Precompute indices for the FFT
# We need (1/eps)_G for G in the set of differences between the 289 G vectors.
# The differences (Gx' - Gx, Gy' - Gy) range from -16 to 16.
# We can compute the full 32x32 FFT of (1/eps) and extract the needed components.
# The FFT gives coefficients for frequencies -16..15.
# We need frequencies -16..16.
# Note: FFT index k corresponds to frequency k if k <= 15, and k-32 if k > 15.
# So index 16 corresponds to frequency -16.

# Precompute the mapping from G-diff to FFT index
# diff_x is in [-16, 16]
# fft_idx_x = diff_x if diff_x >= 0 else diff_x + 32
# Wait, standard FFT:
# X[k] = sum x[n] exp(-i 2pi kn/N)
# Inverse: x[n] = 1/N sum X[k] exp(i 2pi kn/N)
# If we want coefficient c_m for exp(i 2pi m n / N), it is X[m] (with wrap-around).
# For m in [-16, 16]:
# if m >= 0, idx = m
# if m < 0, idx = m + 32

Diff_x_all = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y_all = Gy_flat[:, None] - Gy_flat[None, :]

# Map diffs to FFT indices
def diff_to_fft_idx(d):
    return np.where(d >= 0, d, d + 32)

idx_x = diff_to_fft_idx(Diff_x_all)
idx_y = diff_to_fft_idx(Diff_y_all)

# Precompute k-points
# The problem states "30 wave vectors covering the boundary and interior of the irreducible Brillouin zone".
# The seed program used:
# Gamma-X: 10 points
# X-M: 10 points
# M-Gamma: 10 points
# Let's stick to this specific set to ensure we are optimizing for the exact same metric.

def generate_k_points(num_points=30):
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    # The seed used (i+0.5)*pi/10. This is midpoints of 10 intervals.
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
    return np.array(points)

K_POINTS = generate_k_points()

# Precompute (k+G)^2 for each k-point
KG_SQ = []
for k in K_POINTS:
    kx, ky = k
    # k is in radians? 
    # The seed used kx, ky directly in kg = k + G * 2pi.
    # Wait, the seed code:
    # kg = k[None, :] + G_vectors * (2 * np.pi)
    # This implies k is in units of 1/length, and G is integer index.
    # So G_vector * 2pi is the actual wave vector.
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ)

def evaluate_cell(cell):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: 32x32 numpy array of 0/1.
    """
    size = 32
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    eps = np.where(cell == 1, 12.0, 1.0)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    # mu_fft[kx, ky] = sum mu[x, y] exp(-i 2pi (kx x + ky y) / 32)
    mu_fft = np.fft.fft2(mu)
    # Normalize by N^2
    mu_coeffs = mu_fft / (size * size)
    
    # Extract the needed coefficients for the 289x289 matrix
    # mu_diff[i, j] = mu_coeffs[idx_x[i,j], idx_y[i,j]]
    mu_diff = mu_coeffs[idx_x, idx_y]
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        kg_sq = KG_SQ[ki]
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Solve for eigenvalues
        # We need the 5 smallest eigenvalues.
        # Using eigvalsh is fast enough for 289x289 if done 30 times?
        # 289^3 * 30 ~ 70 million ops. Should be fine.
        evals = np.linalg.eigvalsh(S)
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def fourier_to_cell(fourier_coeffs):
    """
    Convert a set of C4v-symmetric Fourier coefficients to a 32x32 cell.
    fourier_coeffs: 1D array of real coefficients for allowed (m, n) modes.
    We define the modes to respect C4v.
    The modes are (m, n) where m, n are integers.
    C4v symmetry requires:
    1. eps(x, y) = eps(-x, y) = eps(x, -y) = eps(-x, -y) (inversion + reflections)
    2. eps(x, y) = eps(y, x) (rotation by 90)
    
    In Fourier space, this means c_{m,n} = c_{-m,n} = c_{m,-n} = c_{-m,-n} = c_{n,m}.
    And c_{m,n} must be real (since eps is real).
    
    We can parameterize by a set of unique (m, n) pairs.
    Let's pick a set of low-frequency modes.
    Max frequency is 16.
    We'll iterate m from 0 to 16, n from 0 to 16.
    Unique modes:
    - (0, 0)
    - (m, 0) for m=1..16
    - (0, n) for n=1..16 -- but (0, n) is equivalent to (n, 0) by rotation? 
      Wait, rotation by 90 maps (x,y) to (-y,x).
      Fourier: c_{m,n} -> c_{-n,m}.
      So (0, n) -> (-n, 0) -> (n, 0). Yes, (0, n) and (n, 0) are the same mode under C4.
      So we only need one of them. Let's say (m, 0) with m>0 covers both.
    - (m, n) with m > n > 0.
    
    So the independent modes are:
    1. (0, 0)
    2. (m, 0) for m = 1..16
    3. (m, n) for m > n >= 1, m+n <= some limit? Or just all m,n in range.
    
    Let's limit to m, n <= 4 or 5 to keep the search space small and fast.
    Higher frequencies create fine details (noise) which might not help the gap and hurt convergence.
    Let's try m, n up to 3 or 4.
    
    Modes:
    (0,0)
    (1,0), (2,0), (3,0), (4,0)
    (1,1), (2,1), (3,1), (4,1)
    (2,2), (3,2), (4,2)
    (3,3), (4,3)
    (4,4)
    
    Total: 1 + 4 + 4 + 3 + 2 + 1 = 15 parameters.
    This is a very small search space.
    
    Let's define the list of (m, n) pairs.
    """
    size = 32
    # List of unique (m, n) for C4v
    modes = []
    modes.append((0, 0))
    for m in range(1, 5):
        modes.append((m, 0))
        for n in range(1, m):
            modes.append((m, n))
        modes.append((m, m))
        
    # The input vector 'fourier_coeffs' has length len(modes)
    # Reconstruct the 2D array of coefficients c_{m,n}
    # c_{m,n} is real.
    # We need to fill a 32x32 array of complex coefficients (or just real, since symmetric).
    # Actually, we can just compute the inverse DFT directly.
    # eps(x,y) = sum_{m,n} c_{m,n} exp(i 2pi (m x + n y) / 32)
    
    # Initialize the Fourier coefficient array (32x32)
    C = np.zeros((size, size), dtype=float)
    
    for i, (m, n) in enumerate(modes):
        val = fourier_coeffs[i]
        # Fill in the symmetric partners
        # (m, n)
        C[m % 32, n % 32] += val
        # (-m, n) -> (32-m, n)
        C[(-m) % 32, n % 32] += val
        # (m, -n) -> (m, 32-n)
        C[m % 32, (-n) % 32] += val
        # (-m, -n)
        C[(-m) % 32, (-n) % 32] += val
        
        # Rotation partners: (n, m), (-n, m), (n, -m), (-n, -m)
        # If m == n, these are the same.
        if m != n:
            C[n % 32, m % 32] += val
            C[(-n) % 32, m % 32] += val
            C[n % 32, (-m) % 32] += val
            C[(-n) % 32, (-m) % 32] += val
            
    # Inverse DFT
    # numpy ifft2 assumes X[k] = sum x[n] exp(-i 2pi kn/N)
    # So x[n] = 1/N sum X[k] exp(i 2pi kn/N)
    # Our C is the array of coefficients for exp(i 2pi ...).
    # So we just need ifft2(C) * N?
    # ifft2(C) gives 1/N sum C[k] exp(i 2pi kn/N).
    # So eps = N * ifft2(C) ?
    # Let's check.
    # ifft2(C)[x,y] = 1/32^2 sum_{k,l} C[k,l] exp(i 2pi (kx + ly)/32)
    # We want sum C[k,l] exp(i 2pi (kx + ly)/32)
    # So eps = 32^2 * ifft2(C)
    
    eps_float = (size * size) * np.fft.ifft2(C).real
    
    # Threshold to 0/1
    # The mean of eps should be controlled by the (0,0) mode.
    # To ensure a good fill fraction, we might need to adjust the threshold.
    # However, for optimization, it's better to let the optimizer find the threshold.
    # But the problem requires 0/1.
    # We can use a fixed threshold (e.g., 6.5) or an adaptive one.
    # Let's use a fixed threshold of 6.5 (midpoint of 1 and 12).
    # Actually, the value of the cell is 0 or 1.
    # If eps_float > 6.5, cell=1, else 0.
    
    cell = (eps_float > 6.5).astype(int)
    
    # Enforce exact C4v symmetry on the discrete grid to avoid numerical artifacts
    # The FFT reconstruction should be symmetric, but rounding might break it slightly.
    # We can enforce it by averaging the symmetric pixels.
    # cell[i,j] = cell[31-i, j] = cell[i, 31-j] = cell[31-i, 31-j] = cell[j, i] ...
    # Let's just ensure the output is perfectly symmetric by taking the average of symmetric neighbors and rounding.
    # Or simpler: just return the FFT result, it's usually close enough.
    # But the evaluator checks symmetry? "invariant under the square symmetry group".
    # The FFT result *is* invariant if the coefficients are invariant.
    # Due to floating point, it might be 0.99999.
    # The thresholding makes it 0 or 1.
    # Let's verify symmetry.
    return cell

def objective(params):
    """
    Objective function for optimization.
    Returns negative score (to maximize score, minimize -score).
    """
    cell = fourier_to_cell(params)
    score = evaluate_cell(cell)
    return -score

def design(problem):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Initial parameters
    # We have 15 parameters.
    # Let's initialize with a random guess or a specific pattern.
    # A good starting point might be a ring or a cross.
    # Let's try a few random starts.
    
    np.random.seed(42)
    
    best_cell = None
    best_score = -float('inf')
    
    # Define the modes again for clarity
    modes = []
    modes.append((0, 0))
    for m in range(1, 5):
        modes.append((m, 0))
        for n in range(1, m):
            modes.append((m, n))
        modes.append((m, m))
    num_params = len(modes)
    
    # Try a few different initializations
    # 1. All zeros (air)
    # 2. All ones (dielectric) -> (0,0) mode high
    # 3. Random
    
    initializations = []
    # Air
    initializations.append(np.zeros(num_params))
    # Dielectric
    init_dielectric = np.zeros(num_params)
    init_dielectric[0] = 1.0 # (0,0) mode
    initializations.append(init_dielectric)
    
    # Random
    for _ in range(5):
        initializations.append(np.random.randn(num_params) * 0.5)
        
    # Also try a specific "ring" initialization
    # A ring is high in the (0,0) mode and has specific higher modes.
    # Let's just rely on the optimizer.
    
    for init_params in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for derivative-free optimization
            # It works well for low-dimensional, non-smooth (due to thresholding) problems
            result = opt.minimize(
                objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            final_params = result.x
            cell = fourier_to_cell(final_params)
            score = evaluate_cell(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell
                # print(f"New best: {best_score:.4f} at time {time.time()-start_time:.2f}s")
                
        except Exception as e:
            # print(f"Optimization failed: {e}")
            pass

    if best_cell is None:
        # Fallback: simple cross
        size = 32
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                    best_cell[i][j] = 1

    # Convert to list of lists
    return best_cell.tolist()

# To allow testing
if __name__ == "__main__":
    problem = {}
    result = design(problem)
    print("Result shape:", len(result), len(result[0]))
    print("Score:", evaluate_cell(np.array(result)))
