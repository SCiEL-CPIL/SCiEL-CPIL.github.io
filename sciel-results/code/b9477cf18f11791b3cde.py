import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
dw = 1.0 / N
warnings.filterwarnings('ignore')

# Pre-compute geometry for fast convolution
x = np.linspace(-0.25, 0.25, N + 1)
# Fine grid for evaluating the supremum of the autoconvolution
n_fine = 200
t_fine = np.linspace(-0.5, 0.5, n_fine)

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution on a fine grid
    # We can vectorize the inner loop somewhat, but N=50 is small.
    # autoconv(t) = sum_i sum_j h_i h_j * measure({tau in [x_i, x_{i+1}] : t-tau in [x_j, x_{j+1}]})
    # = sum_i sum_j h_i h_j * measure([x_i, x_{i+1}] \cap [t - x_{j+1}, t - x_j])
    
    # For speed, we pre-compute the intersection lengths for all pairs (i, j) at all t_fine points?
    # That's N*N*n_fine = 50*50*200 = 500,000 operations. This is very fast in numpy.
    
    # Let's create a 3D array or compute efficiently.
    # Actually, let's just loop over t_fine and use vectorized ops for i, j.
    
    autoconv = np.zeros(n_fine)
    
    # Pre-compute intervals for j
    # For a fixed t, the condition is tau in [x_i, x_{i+1}] and t-tau in [x_j, x_{j+1}]
    # => tau in [t - x_{j+1}, t - x_j]
    # Intersection of [x_i, x_{i+1}] and [t - x_{j+1}, t - x_j]
    
    # Let's pre-compute the "kernel" K[i, j, k] where k is index in t_fine?
    # Memory: 50*50*200 = 500k floats = 4MB. Acceptable.
    
    # But we can do it even faster.
    # Let's just compute the convolution using the property that it's piecewise quadratic?
    # No, the sum of triangles is piecewise linear? No, sum of triangles is piecewise quadratic?
    # Wait, R_i * R_j is a triangle (linear pieces). Sum of linear pieces is linear on intervals where the set of active triangles is constant.
    # The breakpoints are at x_i + x_j and x_i + x_{j+1}.
    # Since x_i are equally spaced, the breakpoints are at integer multiples of dw.
    # So the function is linear on intervals [k*dw, (k+1)*dw].
    # Therefore, the maximum must occur at one of the endpoints of these intervals, i.e., at t = k*dw.
    # Our t_fine grid with n_fine=200 over [-0.5, 0.5] has step 0.5/199 ≈ 0.0025.
    # dw = 1/50 = 0.02.
    # So t_fine is much finer than the breakpoints.
    # Thus, sampling on t_fine is sufficient to find the exact maximum (which occurs at a breakpoint or vertex).
    # Actually, since it's piecewise linear, the max is at a vertex. The vertices are at t = x_i + x_j.
    # x_i + x_j ranges from -0.5 to 0.5 in steps of dw.
    # So we only need to evaluate at t = -0.5, -0.5+dw, ..., 0.5.
    # Number of points: 100.
    
    t_vertices = np.linspace(-0.5, 0.5, 101)
    
    # Let's re-calculate autoconv on t_vertices for better precision/speed
    autoconv = np.zeros(len(t_vertices))
    
    # Vectorized computation for all t_vertices at once?
    # For each i, j, we need the length of intersection of [x_i, x_{i+1}] and [t - x_{j+1}, t - x_j] for all t.
    
    # Let's just loop over i, j and add to autoconv.
    # 50*50 = 2500 iterations. Each does vectorized ops on array of size 101.
    # This is very fast.
    
    for i in range(N):
        if h[i] == 0: continue
        for j in range(N):
            if h[j] == 0: continue
            
            # Interval for tau: [x_i, x_{i+1}]
            # Interval for t-tau: [x_j, x_{j+1}] => tau in [t - x_{j+1}, t - x_j]
            # Intersection length L(t) = max(0, min(x_{i+1}, t - x_j) - max(x_i, t - x_{j+1}))
            
            lo = np.maximum(x[i], t_vertices - x[j+1])
            hi = np.minimum(x[i+1], t_vertices - x[j])
            lengths = np.maximum(0, hi - lo)
            
            autoconv += h[i] * h[j] * lengths
            
    sup_conv = np.max(autoconv)
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_from_z(z):
    """Compute score from complex vector z where h_i = |z_i|^2."""
    h = np.abs(z)**2
    return _compute_score(h)

def _neg_score_from_z(z):
    """Negative score for minimization."""
    return -_score_from_z(z)

def _generate_sidon_set(n, size):
    """Generate a Sidon subset of {0,...,n-1} of given size using greedy."""
    S = []
    sums = set()
    for i in range(n):
        if len(S) >= size:
            break
        new_sums = set()
        valid = True
        for j in S + [i]:
            for k in S + [i]:
                if j <= k:
                    s = j + k
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
        if valid:
            S.append(i)
            sums |= new_sums
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes with complex parameterization
    for size in range(2, 15):
        if time.time() > deadline - 10:
            break
        
        S = _generate_sidon_set(N, size)
        if len(S) < size:
            continue
        
        # Initialize z with equal magnitudes at Sidon positions
        z0 = np.zeros(N, dtype=complex)
        for idx in S:
            z0[idx] = 1.0 / np.sqrt(size)
        
        # Also add small random phases to break symmetry
        np.random.seed(42 + size)
        z0[S] *= np.exp(1j * np.random.uniform(0, 2*np.pi, size))
        
        h0 = np.abs(z0)**2
        score0 = _compute_score(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Optimize over z (complex -> 2*N real variables)
        try:
            def _obj_flat(z_flat):
                z = z_flat[:N].astype(complex) + 1j * z_flat[N:]
                return _neg_score_from_z(z)
            
            z0_flat = np.concatenate([z0.real, z0.imag])
            
            # Use L-BFGS-B for optimization
            result = minimize(_obj_flat, z0_flat, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12})
            
            z_trial = result.x[:N].astype(complex) + 1j * result.x[N:]
            h_trial = np.abs(z_trial)**2
            # Keep only Sidon positions
            h_trial[~np.isin(np.arange(N), S)] = 0.0
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Also try the original 2-element support
    h0 = np.zeros(N)
    h0[0] = 2.0/3.0
    h0[1] = 1.0/3.0
    score0 = _compute_score(h0)
    if score0 > best_score:
        best_score = score0
        best_h = h0.copy()
    
    # Final polish using complex parameterization
    if time.time() < deadline - 5:
        try:
            # Convert best_h to z
            z_init = np.sqrt(best_h) + 0j
            # Add small imaginary part to break symmetry
            np.random.seed(123)
            z_init += 1j * 0.01 * np.random.randn(N)
            
            def _obj_flat(z_flat):
                z = z_flat[:N].astype(complex) + 1j * z_flat[N:]
                return _neg_score_from_z(z)
            
            z0_flat = np.concatenate([z_init.real, z_init.imag])
            
            result = minimize(_obj_flat, z0_flat, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14})
            
            z_trial = result.x[:N].astype(complex) + 1j * result.x[N:]
            h_trial = np.abs(z_trial)**2
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
