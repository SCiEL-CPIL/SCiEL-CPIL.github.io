import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 140.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _fourier_to_points(coeffs, n_harmonics=3):
    """
    Convert Fourier coefficients to N points on/near unit circle.
    coeffs: array of shape (2*n_harmonics + 1, 2) = (real, imag) pairs for each harmonic
    Returns: array of shape (N, 2)
    
    We use: z(t) = sum_{k=0}^{K} (a_k + i*b_k) * exp(i*k*t)
    where t = 2*pi*n/N for n=0,...,N-1
    
    To keep points inside unit disk, we scale appropriately.
    """
    K = n_harmonics
    angles = 2 * np.pi * np.arange(N) / N
    
    # coeffs shape: (K+1, 2) where [:,0]=real, [:,1]=imag
    # z(t) = sum_{k=0}^{K} c_k * exp(i*k*t)
    # x(t) = Re(z(t)), y(t) = Im(z(t))
    
    x = np.zeros(N)
    y = np.zeros(N)
    
    for k in range(K + 1):
        a_k = coeffs[k, 0]  # real part
        b_k = coeffs[k, 1]  # imag part
        
        # exp(i*k*t) = cos(k*t) + i*sin(k*t)
        # (a_k + i*b_k) * (cos(k*t) + i*sin(k*t))
        # = (a_k*cos(k*t) - b_k*sin(k*t)) + i*(a_k*sin(k*t) + b_k*cos(k*t))
        
        kt = k * angles
        x += a_k * np.cos(kt) - b_k * np.sin(kt)
        y += a_k * np.sin(kt) + b_k * np.cos(kt)
    
    pts = np.column_stack([x, y])
    
    # Normalize to unit area hull
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        pts *= scale
    
    return pts

def _points_to_fourier(pts, n_harmonics=3):
    """
    Invert _fourier_to_points approximately using least squares.
    This is for initialization purposes.
    """
    K = n_harmonics
    angles = 2 * np.pi * np.arange(N) / N
    
    # Build design matrix
    # For each point n, we have:
    # x_n = sum_{k=0}^{K} [a_k*cos(k*theta_n) - b_k*sin(k*theta_n)]
    # y_n = sum_{k=0}^{K} [a_k*sin(k*theta_n) + b_k*cos(k*theta_n)]
    
    # We can solve for all 2*(K+1) coefficients at once
    # Stack x and y equations
    
    A = np.zeros((2*N, 2*(K+1)))
    b = np.zeros(2*N)
    
    for n in range(N):
        theta = angles[n]
        for k in range(K + 1):
            col = 2*k
            # x equation
            A[2*n, col] = np.cos(k*theta)
            A[2*n, col+1] = -np.sin(k*theta)
            # y equation
            A[2*n+1, col] = np.sin(k*theta)
            A[2*n+1, col+1] = np.cos(k*theta)
    
    b[:N] = pts[:, 0]
    b[N:] = pts[:, 1]
    
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.reshape(K+1, 2)

def _neg_score_fourier(coeffs, n_harmonics=3):
    """Negative score for Fourier coefficient optimization."""
    try:
        pts = _fourier_to_points(coeffs, n_harmonics)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return 1e10
        score = _score_points(pts)
        return -score
    except Exception:
        return 1e10

def _make_initial_guesses_fourier(n_harmonics=3):
    """Generate diverse initial Fourier coefficient configurations."""
    inits = []
    K = n_harmonics
    
    # 1. Regular polygon: only k=1 harmonic
    c1 = np.zeros((K+1, 2))
    c1[1, 0] = 1.0  # a_1 = 1
    inits.append(c1.flatten())
    
    # 2. Slightly distorted regular polygon
    rng = np.random.RandomState(42)
    c2 = np.zeros((K+1, 2))
    c2[1, 0] = 1.0
    c2[2, 0] = 0.1
    c2[2, 1] = 0.05
    inits.append(c2.flatten())
    
    # 3. Random small perturbations
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        c = np.zeros((K+1, 2))
        c[1, 0] = 1.0
        for k in range(2, K+1):
            c[k, 0] = rng.uniform(-0.2, 0.2)
            c[k, 1] = rng.uniform(-0.2, 0.2)
        inits.append(c.flatten())
    
    # 4. Different base configurations
    # Ellipse-like
    c4 = np.zeros((K+1, 2))
    c4[1, 0] = 1.0
    c4[1, 1] = 0.3
    inits.append(c4.flatten())
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Evaluate some traditional initial guesses first
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    s1 = _score_points(p1)
    if s1 > best_score:
        best_score = s1
        best_params = p1.copy()
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    s2 = _score_points(p2)
    if s2 > best_score:
        best_score = s2
        best_params = p2.copy()
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    s3 = _score_points(p3)
    if s3 > best_score:
        best_score = s3
        best_params = p3.copy()
    
    # Now use Fourier parameterization with differential evolution
    n_harmonics = 3
    coeff_dim = 2 * (n_harmonics + 1)  # 8 dimensions
    
    # Define bounds for Fourier coefficients
    # a_0, b_0: center offset (small)
    # a_k, b_k for k>=1: shape coefficients
    bounds = [(-0.1, 0.1)] * 2  # k=0
    for k in range(1, n_harmonics + 1):
        bounds.append((-1.5, 1.5))  # a_k
        bounds.append((-1.5, 1.5))  # b_k
    
    # Evaluate initial Fourier guesses
    inits_fourier = _make_initial_guesses_fourier(n_harmonics)
    for c in inits_fourier:
        try:
            pts = _fourier_to_points(c, n_harmonics)
            s = _score_points(pts)
            if s > best_score:
                best_score = s
                best_params = pts.copy()
        except Exception:
            pass
    
    # Differential Evolution on Fourier coefficients
    try:
        result = differential_evolution(
            _neg_score_fourier,
            bounds=bounds,
            args=(n_harmonics,),
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.fun < 1e9:  # Valid result
            best_coeffs = result.x
            pts_de = _fourier_to_points(best_coeffs, n_harmonics)
            s_de = _score_points(pts_de)
            if s_de > best_score:
                best_score = s_de
                best_params = pts_de.copy()
    except Exception:
        pass
    
    # Check time and do local refinement if possible
    if (time.time() - t_start) < (DEADLINE - 15.0):
        # Local refinement on the best point found
        try:
            def _neg_local(pts_flat):
                pts = pts_flat.reshape(N, 2)
                return -_score_points(pts)
            
            res = minimize(
                _neg_local,
                best_params.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < -best_score:
                best_params = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass
    
    # Final check
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_params = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_params
