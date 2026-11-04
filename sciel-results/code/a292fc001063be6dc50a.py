import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

N = 13

def construct():
    # Parameters for the banded matrix
    # We use a tridiagonal matrix with learnable main and off-diagonal elements.
    # To ensure SPD, we parameterize off-diagonals as tanh(x) * 0.9 and main diagonals as 1 + x^2.
    
    def points_from_params(params):
        # params: array of shape (2*N - 1, )
        # First N: main diagonal offsets
        # Next N-1: off-diagonal offsets
        
        n_main = N
        n_off = N - 1
        
        main_offsets = params[:n_main]
        off_offsets = params[n_main:n_main+n_off]
        
        # Construct main diagonal: 1 + offset^2 to ensure positive
        main_diag = 1.0 + main_offsets**2
        
        # Construct off-diagonals: tanh(offset) * 0.9 to ensure |off| < 1, helping SPD
        off_diag = np.tanh(off_offsets) * 0.9
        
        # Build the matrix
        M = np.zeros((N, N))
        for i in range(N):
            M[i, i] = main_diag[i]
        for i in range(N - 1):
            M[i, i+1] = off_diag[i]
            M[i+1, i] = off_diag[i]
            
        # Ensure SPD (just in case, though construction should guarantee it)
        # M = (M + M.T) / 2
        # Eigenvalues should be positive.
        
        # Compute eigenvectors
        try:
            w, v = eigh(M)
        except Exception:
            # Fallback to regular polygon if eigendecomposition fails
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack((np.cos(angles), np.sin(angles)))
            
        # v is N x N, columns are eigenvectors
        # We take the first 2 columns as the coordinates for the N points
        # This maps the N-dimensional eigenvector space to 2D
        pts = v[:, :2]
        
        # Normalize points to avoid scale issues during optimization
        # Center at 0
        pts = pts - np.mean(pts, axis=0)
        
        # Scale to have unit variance or fixed norm
        # The score is scale invariant, but scaling helps gradient
        norm = np.linalg.norm(pts)
        if norm > 1e-6:
            pts = pts / norm * np.sqrt(N) # Scale to have sum of squares = N
        
        return pts

    def objective(params):
        pts = points_from_params(params)
        
        # Calculate convex hull area
        try:
            hull = __import__('scipy.spatial', fromlist=['ConvexHull']).ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
            
        # Calculate minimum triangle area
        min_area = float('inf')
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 0.0
            
        score = min_area / hull_area
        return -score

    # Initial parameters
    # Start with regular tridiagonal matrix (constant main and off)
    # Main: 1, Off: 0.5
    init_params = np.zeros(2 * N - 1)
    init_params[:N] = 0.0 # main diag = 1
    init_params[N:] = np.arctanh(0.5 / 0.9) # off diag = 0.5
    
    # Optimize
    # Use L-BFGS-B or similar
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 500})
    
    if result.success or result.fun < objective(init_params):
        final_params = result.x
    else:
        final_params = init_params
        
    points = points_from_params(final_params)
    
    # Final normalization for output (optional, but good for consistency)
    # The evaluator normalizes by hull area, so raw points are fine.
    return points
