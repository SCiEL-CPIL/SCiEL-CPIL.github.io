import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Parameters for a symmetric tridiagonal matrix
    # d: diagonal (N), o: off-diagonal (N-1)
    n_params = 2 * N - 1
    
    def get_points(params):
        d = params[:N]
        o = params[N:]
        
        # Construct the symmetric tridiagonal matrix
        A = np.diag(d)
        for i in range(N - 1):
            A[i, i + 1] = o[i]
            A[i + 1, i] = o[i]
            
        # Eigendecomposition
        # We want the eigenvectors. 
        # scipy.linalg.eigh returns eigenvalues in ascending order and eigenvectors as columns.
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            # Fallback to identity if it fails (shouldn't happen for symmetric)
            return np.column_stack((np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)), 
                                    np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))))
        
        # The points are the first two columns of the eigenvector matrix?
        # Or the first two rows?
        # "Representing the 13 points as the first 13 non-trivial eigenvectors"
        # If we take the first 2 columns of Q (eigenvectors 0 and 1), we get 13 points in R2.
        # Q is N x N. Q[:, 0] is the first eigenvector (N components).
        # So Point i = (Q[i, 0], Q[i, 1]).
        
        pts = np.column_stack((eigvecs[:, 0], eigvecs[:, 1]))
        
        # Normalize or scale? 
        # The eigenvectors are normalized to unit length.
        # The points will be bounded by 1.
        # We might need to scale them up or down, but the score is ratio, so scale cancels out.
        # However, if the points are very small, numerical issues? Unlikely.
        
        return pts

    def score(points):
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 0.0
            
        return min_area / hull_area

    def objective(params):
        pts = get_points(params)
        
        # Check convexity: all points should be on the hull
        try:
            hull = ConvexHull(pts)
            if len(hull.vertices) < N:
                # Penalize non-convex positions
                # We want to maximize score, so minimize -score.
                # Return a large positive value to discourage this.
                return 1.0
        except Exception:
            return 1.0
            
        s = score(pts)
        return -s

    # Initial guess
    # Start with a matrix that produces a regular-polygon-like structure?
    # A regular polygon corresponds to points on a circle.
    # Can we find a tridiagonal matrix whose first two eigenvectors form a circle?
    # Probably not exactly, but we can start with a simple diagonal matrix.
    # If A is diagonal, eigenvectors are standard basis vectors.
    # Q = I. Points: (1,0), (0,1), (0,0)... (0,0). This is degenerate.
    
    # We need off-diagonal terms to mix the components.
    # Let's start with a Toeplitz-like matrix: constant diagonal, constant off-diagonal.
    # d = [1]*N, o = [0.5]*(N-1)
    x0 = np.zeros(n_params)
    x0[:N] = 1.0
    x0[N:] = 0.5
    
    # Bounds
    # Keep parameters in a reasonable range to avoid numerical instability
    bounds = [(-2.0, 2.0)] * n_params
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_params = res.x
    best_points = get_points(best_params)
    
    # Ensure validity
    try:
        hull = ConvexHull(best_points)
        if len(hull.vertices) < N:
             # Fallback to regular polygon
             angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
             return np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
