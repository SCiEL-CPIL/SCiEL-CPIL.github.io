import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # 1. Spectral Initialization:
    # Construct a circulant Laplacian for a ring of N nodes.
    # This yields a regular N-gon as the embedding of the first 2 non-trivial eigenvectors.
    
    # Parameters for the Laplacian: weights for 1st and 2nd neighbors
    # For a regular polygon, any positive weights work, the eigenvectors are Fourier modes.
    # We use a standard ring graph Laplacian.
    
    def get_points_from_spectral():
        # Create a circulant Laplacian matrix
        # L[i, i] = 2, L[i, i+1] = -1, L[i, i-1] = -1
        L = np.zeros((N, N))
        for i in range(N):
            L[i, i] = 2
            L[i, (i + 1) % N] = -1
            L[i, (i - 1) % N] = -1
        
        # Compute eigenvectors
        # We want the first 2 non-trivial eigenvectors (indices 1 and 2)
        # The trivial one (index 0) is the constant vector.
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # eigh returns eigenvalues in ascending order.
        # Index 0: lambda=0 (constant)
        # Index 1: lambda=2(1-cos(2pi/N)) (first harmonic)
        # Index 2: lambda=2(1-cos(4pi/N)) (second harmonic)
        
        # The coordinates are given by the components of these eigenvectors.
        # v1 = eigvecs[:, 1]
        # v2 = eigvecs[:, 2]
        
        # Note: For a circulant matrix, eigvecs[:, 1] is proportional to cos(2pi*i/N)
        # and eigvecs[:, 2] is proportional to sin(2pi*i/N) (or vice versa, with phase shifts).
        # This forms a regular polygon.
        
        p1 = eigvecs[:, 1]
        p2 = eigvecs[:, 2]
        
        points = np.column_stack((p1, p2))
        
        # Normalize: Center at origin
        points -= np.mean(points, axis=0)
        
        # Scale to unit area? 
        # The score is scale invariant, but normalization helps with numerical stability in optimization.
        # Let's normalize such that the convex hull area is 1.
        try:
            hull = ConvexHull(points)
            area = hull.volume
            if area > 1e-10:
                points = points / np.sqrt(area)
        except Exception:
            pass
            
        return points

    # Initial points from spectral method
    points = get_points_from_spectral()
    
    # 2. Optimization:
    # Refine the points to maximize the minimum triangle area.
    # We optimize the 2N coordinates directly.
    
    def score(x):
        pts = x.reshape(-1, 2)
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
            
        min_area = float('inf')
        n = len(pts)
        # Vectorized calculation of triangle areas for speed
        # Indices
        i, j, k = np.indices((n, n, n))
        mask = (i < j) & (j < k)
        
        # Get points
        pi = pts[i[mask]]
        pj = pts[j[mask]]
        pk = pts[k[mask]]
        
        # Area = 0.5 * |det(pj-pi, pk-pi)|
        # det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        det = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(det)
        
        min_area = np.min(areas)
        
        if min_area == float('inf'):
            return 0.0
            
        return min_area / hull_area

    def objective(x):
        return -score(x)

    # Flatten points for optimization
    x0 = points.flatten()
    
    # Use Nelder-Mead or L-BFGS-B. 
    # Nelder-Mead is derivative-free and robust for small dimensions (26 vars).
    # L-BFGS-B requires gradients. We can use numerical gradients or just Nelder-Mead.
    # Given 120s limit, Nelder-Mead with reasonable maxiter is safe.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < objective(x0):
            points = res.x.reshape(-1, 2)
        else:
            # Fallback to initial spectral points if optimization fails
            points = get_points_from_spectral()
    except Exception:
        points = get_points_from_spectral()

    return points
