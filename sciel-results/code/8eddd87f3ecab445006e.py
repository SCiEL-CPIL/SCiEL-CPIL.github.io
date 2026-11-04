import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Parameterization: 13 edge weights for a cycle graph
    # We use a softplus or exp to ensure weights are positive
    def get_weights(x):
        # x is the parameter vector
        # We can center x to allow optimization around 0
        return np.exp(x)
    
    def objective(x):
        w = get_weights(x)
        # Normalize weights to fix scale? 
        # The embedding scale depends on eigenvalues.
        # Let's just compute the Laplacian
        L = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N
            L[i, j] = -w[i]
            L[j, i] = -w[i]
            L[i, i] += w[i]
            L[j, j] += w[i]
        
        # Make it PD by adding epsilon
        eps = 1e-6
        M = L + eps * np.eye(N)
        
        # Eigendecomposition
        try:
            vals, vecs = np.linalg.eigh(M)
        except Exception:
            return 0.0
            
        # Sort by eigenvalue (eigh returns sorted ascending)
        # The smallest eigenvalue corresponds to the trivial vector (all ones)
        # We want the next two: index 1 and 2
        # Note: If the graph is disconnected or has symmetries, eigenvalues might be degenerate.
        # For a cycle, they are distinct for N>4.
        
        v1 = vecs[:, 1]
        v2 = vecs[:, 2]
        
        # Center the points to remove translation invariance in the embedding?
        # The eigenvectors are orthogonal to v0 (which is 1), so they are already centered (sum to 0).
        
        points = np.column_stack((v1, v2))
        
        # Calculate score
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
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
            
        score = min_area / hull_area
        return -score

    # Initial guess: uniform weights
    x0 = np.zeros(N)
    
    # Optimize
    # Use Nelder-Mead for small dimensional problems without gradients
    # N=13 is small
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    w_opt = get_weights(res.x)
    
    # Reconstruct optimal points
    L = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        L[i, j] = -w_opt[i]
        L[j, i] = -w_opt[i]
        L[i, i] += w_opt[i]
        L[j, j] += w_opt[i]
        
    eps = 1e-6
    M = L + eps * np.eye(N)
    vals, vecs = np.linalg.eigh(M)
    
    v1 = vecs[:, 1]
    v2 = vecs[:, 2]
    points = np.column_stack((v1, v2))
    
    # The points might be very small or large. 
    # The score is scale invariant (min area / hull area).
    # So we can return them as is.
    
    return points
