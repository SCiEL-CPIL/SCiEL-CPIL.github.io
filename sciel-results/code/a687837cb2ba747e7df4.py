import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
K = 4  # Number of eigenvectors to use for projection (excluding constant)

def construct():
    # Initial parameters
    # We want B to look like a cycle graph Laplacian initially to get a regular polygon
    # Cycle graph Laplacian has 2 on diagonal, -1 on sub/super diagonal
    # We parametrize B = diag(d) + off_diag(u)
    # To ensure SPD, we set d = 1 + 2*|u| roughly, or just optimize d and u with constraints
    
    # Initial guess: Regular polygon
    # For a regular polygon, the eigenvectors of the cycle graph are sinusoids.
    # We can start with d = 2, u = 1 (which gives eigenvalues 2 - 2cos(2pi k/N))
    d_init = np.ones(N) * 2.0
    u_init = np.ones(N - 1) * 1.0
    
    # Projection matrix C (K x 2)
    # For regular polygon, we use the 2nd and 3rd eigenvectors (indices 1, 2)
    # The others should be 0.
    C_init = np.zeros((K, 2))
    C_init[0, 0] = 1.0  # First non-trivial eigenvector to x-coord
    C_init[1, 1] = 1.0  # Second non-trivial eigenvector to y-coord
    
    x0 = np.concatenate([d_init, u_init, C_init.flatten()])
    
    def get_points(x):
        d = x[:N]
        u = x[N:2*N-1]
        C = x[2*N-1:].reshape(K, 2)
        
        # Construct Banded SPD Matrix B
        # B is N x N symmetric
        # Diagonal is d
        # Sub/Super diagonal is u
        # To ensure SPD, we rely on the optimizer, but we can add a small regularization
        # If u is too large relative to d, it might not be SPD.
        # Let's constrain d > 2*abs(u) implicitly by scaling or just let it fail and return bad score.
        
        B = np.zeros((N, N))
        for i in range(N):
            B[i, i] = d[i]
        for i in range(N - 1):
            B[i, i+1] = u[i]
            B[i+1, i] = u[i]
            
        # Add small identity for numerical stability
        B += 1e-6 * np.eye(N)
        
        try:
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            
            # Sort by eigenvalues (eigh already returns them sorted ascending)
            # Take first K+1 eigenvectors (skip index 0 which is constant mode for Laplacian-like)
            # If the graph is not connected or d is not uniform, the first mode might not be constant.
            # But for smooth convex shapes, we usually use the low-frequency modes.
            
            # We use modes 1 through K (indices 1 to K)
            # This gives K vectors of length N
            V_selected = eigenvectors[:, 1:K+1]
            
            # Project to 2D
            points = V_selected @ C
            
            return points
            
        except Exception:
            # If matrix is not SPD or other numerical issues
            return None

    def objective(x):
        points = get_points(x)
        if points is None:
            return 1e6
        
        # Ensure points are not degenerate
        if np.allclose(points, points[0]):
            return 1e6
            
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6
            
        if hull_area < 1e-10:
            return 1e6
            
        # Calculate minimum triangle area
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
            return 1e6
            
        score = min_area / hull_area
        return -score

    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
    
    best_points = get_points(result.x)
    
    # Fallback to regular polygon if optimization fails or produces bad result
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        # Sanity check: if the result is worse than the regular polygon, use regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Quick score comparison
        try:
            hull_reg = ConvexHull(reg_points)
            min_area_reg = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(j + 1, N):
                        p1 = reg_points[i]
                        p2 = reg_points[j]
                        p3 = reg_points[k]
                        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                        if area < min_area_reg:
                            min_area_reg = area
            score_reg = min_area_reg / hull_reg.volume
            
            hull_opt = ConvexHull(best_points)
            min_area_opt = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(j + 1, N):
                        p1 = best_points[i]
                        p2 = best_points[j]
                        p3 = best_points[k]
                        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                        if area < min_area_opt:
                            min_area_opt = area
            score_opt = min_area_opt / hull_opt.volume
            
            if score_opt > score_reg:
                best_points = reg_points
        except Exception:
            best_points = reg_points
            
    return best_points
