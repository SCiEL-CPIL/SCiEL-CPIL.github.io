import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

N = 13

def construct():
    # Parameters to optimize:
    # We will parameterize the edge weights of a cycle graph.
    # w_i = base_weight + perturbation
    # To ensure connectivity and SPD, we keep weights positive.
    
    def get_points(params):
        # params: array of length 2*2 (for 2 Fourier modes of perturbation)
        # Or just a vector of length N for weights? N=13 is small.
        # Let's use 4 parameters to define a smooth perturbation of the weights.
        # w_i = 1.0 + a1*cos(2*pi*i/N) + b1*sin(2*pi*i/N) + a2*cos(4*pi*i/N) + b2*sin(4*pi*i/N)
        
        a1, b1, a2, b2 = params
        
        # Ensure weights are positive
        # Max perturbation magnitude should be < 1
        # Let's scale params to be in [-0.5, 0.5]
        a1 = np.clip(a1, -0.5, 0.5)
        b1 = np.clip(b1, -0.5, 0.5)
        a2 = np.clip(a2, -0.5, 0.5)
        b2 = np.clip(b2, -0.5, 0.5)
        
        i = np.arange(N)
        # Base weight 1.0
        weights = 1.0 + a1 * np.cos(2 * np.pi * i / N) + b1 * np.sin(2 * np.pi * i / N) + \
                      a2 * np.cos(4 * np.pi * i / N) + b2 * np.sin(4 * np.pi * i / N)
        
        # Ensure weights are strictly positive
        weights = np.maximum(weights, 0.1)
        
        # Construct Laplacian of the cycle graph
        # L = D - A
        # For a cycle, degree is 2 (weighted).
        # A[i, i+1] = w_i, A[i+1, i] = w_i (cyclic)
        
        # Using sparse matrix
        main_diag = np.zeros(N)
        off_diag1 = np.zeros(N)
        off_diag2 = np.zeros(N)
        
        for i in range(N):
            # Edge between i and (i+1)%N has weight w_i
            w = weights[i]
            off_diag1[i] = -w
            off_diag2[(i+1)%N] = -w # This is not quite right for diags
            
        # Let's build it manually or use diags correctly
        # diags(data, offsets)
        # offset 0: main
        # offset 1: super-diagonal
        # offset -1: sub-diagonal
        # For cycle, we need (0, N-1) and (N-1, 0)
        
        main = np.zeros(N)
        super_diag = np.zeros(N-1)
        sub_diag = np.zeros(N-1)
        
        for i in range(N-1):
            w = weights[i]
            super_diag[i] = -w
            sub_diag[i] = -w
            main[i] += w
            main[i+1] += w
            
        # Edge N-1 to 0
        w_last = weights[N-1]
        main[N-1] += w_last
        main[0] += w_last
        
        # Construct sparse matrix
        # We need to handle the cyclic boundary in the sparse diags
        # scipy.sparse.diags doesn't handle wrap-around easily.
        # Let's use a dense matrix since N=13 is small.
        
        L = np.zeros((N, N))
        for i in range(N):
            j1 = (i + 1) % N
            j2 = (i - 1) % N
            w1 = weights[i] # weight of edge i -> i+1
            w2 = weights[(i-1)%N] # weight of edge i-1 -> i
            
            L[i, i] = w1 + w2
            L[i, j1] = -w1
            L[i, j2] = -w2
            
        # Compute eigenvalues and eigenvectors
        # We want the first 2 non-trivial eigenvectors (smallest non-zero eigenvalues)
        # The Laplacian has a zero eigenvalue with eigenvector [1,1,...,1].
        # We can project out the mean or just take the next two.
        
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Sort by eigenvalue
        # eigh returns sorted ascending
        # eigvals[0] should be ~0
        # eigvecs[:, 1] is the first non-trivial
        # eigvecs[:, 2] is the second non-trivial
        
        u1 = eigvecs[:, 1]
        u2 = eigvecs[:, 2]
        
        # Center the points to (0,0)
        u1 = u1 - np.mean(u1)
        u2 = u2 - np.mean(u2)
        
        points = np.column_stack((u1, u2))
        
        return points

    def objective(params):
        pts = get_points(params)
        
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

    # Initial guess: regular polygon (zero perturbation)
    x0 = np.zeros(4)
    
    # Optimize
    # Use Nelder-Mead or Powell as they don't require gradients and are robust for low-dim
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = result.x
    points = get_points(best_params)
    
    return points
