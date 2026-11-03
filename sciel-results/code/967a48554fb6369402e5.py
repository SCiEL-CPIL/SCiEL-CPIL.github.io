# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import eigh
import warnings

N = 50

def construct():
    """Construct packing of N equal circles in unit circle."""
    # Initialization: Hexagonal grid
    x = np.zeros((N, 2))
    # Generate points on a hex grid within radius 0.9
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            px = (i + 0.5 * (j % 2)) * 0.15
            py = j * 0.15 * np.sqrt(3) / 2
            if px*px + py*py < 0.81:
                pts.append([px, py])
    pts = np.array(pts)
    if len(pts) >= N:
        # Select N points closest to center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        x = pts[idx]
    else:
        # Random fallback
        np.random.seed(42)
        x = np.random.uniform(-0.9, 0.9, (N, 2))
        # Project to circle
        r = np.linalg.norm(x, axis=1)
        mask = r > 0.9
        x[mask] *= 0.9 / r[mask]
        
    # Optimization Loop
    # Stochastic Gradient Ascent on lambda_2
    # With weighted Laplacian w_ij = d_ij
    
    T_max = 5000
    for t in range(T_max):
        # Compute Delaunay
        try:
            tri = Delaunay(x)
        except:
            # Perturb if triangulation fails
            x += np.random.normal(0, 0.01, x.shape)
            continue
            
        # Build neighbor list and weights
        # Weights w_ij = d_ij
        # We need to compute T_i for each i
        # T_i = v_i^2 * sum_{k \in N(i)} u_k - sum_{k \in N(i)} v_k^2 u_k
        # where u_k = (x_i - x_k) / d_ik
        
        # First, get lambda_2 and v_2
        # We need L to get v_2
        A = np.zeros((N, N))
        neighbors = [set() for _ in range(N)]
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    idx_i = simplex[i]
                    idx_j = simplex[j]
                    d = np.linalg.norm(x[idx_i] - x[idx_j])
                    if d > 1e-8:
                        A[idx_i, idx_j] = d
                        A[idx_j, idx_i] = d
                        neighbors[idx_i].add(idx_j)
                        neighbors[idx_j].add(idx_i)
        
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        eigvals, eigvecs = eigh(L)
        v_2 = eigvecs[:, 1]
        lambda_2 = eigvals[1]
        
        # Compute gradient
        grad = np.zeros((N, 2))
        for i in range(N):
            sum_u = np.zeros(2)
            sum_v2_u = np.zeros(2)
            for k in neighbors[i]:
                d_ik = np.linalg.norm(x[i] - x[k])
                if d_ik < 1e-8:
                    continue
                u = (x[i] - x[k]) / d_ik
                sum_u += u
                sum_v2_u += (v_2[k] ** 2) * u
            
            T_i = (v_2[i] ** 2) * sum_u - sum_v2_u
            grad[i] = -2.0 * T_i
        
        # Update
        # Step size decay
        step = 0.01 / (1 + t * 0.001)
        x += step * grad
        
        # Project to unit circle
        r = np.linalg.norm(x, axis=1)
        mask = r > 0.99
        x[mask] *= 0.99 / r[mask]
        
        # Optional: Random restarts if stuck?
        # Check score
        if t % 100 == 0:
            # Calculate actual packing radius
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair = np.min(dist) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
            r_packing = min(min_pair, min_bound)
            # print(f"t={t}, lambda2={lambda_2}, r_packing={r_packing}")
            
    # Final cleanup: ensure inside
    r = np.linalg.norm(x, axis=1)
    mask = r > 0.999
    x[mask] *= 0.999 / r[mask]
    
    return x
# EVOLVE-BLOCK-END
