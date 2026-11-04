"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. MDS Embedding of locations
    # b is distance matrix. We need squared distances for MDS.
    D = b ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D @ J
    # Symmetrize M to handle numerical errors
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Take top d eigenvectors (use full rank d=n for simplicity and accuracy in relaxation)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure eigenvalues are non-negative (clip small negative values)
    eigenvalues = np.maximum(eigenvalues, 0)
    Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    # Gram matrix of locations
    G = Y.T @ Y
    
    # 2. Dynamics Simulation
    # Initialize X (facility coordinates) deterministically
    # Start with X = Y (identity assignment)
    X = Y.copy()
    V = np.zeros_like(X)
    
    # Symplectic Integrator: Velocity Verlet
    # Potential V(X) = -tr(A X G X^T)
    # Gradient = -2 A X G
    # We want to minimize Cost = tr(A P B P^T) approx -tr(A X G X^T)
    # So we follow the gradient of the potential which is the negative of the cost gradient?
    # Cost ~ -tr(A X G X^T).
    # We want to minimize Cost.
    # So we want to increase tr(A X G X^T).
    # Let's define Energy E(X) = -tr(A X G X^T).
    # Force F = -grad E = 2 A X G.
    # Acceleration a = F / m (m=1).
    
    dt = 0.01
    num_steps = 500
    sample_freq = 10
    
    candidates = []
    
    for step in range(num_steps):
        # Compute force at current X
        F = 2 * (a @ X) @ G
        
        # Half step position
        X_half = X + 0.5 * dt * V
        # Project X_half to Stiefel manifold (orthogonalize)
        U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
        X_half_proj = U @ Vt
        
        # Compute force at X_half
        F_half = 2 * (a @ X_half_proj) @ G
        
        # Update velocity
        V = V + 0.5 * dt * (F + F_half)
        
        # Full step position
        X = X_half_proj + 0.5 * dt * V
        # Project X to Stiefel manifold
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = U @ Vt
        
        # Sample state
        if step % sample_freq == 0:
            # Discretize X to permutation
            # Cost matrix C[i, k] = ||X[i] - Y[k]||^2
            # ||x - y||^2 = x.x + y.y - 2 x.y
            # C = sum(X^2, axis=1)[:, None] + sum(Y^2, axis=1)[None, :] - 2 * X @ Y.T
            C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
            row_ind, col_ind = linear_sum_assignment(C)
            p = col_ind.copy()
            # Verify p is a permutation
            # Compute cost
            D_p = b[p[:, None], p[None, :]]
            cost = np.sum(a * D_p)
            candidates.append((cost, p))
    
    # 3. Local Search on best candidates
    # Sort candidates by cost
    candidates.sort(key=lambda x: x[0])
    
    best_p = None
    best_cost = float('inf')
    seen = set()
    
    start_time = time.time()
    time_limit = 110
    
    # Precompute interaction matrices for fast delta calculation
    # M[i, v] = sum_k a[i, k] * b[v, k]
    # M[i, v] = a[i, :] @ b[v, :] -> No, b is symmetric, so b[v, k] = b[k, v].
    # M[i, v] = sum_k a[i, k] * b[k, v] = (a @ b.T)[i, v] = (a @ b)[i, v]
    M = a @ b
    # M2[i, v] = sum_k a[k, i] * b[k, v] = (a.T @ b)[i, v]
    M2 = a.T @ b
    
    for cost, p in candidates:
        if (time.time() - start_time) > time_limit - 10:
            break
        p_tuple = tuple(p)
        if p_tuple in seen:
            continue
        seen.add(p_tuple)
        
        # Run 2-opt local search with vectorized delta
        p = p.copy()
        current_cost = cost
        
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5:
            improved = False
            for i in range(n):
                if (time.time() - start_time) >= time_limit - 5:
                    break
                u = p[i]
                
                # Calculate delta for swapping i with all j > i
                # We only need to check j > i to avoid duplicates, but for best improvement we can check all j != i
                # Let's check all j != i for simplicity in vectorization, then pick best
                js = np.arange(n)
                js = js[js != i]
                vs = p[js]
                
                # S1 = M[i, vs] (vector)
                # S2 = M[i, u] (scalar)
                # S3 = M2[i, vs] (vector)
                # S4 = M2[i, u] (scalar)
                # S5 = M[js, u] (vector)
                # S6 = M[js, vs] (vector)
                # S7 = M2[js, u] (vector)
                # S8 = M2[js, vs] (vector)
                
                S1 = M[i, vs]
                S2 = M[i, u]
                S3 = M2[i, vs]
                S4 = M2[i, u]
                S5 = M[js, u]
                S6 = M[js, vs]
                S7 = M2[js, u]
                S8 = M2[js, vs]
                
                # Correction terms for k=i and k=j
                # Term_k_i = 2 * (b[vs, i] - b[u, i]) * (a[i, i] - a[js, i])
                # Note: b is symmetric. b[vs, i] is vector, b[u, i] is scalar.
                # a[i, i] is scalar, a[js, i] is vector.
                corr_i = 2 * (b[vs, i] - b[u, i]) * (a[i, i] - a[js, i])
                
                # Term_k_j = 2 * (b[vs, js] - b[u, js]) * (a[i, js] - a[js, js])
                # b[vs, js] is diagonal of b[vs, js.T]? No.
                # vs is array of locations. js is array of indices.
                # b[vs, js] gives b[v_j, j] for each j.
                # b[u, js] gives b[u, j] for each j.
                # a[i, js] gives a[i, j] for each j.
                # a[js, js] gives a[j, j] for each j.
                corr_j = 2 * (b[vs, js] - b[u, js]) * (a[i, js] - a[js, js])
                
                # Direct terms
                # a[i, js] * (b[vs, u] - b[u, vs]) + a[js, i] * (b[u, vs] - b[vs, u])
                # b[vs, u] is vector b[v, u]. b[u, vs] is vector b[u, v].
                # Since b is symmetric, b[v, u] = b[u, v]. So this term is 0?
                # Wait. b[vs, u] is b[v, u]. b[u, vs] is b[u, v].
                # If b is symmetric, they are equal.
                # Let's check the original formula:
                # a[i, j] * (b[y, x] - b[x, y]) where x=u, y=v.
                # b[v, u] - b[u, v]. If symmetric, this is 0.
                # So direct terms vanish if b is symmetric.
                # QAP distance matrices are symmetric.
                
                deltas = (S1 - S2 + S3 - S4 + S5 - S6 + S7 - S8) - corr_i - corr_j
                
                best_idx = np.argmin(deltas)
                min_delta = deltas[best_idx]
                
                if min_delta < 0:
                    best_j = js[best_idx]
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += min_delta
                    improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    if best_p is None:
        # Fallback to initial heuristic if dynamics failed
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
