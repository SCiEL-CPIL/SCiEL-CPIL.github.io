import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Build the tensor index mapping
    # U indexed by (i*m + j), V by (j*p + k), W by (k*n + i)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook baseline (rank 36)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_result = to_lists(U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)  # 36
    
    def verify(U, V, W, tol=1e-7):
        """Check if decomposition is exact."""
        rank = U.shape[1]
        for (i, j, k) in triples:
            s = 0.0
            for r in range(rank):
                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if abs(s - 1.0) > tol:
                return False
        return True
    
    def try_rank(R, seed=42, max_iter=200):
        """Try to find a rank-R decomposition using coordinate descent / alternating minimization."""
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        
        # Initialize with structured random values
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, R))
        
        # Make sure no column is all zeros
        for r in range(R):
            if np.all(U[:, r] == 0):
                U[np.random.randint(n*m), r] = 1
            if np.all(V[:, r] == 0):
                V[np.random.randint(m*p), r] = 1
            if np.all(W[:, r] == 0):
                W[np.random.randint(p*n), r] = 1
        
        def compute_residual(U, V, W):
            """Compute sum of squared residuals."""
            res = 0.0
            for (i, j, k) in triples:
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                res += (s - 1.0)**2
            return res
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Alternating minimization: fix two, optimize the third
            # For each (i,j,k), the contribution to the residual from variable U[i*m+j, r] is:
            # (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
            # This is quadratic in U[i*m+j, r] when V and W are fixed.
            
            # Optimize U
            for idx in range(n*m):
                # Collect all (j,k) pairs that this U entry participates in
                # U[idx] = U[i*m+j] where idx = i*m+j
                i = idx // m
                j = idx % m
                # This U entry is used for all k in range(p)
                # For each k: residual_k = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # Let a_r = V[j*p+k, r] * W[k*n+i, r]
                # Then residual_k = dot(U[idx], a) - 1
                # We want to minimize sum_k (dot(U[idx], a_k) - 1)^2
                # This is: U[idx]^T (sum_k a_k a_k^T) U[idx] - 2 * sum_k a_k^T U[idx] + p
                # Gradient: 2 * (sum_k a_k a_k^T) U[idx] - 2 * sum_k a_k = 0
                # So: (sum_k a_k a_k^T) U[idx] = sum_k a_k
                
                A_mat = np.zeros((R, R))
                b_vec = np.zeros(R)
                for k in range(p):
                    a = V[j*p+k, :] * W[k*n+i, :]
                    A_mat += np.outer(a, a)
                    b_vec += a
                
                # Solve A_mat * U[idx] = b_vec
                try:
                    U_new = np.linalg.solve(A_mat + 1e-10*np.eye(R), b_vec)
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U[idx, :] = U_new
            
            # Optimize V
            for idx in range(m*p):
                j = idx // p
                k = idx % p
                # V[idx] = V[j*p+k]
                # Used for all i in range(n)
                # For each i: residual_i = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # Let a_r = U[i*m+j, r] * W[k*n+i, r]
                # residual_i = dot(V[idx], a_i) - 1
                A_mat = np.zeros((R, R))
                b_vec = np.zeros(R)
                for i in range(n):
                    a = U[i*m+j, :] * W[k*n+i, :]
                    A_mat += np.outer(a, a)
                    b_vec += a
                try:
                    V_new = np.linalg.solve(A_mat + 1e-10*np.eye(R), b_vec)
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V[idx, :] = V_new
            
            # Optimize W
            for idx in range(p*n):
                k = idx // n
                i = idx % n
                # W[idx] = W[k*n+i]
                # Used for all j in range(m)
                # For each j: residual_j = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # Let a_r = U[i*m+j, r] * V[j*p+k, r]
                # residual_j = dot(W[idx], a_j) - 1
                A_mat = np.zeros((R, R))
                b_vec = np.zeros(R)
                for j in range(m):
                    a = U[i*m+j, :] * V[j*p+k, :]
                    A_mat += np.outer(a, a)
                    b_vec += a
                try:
                    W_new = np.linalg.solve(A_mat + 1e-10*np.eye(R), b_vec)
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W[idx, :] = W_new
            
            # Check convergence
            res = compute_residual(U, V, W)
            if res < 1e-12:
                # Snap to nice values if possible
                return U, V, W
        
        # Final check
        res = compute_residual(U, V, W)
        if res < 1e-10:
            return U, V, W
        return None
    
    # Try ranks from 24 upward
    for R in range(24, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 10:
                break
            
            result = try_rank(R, seed=seed, max_iter=300)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_rank = R
                    best_result = to_lists(U, V, W)
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    return best_result
