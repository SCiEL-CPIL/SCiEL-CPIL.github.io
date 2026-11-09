import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-6):
    # Check that sum_c U[:,c] V[:,c]^T W[:,c]^T equals the matmul tensor
    # The tensor T[i,j,k,c] should satisfy:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    rank = U.shape[1]
    # Build T from decomposition
    T_decomp = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Build T from matmul: T[i,j,k] = delta (it's just 1 for the "product" tensor)
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor for matmul is: T[i,j,k] such that (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The rank-1 tensors are U * V * W where U[i,j] = e_i e_j^T, V[j,k] = e_j e_k^T, W[k,i] = e_k e_i^T
    # So T[i,j,k] = 1 for all i,j,k in the sum over c? No, each rank-1 term picks out one (i,j,k).
    # The full tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in this indexing).
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)


def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower rank via random search
    # The tensor T[i,j,k] = 1 for all i,j,k needs rank-r decomposition
    # Each rank-1 term: u[i,j] * v[j,k] * w[k,i]
    # We need sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
    
    # This is a hard problem. Let's try a structured approach.
    # For 3x3x3, best known is 23. For 3x3x5, we can do:
    # Split B into 3x3 and 3x2 parts.
    # Rank(3x3x3) + Rank(3x3x2)
    # If we can get 23 + 17 = 40, that's better than 45.
    # Rank(3x3x2) schoolbook is 18, but maybe 17 is possible?
    
    # Let's try random search for small ranks first
    np.random.seed(42)
    
    for target_rank in range(35, 45):
        if time.time() > deadline - 30:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Random initialization with sparse entries
            for c in range(target_rank):
                # Pick random non-zero pattern
                n_nonzero_U = np.random.randint(1, 4)
                n_nonzero_V = np.random.randint(1, 4)
                n_nonzero_W = np.random.randint(1, 4)
                
                idx_U = np.random.choice(n*m, size=n_nonzero_U, replace=False)
                idx_V = np.random.choice(m*p, size=n_nonzero_V, replace=False)
                idx_W = np.random.choice(p*n, size=n_nonzero_W, replace=False)
                
                for idx in idx_U:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in idx_V:
                    V[idx, c] = np.random.choice([-1, 1])
                for idx in idx_W:
                    W[idx, c] = np.random.choice([-1, 1])
            
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                found = True
                break
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
