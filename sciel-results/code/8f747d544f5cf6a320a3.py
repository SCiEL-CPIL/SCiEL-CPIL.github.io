"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill

    # The tensor T has shape (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if valid (i,j,k), else 0.
    # Since n=2, the tensor decomposes into two independent blocks for i=0 and i=1.
    # Block i: T_i has shape (m, m*p, p)
    # T_i[j, b, c'] = 1 if b = j*p + k and c' = k (since c = k*n + i, and for fixed i, c' is the k-index)
    # Actually, let's just build the full tensor and solve for the full rank, 
    # but we can exploit the block structure to reduce search space or verify easier.
    
    # Full tensor construction
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0

    # We want T_full[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Strategy:
    # 1. Try to find a rank-r decomposition for the full tensor.
    # 2. Since n=2, we can look for a decomposition where the U, V, W matrices have a specific structure
    #    or simply search for any rank-r decomposition.
    # 3. A constructive approach: 
    #    For 1x4x7 multiplication, the rank is at most 4*7=28. 
    #    With n=2, total rank <= 56.
    #    Can we do better? 
    #    For 1xm x mp, the tensor is T[j, b, c] = delta_{b, j*p+c} (if we view output as vector of length p).
    #    This is equivalent to 7 independent 1x4 scalar multiplications? No.
    #    C[k] = sum_j A[j] B[j, k].
    #    This is a set of 7 linear forms.
    #    The tensor is T[j, (j', k), k'] = 1 if j=j' and k=k'.
    #    This is a block diagonal tensor in the middle index? No.
    #    Let's analyze the 1x4x7 tensor T_1.
    #    T_1[j, b, c] = 1 iff b = j*p + c.
    #    This is a "slice" of the identity.
    #    The rank of this tensor is exactly m*p = 28? 
    #    Let's check if we can do better.
    #    If we use Winograd or other schemes, maybe.
    #    However, for 1xm x mp, it's just 7 separate dot products of length m.
    #    Each dot product of length 4 has rank 4 (standard) or 3 (Strassen for 1x2x2 extended?)
    #    Actually, 1x4 dot product can be done with rank 4? 
    #    Wait, 1x2 dot product: a1*b1 + a2*b2. Rank 2? No, rank 2 is trivial. Rank 3 is Strassen? No, Strassen is 2x2.
    #    For 1x2, rank is 2. (a1b1, a2b2).
    #    For 1x4, rank is 4? 
    #    Is there a rank 3 scheme for 1x4?
    #    Let's try to find a rank-r decomposition for the 1x4x7 tensor first.
    #    If we find rank r1 for 1x4x7, then for 2x4x7, we might be able to use 2*r1 or maybe less if we mix?
    #    But the outputs are separate (C[0,k] and C[1,k]).
    #    The inputs are separate (A[0,j] and A[1,j]).
    #    The middle matrix B is shared!
    #    This is the key. B is shared between the two rows.
    #    So we can potentially share terms in the decomposition.
    
    # Let's use a generic solver for small ranks.
    # We will try to find U, V, W of rank r such that sum_r U[:,r] V[:,r]^T W[:,r]^T = T (flattened appropriately)
    
    def try_rank(r):
        """
        Attempt to find a rank-r decomposition.
        Uses a grid search on the support of V (middle matrix) because m*p=28 is smallish,
        but 28 choose r is too big.
        
        Alternative: Randomized ALS with restarts and exact solving.
        """
        if time.time() > deadline:
            return None
            
        # We'll use a heuristic: 
        # Fix V to be a random sparse matrix or random dense, solve for U and W.
        # Or better: Use the fact that T is sparse.
        
        # Let's try a structured approach for the 1x4x7 case first, then combine.
        # Actually, let's just run a robust ALS on the full tensor with multiple restarts.
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        num_restarts = 20
        max_iters = 100
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed a: T[a, :, :] = sum_r U[a, r] (V[:, r] \otimes W[:, r])
                # Let M_r = V[:, r] \otimes W[:, r] (vector of length m*p * p*n)
                # T_a = T[a].reshape(-1)
                # T_a = U[a] @ M^T where M is (r, m*p*p*n)
                
                M_U = np.zeros((r, m*p * p*n))
                for r_idx in range(r):
                    M_U[r_idx, :] = np.kron(V[:, r_idx], W[:, r_idx])
                    
                T_U = T_full.reshape(n*m, -1)
                
                # Solve U[a] @ M_U.T = T_U[a]
                # U[a] = T_U[a] @ pinv(M_U.T)
                try:
                    pinv_M_U = np.linalg.pinv(M_U.T)
                    U_new = T_U @ pinv_M_U
                except:
                    return None
                
                # Update V
                M_V = np.zeros((r, n*m * p*n))
                for r_idx in range(r):
                    M_V[r_idx, :] = np.kron(U[:, r_idx], W[:, r_idx])
                    
                T_V = T_full.transpose(1, 0, 2).reshape(m*p, -1)
                
                try:
                    pinv_M_V = np.linalg.pinv(M_V.T)
                    V_new = T_V @ pinv_M_V
                except:
                    return None
                    
                # Update W
                M_W = np.zeros((r, n*m * m*p))
                for r_idx in range(r):
                    M_W[r_idx, :] = np.kron(U[:, r_idx], V[:, r_idx])
                    
                T_W = T_full.transpose(2, 0, 1).reshape(p*n, -1)
                
                try:
                    pinv_M_W = np.linalg.pinv(M_W.T)
                    W_new = T_W @ pinv_M_W
                except:
                    return None
                    
                U, V, W = U_new, V_new, W_new
                
                # Calculate error
                T_approx = np.zeros_like(T_full)
                for r_idx in range(r):
                    T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    
                err = np.linalg.norm(T_full - T_approx) / np.linalg.norm(T_full)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < 1e-10:
                    break
                    
        if best_err < 1e-9:
            return best_U, best_V, best_W, best_err
        return None

    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) is not a lower bound.
    # Trivial rank is n*m*p = 56.
    # For 1x4x7, rank is likely 28.
    # For 2x4x7 with shared B, maybe we can do better than 56?
    # If we use 2x Strassen-like blocks?
    # Let's try from 30 up to 56.
    
    for r in range(30, 57):
        if time.time() > deadline:
            break
            
        res = try_rank(r)
        if res is not None:
            U, V, W, err = res
            
            # Round to half-integers or integers if close
            def clean_matrix(M):
                M_rounded = np.round(M * 2) / 2
                if np.allclose(M, M_rounded, atol=1e-6):
                    return M_rounded
                M_int = np.round(M)
                if np.allclose(M, M_int, atol=1e-6):
                    return M_int
                return M
            
            U = clean_matrix(U)
            V = clean_matrix(V)
            W = clean_matrix(W)
            
            # Verify
            T_approx = np.zeros_like(T_full)
            for r_idx in range(r):
                T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                
            if np.allclose(T_full, T_approx, atol=1e-6):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback: Schoolbook rank 56
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
