import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 120s limit, keep margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    dim_u = n * m  # 6
    dim_v = m * p  # 21
    dim_w = p * n  # 14
    
    # Precompute the "valid" (u, v) pairs and their corresponding w
    valid_pairs = {}  # (u, v) -> w
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_pairs[(u, v)] = w
    
    # Precompute b_w for all w
    b_w = np.zeros((dim_w, dim_u * dim_v))
    for (u, v), w in valid_pairs.items():
        b_w[w, u * dim_v + v] = 1.0
    
    # Strategy: Try structured low-rank decompositions based on Strassen-like ideas
    # For m=3, we can try to use 2x2 block structure
    
    # First, let's try a systematic approach with small ranks
    # We'll try ranks from 21 up to 41, but with a more structured search
    
    # Key insight: For matrix multiplication, the tensor has a special structure.
    # We can try to find a decomposition where U, V, W have sparse patterns.
    
    # Try rank 21 first (minimum possible for 2x3x7 is at least 21 by known bounds)
    # Actually, the minimum rank for 2x3x7 is not known exactly, but it's between 21 and 42.
    
    # Let's try a different approach: use the fact that m=3 is small.
    # We can think of the multiplication as:
    # A (2x3) * B (3x7) = C (2x7)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For each output entry C[i,k], we need to compute sum_j A[i,j] * B[j,k]
    # This is a sum of 3 products.
    
    # Can we do better than 3 products per output entry?
    # For 2x2x2, Strassen uses 7 products instead of 8.
    # For 2x3x7, maybe we can use some sharing.
    
    # Let's try a structured decomposition.
    # Suppose we use r rank-1 terms.
    # Each term t contributes: U[:,t] * V[:,t] * W[:,t]
    # U[:,t] is a 6-vector, V[:,t] is a 21-vector, W[:,t] is a 14-vector.
    
    # The constraint is that for each (i,j,k):
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # And for all other (u,v,w) combinations, the sum should be 0.
    
    # Let's try to construct a solution with a specific pattern.
    # For m=3, we can try to use 2x2 Strassen-like blocks.
    
    # Actually, let's just try the random search more efficiently.
    # The key is to find U and V such that the system A @ W[w,:] = b_w[w,:] has an exact solution.
    
    # For a given U and V, A is determined. The system is consistent if and only if
    # b_w[w,:] is in the column space of A for all w.
    
    # Let's try random U and V with entries from a small set {-1, 0, 1} to increase the chance of exact solutions.
    
    for r in range(21, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try random initializations for U and V
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            # Random U and V with -1/0/1 entries
            U = np.random.randint(-1, 2, size=(dim_u, r))
            V = np.random.randint(-1, 2, size=(dim_v, r))
            
            # Build A: A[(u * dim_v + v), t] = U[u, t] * V[v, t]
            A = np.zeros((dim_u * dim_v, r))
            for t in range(r):
                A[:, t] = np.kron(U[:, t], V[:, t])
            
            # For each w, solve A @ W[w, :] = b_w[w, :]
            valid = True
            W = np.zeros((dim_w, r))
            for w in range(dim_w):
                try:
                    W[w, :], residuals, rank_sol, s = np.linalg.lstsq(A, b_w[w], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if the solution is exact
            exact = True
            for w in range(dim_w):
                T_recon_w = (U * W[w, :][None, :]) @ V.T
                T_recon_w_flat = T_recon_w.flatten()
                
                if not np.allclose(T_recon_w_flat, b_w[w], atol=1e-9):
                    exact = False
                    break
            
            if exact:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Stop searching for this rank
        
        if best_rank < rank_school:
            break

    return best_result
