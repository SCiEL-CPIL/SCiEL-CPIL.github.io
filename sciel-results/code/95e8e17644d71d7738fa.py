import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Focus on 2x3x3 subproblem, then tile 3 times for 2x3x9
    n_s, m_s, p_s = 2, 3, 3
    # Precompute valid (a,b,c) indices for the tensor
    valid_indices = []
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                a = i * m_s + j
                b = j * p_s + k
                c = k * n_s + i
                valid_indices.append((a, b, c))
    
    # Also precompute inconsistent pairs
    inconsistent_pairs = []
    for a in range(n_s * m_s):
        for b in range(m_s * p_s):
            i_a = a // m_s
            j_a = a % m_s
            j_b = b // p_s
            k_b = b % p_s
            if j_a != j_b:
                inconsistent_pairs.append((a, b))
    
    def verify_decomposition(U, V, W, rank):
        """Fast vectorized verification. Returns True if valid."""
        # Check valid entries equal 1
        for a, b, c in valid_indices:
            val = np.dot(U[a], V[b]) * 0  # placeholder
            val = np.sum(U[a] * V[b] * W[c])
            if val != 1:
                return False
        # Check inconsistent entries equal 0
        for a, b in inconsistent_pairs:
            i_a = a // m_s
            j_a = a % m_s
            j_b = b // p_s
            k_b = b % p_s
            # For each possible c, check U[a]·V[b]·W[c] summed... 
            # Actually T[a,b,c] = sum_rank U[a,r]*V[b,r]*W[c,r]
            # This must be 0 for all c when a,b inconsistent
            # Vectorize: for each c, compute sum_r U[a,r]*V[b,r]*W[c,r]
            uv = U[a] * V[b]  # shape (rank,)
            for c in range(n_s * p_s):
                if np.sum(uv * W[c]) != 0:
                    return False
        return True

    def verify_decomposition_fast(U, V, W, rank):
        """Even faster: build full tensor via outer products."""
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U @ V.T) doesn't work directly... 
        # Actually: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We can check: for each r, add outer product
        # But that's O(rank * n*m*p) which is fine for small dims
        # Let's just check the specific entries we care about
        # Valid entries:
        for a, b, c in valid_indices:
            if np.sum(U[a] * V[b] * W[c]) != 1:
                return False
        # Inconsistent entries: for each (a,b) inconsistent, T[a,b,c]=0 for all c
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U[a] * V[b]) · W[c]  (dot product of element-wise product with W[c])
        for a, b in inconsistent_pairs:
            uv = U[a] * V[b]
            # Check uv · W[c] = 0 for all c
            # This is equivalent to W.T @ uv = 0 vector
            if np.any(W.T @ uv != 0):
                return False
        return True

    best_rank = None
    best_UVW = None
    
    # Try ranks from 15 up to 18 for 2x3x3
    for target_rank in [15, 16, 17, 18]:
        if time.time() > deadline - 30:
            break
        
        max_trials = 20000 if target_rank <= 15 else 50000
        for t in range(max_trials):
            if time.time() > deadline - 30:
                break
            
            # Random integer initialization in {-2,-1,0,1,2}
            U = np.random.randint(-2, 3, size=(n_s * m_s, target_rank))
            V = np.random.randint(-2, 3, size=(m_s * p_s, target_rank))
            W = np.random.randint(-2, 3, size=(p_s * n_s, target_rank))
            
            if verify_decomposition_fast(U, V, W, target_rank):
                best_rank = target_rank
                best_UVW = (U, V, W)
                break
        
        if best_rank is not None:
            break
    
    if best_rank is not None:
        U_s, V_s, W_s = best_UVW
        rank_full = best_rank * 3
        U_full = np.zeros((n_rows, rank_full), dtype=np.int64)
        V_full = np.zeros((m_rows, rank_full), dtype=np.int64)
        W_full = np.zeros((p_rows, rank_full), dtype=np.int64)
        
        for block in range(3):
            U_full[:, block * best_rank : (block + 1) * best_rank] = U_s
            
            for j in range(m_s):
                for k in range(p_s):
                    v_idx_full = j * p + block * p_s + k
                    v_idx_small = j * p_s + k
                    V_full[v_idx_full, block * best_rank : (block + 1) * best_rank] = V_s[v_idx_small, :]
                    
            for k_s in range(p_s):
                for i in range(n_s):
                    k_full = block * p_s + k_s
                    w_idx_full = k_full * n + i
                    w_idx_small = k_s * n + i
                    W_full[w_idx_full, block * best_rank : (block + 1) * best_rank] = W_s[w_idx_small, :]
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        return schoolbook()
