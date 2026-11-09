"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T."""
        rank = len(U[0]) if U else 0
        T_check = np.zeros((n * m, m * p, p * n), dtype=float)
        for s in range(rank):
            u = np.array(U, dtype=float)[:, s]
            v = np.array(V, dtype=float)[:, s]
            w = np.array(W, dtype=float)[:, s]
            # T_check += np.einsum('a,b,c->abc', u, v, w)
            for a in range(n * m):
                if u[a] == 0:
                    continue
                for b in range(m * p):
                    if v[b] == 0:
                        continue
                    T_check[a, b, :] += u[a] * v[b] * w
        return np.allclose(T_check, T, atol=1e-6)
    
    def schoolbook():
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
        return U, V, W
    
    # Try structured block decomposition: split p into blocks of size 2
    # For each block of 2 columns, use a rank-4 decomposition for 3x4 times 4x2
    def try_block_decomposition():
        """Try to decompose using block structure for the p dimension."""
        if p % 2 != 0:
            return None
        
        # For a 3x4 times 4x2 multiplication, we can potentially use rank < 24
        # Let's try a greedy approach on the full tensor but with structured supports
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # For each pair of k values (k, k+1), try to find a good decomposition
        # The key insight: entries where i, j, k matter
        # T[i*m+j, j*p+k, k*n+i] = 1
        
        # Group by (i, j) pairs and try to cover multiple k values
        for i in range(n):
            for j in range(m):
                # For this (i,j), we need to cover all k in 0..p-1
                # Each k gives: T[i*m+j, j*p+k, k*n+i] = 1
                # A rank-1 term with u supported on {i*m+j}, v on some set, w on some set
                # can cover multiple k if w[k*n+i] is nonzero for those k
                
                # Try to find a w vector that covers multiple k's for this (i,j)
                # w[c] where c = k*n + i
                # We want w[k*n+i] to be nonzero for as many k as possible
                # and v[j*p+k] to be nonzero for those k
                
                # Simple approach: one rank-1 per k (schoolbook for this block)
                for k in range(p):
                    U_cols.append([0] * (n * m))
                    V_cols.append([0] * (m * p))
                    W_cols.append([0] * (p * n))
                    U_cols[-1][i * m + j] = 1
                    V_cols[-1][j * p + k] = 1
                    W_cols[-1][k * n + i] = 1
        
        # This is just schoolbook, so let's try something smarter
        return None
    
    # Greedy rank-1 decomposition with better scoring
    def greedy_decompose(max_rank=96, n_random=2000):
        T_rem = T.copy().astype(float)
        U_cols = []
        V_cols = []
        W_cols = []
        
        for s in range(max_rank):
            if time.time() > deadline - 10:
                break
            
            # Check if T_rem is zero
            if np.max(np.abs(T_rem)) < 1e-10:
                break
            
            best_score = -1
            best_u, best_v, best_w = None, None, None
            
            # Strategy 1: Try supports based on structure
            # For matrix multiplication, natural supports are:
            # u: subset of (i,j) indices
            # v: subset of (j,k) indices  
            # w: subset of (k,i) indices
            
            # Try structured supports first
            for trial in range(n_random):
                if time.time() > deadline - 10:
                    break
                
                if trial < 500:
                    # Structured: pick a subset of i, j, k indices
                    n_i = random.randint(1, min(2, n))
                    n_j = random.randint(1, min(2, m))
                    n_k = random.randint(1, min(2, p))
                    
                    i_set = random.sample(range(n), n_i)
                    j_set = random.sample(range(m), n_j)
                    k_set = random.sample(range(p), n_k)
                    
                    u_support = set()
                    v_support = set()
                    w_support = set()
                    
                    for i in i_set:
                        for j in j_set:
                            u_support.add(i * m + j)
                            for k in k_set:
                                v_support.add(j * p + k)
                                w_support.add(k * n + i)
                    
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    for idx in u_support:
                        u[idx] = random.choice([-1, 1])
                    for idx in v_support:
                        v[idx] = random.choice([-1, 1])
                    for idx in w_support:
                        w[idx] = random.choice([-1, 1])
                else:
                    # Random supports
                    u_support = random.sample(range(n * m), random.randint(1, 3))
                    v_support = random.sample(range(m * p), random.randint(1, 3))
                    w_support = random.sample(range(p * n), random.randint(1, 3))
                    
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    for idx in u_support:
                        u[idx] = random.choice([-1, 1])
                    for idx in v_support:
                        v[idx] = random.choice([-1, 1])
                    for idx in w_support:
                        w[idx] = random.choice([-1, 1])
                
                # Score: sum over a,b,c of T_rem[a,b,c] * u[a] * v[b] * w[c]
                # Only nonzero where all three are nonzero
                score = 0
                for a in u_support if isinstance(u_support, (set, list)) else range(n * m):
                    if u[a] == 0:
                        continue
                    for b in v_support if isinstance(v_support, (set, list)) else range(m * p):
                        if v[b] == 0:
                            continue
                        for c in w_support if isinstance(w_support, (set, list)) else range(p * n):
                            if w[c] == 0:
                                continue
                            score += u[a] * v[b] * w[c] * T_rem[a, b, c]
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_u is None or best_score <= 0:
                break
            
            # Subtract the rank-1 tensor
            for a in range(n * m):
                if best_u[a] == 0:
                    continue
                for b in range(m * p):
                    if best_v[b] == 0:
                        continue
                    for c in range(p * n):
                        if best_w[c] == 0:
                            continue
                        T_rem[a, b, c] -= best_u[a] * best_v[b] * best_w[c]
            
            U_cols.append(best_u.tolist())
            V_cols.append(best_v.tolist())
            W_cols.append(best_w.tolist())
        
        if not U_cols:
            return None
        
        # Transpose to get U, V, W
        rank = len(U_cols)
        U = [[U_cols[s][a] for s in range(rank)] for a in range(n * m)]
        V = [[V_cols[s][b] for s in range(rank)] for b in range(m * p)]
        W = [[W_cols[s][c] for s in range(rank)] for c in range(p * n)]
        
        if validate(U, V, W):
            return U, V, W, rank
        
        return None
    
    # Try greedy with multiple random restarts
    best_result = None
    best_rank = n * m * p + 1
    
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + restart * 1000)
        result = greedy_decompose(max_rank=best_rank, n_random=3000)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
