import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def verify_decomposition(U, V, W, rank):
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        T_decomp = np.zeros_like(T)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        return np.allclose(T, T_decomp, atol=1e-10)
    
    rank = 18
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    
    # Block 1: A_L (2x2) * B_1a (2x2) -> Rank 7 (Strassen)
    # A_L: i in {0,1}, j in {0,1}
    # B_1a: j in {0,1}, k in {0,1}
    # W: k in {0,1}, i in {0,1}
    
    # Strassen formulas for 2x2 * 2x2
    # M1: (A00+A11)(B00+B11)
    U[u_idx(0,0), col] += 1
    U[u_idx(1,1), col] += 1
    V[v_idx(0,0), col] += 1
    V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M2: (A10+A11)B00
    U[u_idx(1,0), col] += 1
    U[u_idx(1,1), col] += 1
    V[v_idx(0,0), col] += 1
    W[w_idx(0,1), col] += 1
    W[w_idx(1,1), col] -= 1
    col += 1
    
    # M3: A00(B01-B11)
    U[u_idx(0,0), col] += 1
    V[v_idx(0,1), col] += 1
    V[v_idx(1,1), col] -= 1
    W[w_idx(1,0), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M4: A11(B10-B00)
    U[u_idx(1,1), col] += 1
    V[v_idx(1,0), col] += 1
    V[v_idx(0,0), col] -= 1
    W[w_idx(0,0), col] += 1
    W[w_idx(0,1), col] += 1
    col += 1
    
    # M5: (A00+A01)B11
    U[u_idx(0,0), col] += 1
    U[u_idx(0,1), col] += 1
    V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] -= 1
    W[w_idx(1,0), col] += 1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[u_idx(1,0), col] += 1
    U[u_idx(0,0), col] -= 1
    V[v_idx(0,0), col] += 1
    V[v_idx(0,1), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[u_idx(0,1), col] += 1
    U[u_idx(1,1), col] -= 1
    V[v_idx(1,0), col] += 1
    V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] += 1
    col += 1
    
    # Block 2: A_L (2x2) * B_1b (2x3) -> Rank 2
    # A_L: i in {0,1}, j in {0,1}
    # B_1b: j in {0,1}, k in {2,3,4}
    # W: k in {2,3,4}, i in {0,1}
    
    # r=1: u=(1,0), v=(1,1), w=(1,1,1)
    U[u_idx(0,0), col] += 1
    U[u_idx(0,1), col] += 1 # Wait, u is vector for i. v is vector for j.
    # My previous derivation:
    # u^1 = (1,0) for i=0,1? No, u is length n=2.
    # u_i^r.
    # For r=1: u_0=1, u_1=0.
    # v_j^r. v_0=1, v_1=1.
    # w_k^r. w_2=1, w_3=1, w_4=1.
    
    # Correction:
    # Term 1:
    # u: index 0 (i=0) is 1, index 1 (i=1) is 0.
    # v: index 0 (j=0) is 1, index 1 (j=1) is 1.
    # w: index 2 (k=2) is 1, index 3 (k=3) is 1, index 4 (k=4) is 1.
    
    U[u_idx(0,0), col] += 1 # i=0, j=0. u_0=1, v_0=1 -> 1
    U[u_idx(0,1), col] += 1 # i=0, j=1. u_0=1, v_1=1 -> 1
    # i=1 terms are 0.
    
    V[v_idx(0,2), col] += 1 # j=0, k=2. v_0=1, w_2=1 -> 1
    V[v_idx(0,3), col] += 1
    V[v_idx(0,4), col] += 1
    V[v_idx(1,2), col] += 1 # j=1, k=2. v_1=1, w_2=1 -> 1
    V[v_idx(1,3), col] += 1
    V[v_idx(1,4), col] += 1
    
    W[w_idx(2,0), col] += 1 # k=2, i=0. w_2=1, u_0=1 -> 1
    W[w_idx(3,0), col] += 1
    W[w_idx(4,0), col] += 1
    # i=1 terms are 0.
    
    col += 1
    
    # Term 2:
    # u: index 0 is 0, index 1 is 1.
    # v: index 0 is 1, index 1 is 1.
    # w: index 2,3,4 are 1.
    
    U[u_idx(1,0), col] += 1 # i=1, j=0. u_1=1, v_0=1 -> 1
    U[u_idx(1,1), col] += 1 # i=1, j=1. u_1=1, v_1=1 -> 1
    
    V[v_idx(0,2), col] += 1
    V[v_idx(0,3), col] += 1
    V[v_idx(0,4), col] += 1
    V[v_idx(1,2), col] += 1
    V[v_idx(1,3), col] += 1
    V[v_idx(1,4), col] += 1
    
    W[w_idx(2,1), col] += 1 # k=2, i=1. w_2=1, u_1=1 -> 1
    W[w_idx(3,1), col] += 1
    W[w_idx(4,1), col] += 1
    
    col += 1
    
    # Block 3: A_R (2x2) * B_2a (2x2) -> Rank 7 (Strassen)
    # A_R: i in {0,1}, j in {2,3}
    # B_2a: j in {2,3}, k in {0,1}
    # W: k in {0,1}, i in {0,1}
    
    # Map local indices (i', j') in {0,1} to global
    # i' -> i, j' -> j'+2
    
    # M1
    U[u_idx(0,2), col] += 1
    U[u_idx(1,3), col] += 1
    V[v_idx(2,0), col] += 1
    V[v_idx(3,1), col] += 1
    W[w_idx(0,0), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M2
    U[u_idx(1,2), col] += 1
    U[u_idx(1,3), col] += 1
    V[v_idx(2,0), col] += 1
    W[w_idx(0,1), col] += 1
    W[w_idx(1,1), col] -= 1
    col += 1
    
    # M3
    U[u_idx(0,2), col] += 1
    V[v_idx(2,1), col] += 1
    V[v_idx(3,1), col] -= 1
    W[w_idx(1,0), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M4
    U[u_idx(1,3), col] += 1
    V[v_idx(3,0), col] += 1
    V[v_idx(2,0), col] -= 1
    W[w_idx(0,0), col] += 1
    W[w_idx(0,1), col] += 1
    col += 1
    
    # M5
    U[u_idx(0,2), col] += 1
    U[u_idx(0,3), col] += 1
    V[v_idx(3,1), col] += 1
    W[w_idx(0,0), col] -= 1
    W[w_idx(1,0), col] += 1
    col += 1
    
    # M6
    U[u_idx(1,2), col] += 1
    U[u_idx(0,2), col] -= 1
    V[v_idx(2,0), col] += 1
    V[v_idx(2,1), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M7
    U[u_idx(0,3), col] += 1
    U[u_idx(1,3), col] -= 1
    V[v_idx(3,0), col] += 1
    V[v_idx(3,1), col] += 1
    W[w_idx(0,0), col] += 1
    col += 1
    
    # Block 4: A_R (2x2) * B_2b (2x3) -> Rank 2
    # A_R: i in {0,1}, j in {2,3}
    # B_2b: j in {2,3}, k in {2,3,4}
    # W: k in {2,3,4}, i in {0,1}
    
    # Term 1: i=0
    U[u_idx(0,2), col] += 1
    U[u_idx(0,3), col] += 1
    V[v_idx(2,2), col] += 1
    V[v_idx(2,3), col] += 1
    V[v_idx(2,4), col] += 1
    V[v_idx(3,2), col] += 1
    V[v_idx(3,3), col] += 1
    V[v_idx(3,4), col] += 1
    W[w_idx(2,0), col] += 1
    W[w_idx(3,0), col] += 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # Term 2: i=1
    U[u_idx(1,2), col] += 1
    U[u_idx(1,3), col] += 1
    V[v_idx(2,2), col] += 1
    V[v_idx(2,3), col] += 1
    V[v_idx(2,4), col] += 1
    V[v_idx(3,2), col] += 1
    V[v_idx(3,3), col] += 1
    V[v_idx(3,4), col] += 1
    W[w_idx(2,1), col] += 1
    W[w_idx(3,1), col] += 1
    W[w_idx(4,1), col] += 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if not verify_decomposition(U, V, W, rank):
        # Fallback to trivial if verification fails (should not happen)
        rank_trivial = n * m * p
        U_trivial = np.zeros((dim_u, rank_trivial))
        V_trivial = np.zeros((dim_v, rank_trivial))
        W_trivial = np.zeros((dim_w, rank_trivial))
        col_t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_trivial[i*m+j, col_t] = 1.0
                    V_trivial[j*p+k, col_t] = 1.0
                    W_trivial[k*n+i, col_t] = 1.0
                    col_t += 1
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
