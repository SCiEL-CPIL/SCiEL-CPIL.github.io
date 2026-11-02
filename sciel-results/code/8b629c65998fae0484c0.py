import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    # A: n x m, B: m x p, C: p x n (based on index convention c=k*n+i)
    
    # We will build the decomposition manually.
    # Rank 74 strategy.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        """
        Add a rank-1 term.
        u_idx: list of (index, value) pairs for U
        v_idx: list of (index, value) pairs for V
        w_idx: list of (index, value) pairs for W
        """
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        
        for idx, val in u_idx:
            U[idx] = val
        for idx, val in v_idx:
            V[idx] = val
        for idx, val in w_idx:
            W[idx] = val
            
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    # Helper to get global indices
    def idx_a(i, j): return i * m + j
    def idx_b(j, k): return j * p + k
    def idx_c(k, i): return k * n + i

    # --- Strassen Block for i in {0,1}, j in {0,1}, k in {0,1} ---
    
    # 1. M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_term(
        [(idx_a(0,0), 1.0), (idx_a(1,1), 1.0)],
        [(idx_b(0,0), 1.0), (idx_b(1,1), 1.0)],
        [(idx_c(0,0), 1.0), (idx_c(1,1), 1.0)]
    )
    
    # 2. M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_term(
        [(idx_a(1,0), 1.0), (idx_a(1,1), 1.0)],
        [(idx_b(0,0), 1.0)],
        [(idx_c(1,0), 1.0), (idx_c(1,1), -1.0)]
    )
    
    # 3. M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_term(
        [(idx_a(0,0), 1.0)],
        [(idx_b(0,1), 1.0), (idx_b(1,1), -1.0)],
        [(idx_c(0,1), 1.0), (idx_c(1,1), 1.0)]
    )
    
    # 4. M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_term(
        [(idx_a(1,1), 1.0)],
        [(idx_b(1,0), 1.0), (idx_b(0,0), -1.0)],
        [(idx_c(0,0), 1.0), (idx_c(1,0), 1.0)]
    )
    
    # 5. M5 = (A00+A10)B11 -> C00 -= M5, C01 += M5
    add_term(
        [(idx_a(0,0), 1.0), (idx_a(1,0), 1.0)],
        [(idx_b(1,1), 1.0)],
        [(idx_c(0,0), -1.0), (idx_c(0,1), 1.0)]
    )
    
    # 6. M6 = (A01-A00)(B00+B10) -> C11 += M6
    add_term(
        [(idx_a(0,1), 1.0), (idx_a(0,0), -1.0)],
        [(idx_b(0,0), 1.0), (idx_b(1,0), 1.0)],
        [(idx_c(1,1), 1.0)]
    )
    
    # 7. M7 = (A10-A11)(B01+B11) -> C00 += M7
    add_term(
        [(idx_a(1,0), 1.0), (idx_a(1,1), -1.0)],
        [(idx_b(0,1), 1.0), (idx_b(1,1), 1.0)],
        [(idx_c(0,0), 1.0)]
    )
    
    # --- Schoolbook Blocks ---
    
    # Sub-problem 2: i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 3: i in {2}, j in {0,1}, k in {0,1}
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 4: i in {2}, j in {2,3,4}, k in {0,1}
    for i in [2]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 5: i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 6: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 7: i in {2}, j in {0,1}, k in {2,3,4}
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Sub-problem 8: i in {2}, j in {2,3,4}, k in {2,3,4}
    for i in [2]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                add_term(
                    [(idx_a(i, j), 1.0)],
                    [(idx_b(j, k), 1.0)],
                    [(idx_c(k, i), 1.0)]
                )
                
    # Convert lists to arrays
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    # Verify time
    if time.time() > deadline:
        # Fallback to schoolbook if we somehow ran out of time (unlikely for static construction)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
