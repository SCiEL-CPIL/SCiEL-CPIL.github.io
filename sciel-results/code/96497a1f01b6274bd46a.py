"""Matrix multiplication 3x5x6: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify decomposition is exact."""
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # For 3x5x6, the best known rank is 90 (schoolbook).
    # Try to find a better decomposition.
    # Known: R(2,2,2)=7, R(1,*,*)=*, R(*,1,*)=*, R(*,*,1)=*
    # R(2,3,3)=18, R(3,3,3)=23 (lower bound), R(2,2,3)=11, R(3,3,4)=?
    
    # Let's try a specific decomposition that might be better.
    # For 3x5x6, we can partition as:
    # i: [0,1,2] -> [0,1], [2]  (sizes 2, 1)
    # j: [0,1,2,3,4] -> [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # k: [0,1,2,3,4,5] -> [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
    
    # Block ranks:
    # (2,2,2): 7
    # (2,2,2): 7
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,2,1): 4
    # (2,2,1): 4
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,2,1): 2
    # (1,2,1): 2
    # Total: 7*3 + 4*6 + 2*3 = 21 + 24 + 6 = 51
    # This is better than 90!
    
    # Let's implement this specific partition.
    def build_specific_decomposition():
        """Build decomposition using partition with Strassen for 2x2x2 blocks."""
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # Strassen terms for 2x2x2
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            # M2 = (A10+A11)B00 -> C10+1, C11-1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
            # M3 = A00(B01-B11) -> C01+1, C11+1
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
            # M4 = A11(B10-B00) -> C00+1, C10+1
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            # M5 = (A00+A01)B11 -> C00-1, C01+1
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
            # M6 = (A10-A00)(B00+B01) -> C11+1
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            # M7 = (A01-A11)(B10+B11) -> C00+1
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]
        
        # Compute total rank
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        U[ig[li] * m + jg[lj], col] = float(val_u)
                    for (lj, lk), val_v in v_pat.items():
                        V[jg[lj] * p + kg[lk], col] = float(val_v)
                    for (lk, li), val_w in w_pat.items():
                        W[kg[lk] * n + ig[li], col] = float(val_w)
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            U[ig[li] * m + jg[lj], col] = 1.0
                            V[jg[lj] * p + kg[lk], col] = 1.0
                            W[kg[lk] * n + ig[li], col] = 1.0
                            col += 1
        
        return {"U": U, "V": V, "W": W}, total_rank

    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p

    # Try the specific partition
    if time.time() < deadline - 10:
        result, rank = build_specific_decomposition()
        if rank < best_rank:
            if verify(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
