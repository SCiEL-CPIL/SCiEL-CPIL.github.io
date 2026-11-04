import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def strassen_block_solution():
        """Rank-36 solution: split middle dim 4=2+2, output dim 5=2+2+1.
        For each middle block q: Strassen(2x2x2) for cols 0-1, Strassen(2x2x2) for cols 2-3,
        schoolbook(2x2x1) for col 4. Total: 2*(7+7+4) = 36."""
        rank = 36
        num_u = n * m  # 8
        num_v = m * p  # 20
        num_w = p * n  # 10
        
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        
        col = 0
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Strassen 2x2x2 terms (local 2x2 indices 0-3)
        strassen_terms_2x2x2 = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        ]

        # Schoolbook 2x2x1 terms (A is 2x2, B is 2x1, C is 2x1)
        schoolbook_2x2x1 = [
            ({0:1}, {0:1}, {0:1}),
            ({1:1}, {1:1}, {0:1}),
            ({2:1}, {0:1}, {1:1}),
            ({3:1}, {1:1}, {1:1})
        ]

        # For each middle block q in {0, 1}
        for q in range(2):
            # 2x2x2 blocks for output col groups 0-1 and 2-3
            for s in range(2):
                for u_t, v_t, w_t in strassen_terms_2x2x2:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    # U: A[i, j] where i in {0,1}, j in {2q, 2q+1}
                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2*q + lj
                        u_global[gi * m + gj] = val
                    
                    # V: B[j, k] where j in {2q, 2q+1}, k in {2s, 2s+1}
                    for idx, val in v_t.items():
                        lj = idx // 2
                        lk = idx % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        v_global[gj * p + gk] = val
                    
                    # W: C[k, i] where k in {2s, 2s+1}, i in {0,1}
                    for idx, val in w_t.items():
                        lk = idx // 2
                        li = idx % 2
                        gk = 2*s + lk
                        gi = li
                        w_global[gk * n + gi] = val
                    
                    add_term(u_global, v_global, w_global)
            
            # 2x2x1 block for output col 4
            for u_t, v_t, w_t in schoolbook_2x2x1:
                u_global = {}
                v_global = {}
                w_global = {}
                
                # U: A[i, j] where i in {0,1}, j in {2q, 2q+1}
                for idx, val in u_t.items():
                    li = idx // 2
                    lj = idx % 2
                    gi = li
                    gj = 2*q + lj
                    u_global[gi * m + gj] = val
                
                # V: B[j, 4] where j in {2q, 2q+1}
                for idx, val in v_t.items():
                    lj = idx
                    gj = 2*q + lj
                    gk = 4
                    v_global[gj * p + gk] = val
                
                # W: C[4, i] where i in {0,1}
                for idx, val in w_t.items():
                    li = idx
                    gi = li
                    gk = 4
                    w_global[gk * n + gi] = val
                
                add_term(u_global, v_global, w_global)

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose matrix multiplication"""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                # Check valid entries equal 1
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
                # Check invalid entries equal 0
                for a_idx in range(n * m):
                    ai = a_idx // m
                    aj = a_idx % m
                    for b_idx in range(m * p):
                        bj = b_idx // p
                        bk = b_idx % p
                        if ai == i and bk == k and aj == bj:
                            continue
                        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                        if abs(val) > 1e-9:
                            return False
        return True

    # Generate the rank-36 solution
    result = strassen_block_solution()
    
    # Verify correctness
    if verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook rank 40
    n, m, p = DIMENSIONS
    rank = n * m * p  # 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
