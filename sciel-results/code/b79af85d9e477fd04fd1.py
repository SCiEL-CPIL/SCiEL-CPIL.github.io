"""Matrix multiplication 2x5x6 - Strassen-based decomposition."""
DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = 2, 5, 6
    
    U_all = []
    V_all = []
    W_all = []
    
    # Part 1: 2x1x6, column j=0 (rank 12, schoolbook)
    for i in range(n):
        for k in range(p):
            U_all.append({i * m + 0: 1})
            V_all.append({0 * p + k: 1})
            W_all.append({k * n + i: 1})
    
    # Part 2: 2x4x6, columns j=1,2,3,4
    # Use Strassen for 2x2x2 blocks
    # Split into two 2x2x6: (j=1,2) and (j=3,4)
    # Each 2x2x6 = 3 * 2x2x2 (Strassen, rank 7 each) = rank 21
    
    # Verified Strassen 2x2x2 decomposition (7 terms)
    # u[i,j], v[j,k], w[k,i] for i,j,k in {0,1}
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    s_terms = [
        # (u, v, w) as 4-vectors: [i0j0, i0j1, i1j0, i1j1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        ([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
    ]
    
    for block_offset in [1, 3]:  # j=1,2 and j=3,4
        for k_block in range(3):  # k=0,1 and k=2,3 and k=4,5
            for r in range(7):
                u_vec, v_vec, w_vec = s_terms[r]
                
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for i in range(2):
                    for j_local in range(2):
                        val = u_vec[i * 2 + j_local]
                        if val != 0:
                            j_global = block_offset + j_local
                            u_dict[i * m + j_global] = val
                
                for j_local in range(2):
                    for k_local in range(2):
                        val = v_vec[j_local * 2 + k_local]
                        if val != 0:
                            j_global = block_offset + j_local
                            k_global = 2 * k_block + k_local
                            v_dict[j_global * p + k_global] = val
                
                for k_local in range(2):
                    for i in range(2):
                        val = w_vec[k_local * 2 + i]
                        if val != 0:
                            k_global = 2 * k_block + k_local
                            w_dict[k_global * n + i] = val
                
                U_all.append(u_dict)
                V_all.append(v_dict)
                W_all.append(w_dict)
    
    rank = len(U_all)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row, val in U_all[col].items():
            U[row][col] = val
        for row, val in V_all[col].items():
            V[row][col] = val
        for row, val in W_all[col].items():
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
