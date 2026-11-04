"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Partition:
    # n=3 -> [1, 2]
    # m=4 -> [2, 2]
    # p=8 -> [4, 4]
    
    # Blocks:
    # 1. n=[0], m=[0,1], p=[0,1,2,3]: 1x2x4, rank 8
    # 2. n=[0], m=[2,3], p=[0,1,2,3]: 1x2x4, rank 8
    # 3. n=[0], m=[0,1], p=[4,5,6,7]: 1x2x4, rank 8
    # 4. n=[0], m=[2,3], p=[4,5,6,7]: 1x2x4, rank 8
    # 5. n=[1,2], m=[0,1], p=[0,1,2,3]: 2x2x4, rank 14 (2x Strassen 2x2x2)
    # 6. n=[1,2], m=[2,3], p=[0,1,2,3]: 2x2x4, rank 14 (2x Strassen 2x2x2)
    # 7. n=[1,2], m=[0,1], p=[4,5,6,7]: 2x2x4, rank 14 (2x Strassen 2x2x2)
    # 8. n=[1,2], m=[2,3], p=[4,5,6,7]: 2x2x4, rank 14 (2x Strassen 2x2x2)
    
    columns = []
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
        
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    
    def add_strassen(n_start, m_start, p_start):
        """Add Strassen 2x2x2 decomposition for a block at n_start, m_start, p_start."""
        # n_start, m_start, p_start are the top-left indices of the 2x2x2 block
        # A is 2x2, B is 2x2, C is 2x2
        
        # Local indices for A: (i, j) -> global U: (n_start+i)*m + (m_start+j)
        # Local indices for B: (j, k) -> global V: (m_start+j)*p + (p_start+k)
        # Local indices for C: (i, k) -> global W: (p_start+k)*n + (n_start+i)
        
        def u(i, j): return (n_start + i) * m + (m_start + j)
        def v(j, k): return (m_start + j) * p + (p_start + k)
        def w(i, k): return (p_start + k) * n + (n_start + i)
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        add_term({u(0,0): 1, u(1,1): 1}, {v(0,0): 1, v(1,1): 1}, {w(0,0): 1, w(1,1): 1}) # Wait, W coefficients depend on the combination.
        # This direct mapping is tricky because Strassen combines inputs and outputs.
        # I need to expand the terms.
        
        # Let's just add the 7 products and then add the linear combinations to W?
        # No, the interface requires U, V, W matrices such that C = sum U_i V_i W_i^T?
        # No, the tensor decomposition is $T = \sum_{r} U_r \otimes V_r \otimes W_r$.
        # So each term is a rank-1 tensor.
        # Strassen computes $C_{ij} = \sum_k M_k \cdot \alpha_{ijk}$.
        # This means $M_k$ is a product of linear forms in A and B.
        # $M_k = (\sum a_{ij} u_{k,ij}) (\sum b_{jk} v_{k,jk})$.
        # Then $C_{ij} = \sum_k M_k w_{k,ij}$.
        # So the rank-1 terms are $(u_{k,ij}, v_{k,jk}, w_{k,ij})$.
        
        # Strassen 7 products:
        # M1: A: {00:1, 11:1}, B: {00:1, 11:1}. W: {00:1, 11:1, 01:-1, 10:-1}? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each M_k, we add a term with U_k, V_k, and W_k where W_k has 1s where M_k contributes positively and -1s where negatively?
        # No, the term is $U_k \otimes V_k \otimes W_k$.
        # If $C_{ij} = \sum_k M_k w_{k,ij}$, then the $k$-th rank-1 term has $W$ vector with entry $w_{k,ij}$ at position $(i,j)$.
        
        # M1: U={00:1, 11:1}, V={00:1, 11:1}, W={00:1, 11:1} (from C00, C11)
        # M2: U={10:1, 11:1}, V={00:1}, W={10:1, 11:-1} (from C10, C11) -> C10 has +M2, C11 has -M2.
        # M3: U={00:1}, V={01:1, 11:-1}, W={01:1, 11:1} (from C01, C11) -> C01 has +M3, C11 has +M3.
        # M4: U={11:1}, V={10:1, 00:-1}, W={00:1, 10:1} (from C00, C10) -> C00 has +M4, C10 has +M4.
        # M5: U={00:1, 01:1}, V={11:1}, W={00:-1, 01:1} (from C00, C01) -> C00 has -M5, C01 has +M5.
        # M6: U={10:1, 00:-1}, V={00:1, 01:1}, W={11:1} (from C11)
        # M7: U={01:1, 11:-1}, V={10:1, 11:1}, W={00:1} (from C00)
        
        # Let's verify M1:
        # A00+A11, B00+B11.
        # C00 += M1, C11 += M1.
        # Term: U={00:1, 11:1}, V={00:1, 11:1}, W={00:1, 11:1}.
        
        # M2:
        # A10+A11, B00.
        # C10 += M2, C11 -= M2.
        # Term: U={10:1, 11:1}, V={00:1}, W={10:1, 11:-1}.
        
        # M3:
        # A00, B01-B11.
        # C01 += M3, C11 += M3.
        # Term: U={00:1}, V={01:1, 11:-1}, W={01:1, 11:1}.
        
        # M4:
        # A11, B10-B00.
        # C00 += M4, C10 += M4.
        # Term: U={11:1}, V={10:1, 00:-1}, W={00:1, 10:1}.
        
        # M5:
        # A00+A01, B11.
        # C00 -= M5, C01 += M5.
        # Term: U={00:1, 01:1}, V={11:1}, W={00:-1, 01:1}.
        
        # M6:
        # A10-A00, B00+B01.
        # C11 += M6.
        # Term: U={10:1, 00:-1}, V={00:1, 01:1}, W={11:1}.
        
        # M7:
        # A01-A11, B10+B11.
        # C00 += M7.
        # Term: U={01:1, 11:-1}, V={10:1, 11:1}, W={00:1}.
        
        terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # M3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # M4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}) # M7
        ]
        
        for u_dict, v_dict, w_dict in terms:
            # Map local indices to global
            u_g = {u(i, j): val for (i, j), val in u_dict.items()}
            v_g = {v(j, k): val for (j, k), val in v_dict.items()}
            w_g = {w(i, k): val for (i, k), val in w_dict.items()}
            add_term(u_g, v_g, w_g)

    # Add blocks
    # 1. n=0, m=0, p=0, 1x2x4
    add_schoolblock(0, 0, 0, 1, 2, 4)
    # 2. n=0, m=2, p=0, 1x2x4
    add_schoolblock(0, 2, 0, 1, 2, 4)
    # 3. n=0, m=0, p=4, 1x2x4
    add_schoolblock(0, 0, 4, 1, 2, 4)
    # 4. n=0, m=2, p=4, 1x2x4
    add_schoolblock(0, 2, 4, 1, 2, 4)
    
    # 5. n=1, m=0, p=0, 2x2x4 -> two 2x2x2
    add_strassen(1, 0, 0)
    add_strassen(1, 0, 2)
    # 6. n=1, m=2, p=0, 2x2x4
    add_strassen(1, 2, 0)
    add_strassen(1, 2, 2)
    # 7. n=1, m=0, p=4, 2x2x4
    add_strassen(1, 0, 4)
    add_strassen(1, 0, 6)
    # 8. n=1, m=2, p=4, 2x2x4
    add_strassen(1, 2, 4)
    add_strassen(1, 2, 6)
    
    rank = len(columns)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
