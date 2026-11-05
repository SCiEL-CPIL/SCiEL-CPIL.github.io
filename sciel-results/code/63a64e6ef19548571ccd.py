import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices into blocks of size 2, 2, 1
    # Ranges: [0, 1], [2, 3], [4]
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Total rank calculation
    # J=0 (m=2): 46
    # J=1 (m=2): 46
    # J=2 (m=1): 25
    total_rank = 117
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    rank_idx = 0
    
    # Helper to add a term
    def add_term(i_local, j_local, k_local, u_val, v_val, w_val, i_off, j_off, k_off):
        nonlocal rank_idx
        i_global = i_off + i_local
        j_global = j_off + j_local
        k_global = k_off + k_local
        
        u_idx = i_global * m + j_global
        v_idx = j_global * p + k_global
        w_idx = k_global * n + i_global
        
        if u_val != 0:
            U[u_idx, rank_idx] = u_val
        if v_val != 0:
            V[v_idx, rank_idx] = v_val
        if w_val != 0:
            W[w_idx, rank_idx] = w_val
        rank_idx += 1

    # Pre-defined Strassen coefficients for 2x2x2
    # Each entry is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(j,k): val}
    # w_coeffs: dict {(k,i): val}
    
    strassen_terms = [
        # 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),
        # 3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # 4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # 5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # 6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Helper to add schoolbook terms for n x m x p block
    def add_schoolbook_block(i_off, j_off, k_off, n_sz, m_sz, p_sz):
        nonlocal rank_idx
        for i_l in range(n_sz):
            for j_l in range(m_sz):
                for k_l in range(p_sz):
                    # U: A[i_l, j_l]
                    # V: B[j_l, k_l]
                    # W: C[k_l, i_l]
                    add_term(i_l, j_l, k_l, 1, 1, 1, i_off, j_off, k_off)

    # Iterate over J blocks
    for J in range(3):
        j_off = offsets[J]
        m_sz = sizes[J]
        
        for I in range(3):
            i_off = offsets[I]
            n_sz = sizes[I]
            
            for K in range(3):
                k_off = offsets[K]
                p_sz = sizes[K]
                
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    # Use Strassen
                    for u_c, v_c, w_c in strassen_terms:
                        # We need to add these as a single term in the global rank
                        # But add_term expects single values.
                        # We need a different helper for multi-coefficient terms.
                        
                        # Let's construct the term manually
                        u_idx_map = {}
                        for (il, jl), val in u_c.items():
                            ig = i_off + il
                            jg = j_off + jl
                            idx = ig * m + jg
                            u_idx_map[idx] = val
                            
                        v_idx_map = {}
                        for (jl, kl), val in v_c.items():
                            jg = j_off + jl
                            kg = k_off + kl
                            idx = jg * p + kg
                            v_idx_map[idx] = val
                            
                        w_idx_map = {}
                        for (kl, il), val in w_c.items():
                            kg = k_off + kl
                            ig = i_off + il
                            idx = kg * n + ig
                            w_idx_map[idx] = val
                            
                        for idx, val in u_idx_map.items():
                            U[idx, rank_idx] = val
                        for idx, val in v_idx_map.items():
                            V[idx, rank_idx] = val
                        for idx, val in w_idx_map.items():
                            W[idx, rank_idx] = val
                        rank_idx += 1
                else:
                    # Use Schoolbook
                    add_schoolbook_block(i_off, j_off, k_off, n_sz, m_sz, p_sz)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
