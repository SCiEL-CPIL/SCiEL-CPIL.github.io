"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # Target rank 44
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    rank_idx = 0
    
    def add_strassen(a_indices, b_indices, c_indices):
        """
        Adds Strassen decomposition for 2x2x2 block.
        a_indices: list of 4 U indices (row, col in A block)
        b_indices: list of 4 V indices
        c_indices: list of 4 W indices
        Returns next rank_idx
        """
        nonlocal rank_idx
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # Helper to add term
        def add_term(u_idx, v_idx, w_idx, coeff):
            nonlocal rank_idx
            U[u_idx][rank_idx] += coeff
            V[v_idx][rank_idx] += coeff
            W[w_idx][rank_idx] += coeff
            rank_idx += 1

        # We need to expand the linear combinations into rank-1 terms.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # This is NOT rank 1 in the tensor sense if we keep them as M.
        # Wait, Strassen terms ARE rank 1 in the tensor sense?
        # No, M1 = (a00+a11)(b00+b11).
        # This is sum of 4 rank 1 terms?
        # (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
        # That would be rank 4.
        # Strassen reduces the NUMBER OF MULTIPLICATIONS.
        # In tensor decomposition, a "multiplication" corresponds to a rank-1 term.
        # So M1 contributes 1 rank-1 term?
        # No, the tensor entry is T[i,j,k].
        # For M1, the contribution to C is distributed.
        # The term is $u(A) v(B) w(C)$.
        # $u(A) = a00+a11$.
        # $v(B) = b00+b11$.
        # $w(C)$ is the vector that adds to C00, C11, etc.
        # So yes, each Strassen multiplication M_k corresponds to ONE rank-1 term in the decomposition.
        
        # M1
        add_term(a00, b00, c00, 1) # Wait, u is linear combo.
        # My add_term assumes u_idx is a single index.
        # I need to handle linear combinations in U and V.
        
        # Let's refactor add_term to take lists of (idx, coeff)
        pass

    # Let's rewrite the helper to handle linear combos properly.
    def add_rank1_term(u_terms, v_terms, w_terms):
        """
        u_terms: list of (u_idx, coeff)
        v_terms: list of (v_idx, coeff)
        w_terms: list of (w_idx, coeff)
        Adds the tensor term sum_u sum_v sum_w u_coeff v_coeff w_coeff * E_u E_v E_w
        This is a single rank-1 term in the decomposition if we view the linear combos as the factors.
        """
        nonlocal rank_idx
        for u_idx, u_c in u_terms:
            for v_idx, v_c in v_terms:
                for w_idx, w_c in w_terms:
                    val = u_c * v_c * w_c
                    if val != 0:
                        U[u_idx][rank_idx] += val
                        V[v_idx][rank_idx] += val
                        W[w_idx][rank_idx] += val
        rank_idx += 1

    # Define indices for blocks
    # A indices: i*4 + j
    # B indices: j*4 + k
    # C indices: k*3 + i
    
    # Block 1: A_top_L (0,0),(0,1),(1,0),(1,1) x B_11 (0,0),(0,1),(1,0),(1,1) -> C_top_L (0,0),(0,1),(1,0),(1,1)
    a_tl = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # a00, a01, a10, a11
    b_11 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # b00, b01, b10, b11
    c_tl = [0*3+0, 1*3+0, 0*3+1, 1*3+1] # c00, c10, c01, c11 (k=0, i=0; k=0, i=1; k=1, i=0; k=1, i=1)
    # W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[1,0] -> k=0, i=1 -> 1
    # C[0,1] -> k=1, i=0 -> 3
    # C[1,1] -> k=1, i=1 -> 4
    # My c_tl list above was wrong.
    c_tl = [0, 1, 3, 4]
    
    # Strassen M1-M7 for Block 1
    # M1
    add_rank1_term([(a_tl[0], 1), (a_tl[3], 1)], [(b_11[0], 1), (b_11[3], 1)], [(c_tl[0], 1), (c_tl[3], 1)])
    # M2
    add_rank1_term([(a_tl[2], 1), (a_tl[3], 1)], [(b_11[0], 1)], [(c_tl[1], 1), (c_tl[3], -1)]) # C10 += M2, C11 -= M2
    # Wait, C10 is c_tl[1], C11 is c_tl[3].
    # M2 contributes to C10 (+) and C11 (-).
    # So w_terms should be [(c_tl[1], 1), (c_tl[3], -1)]
    
    # M3
    add_rank1_term([(a_tl[0], 1)], [(b_11[1], 1), (b_11[3], -1)], [(c_tl[2], 1), (c_tl[3], 1)]) # C01 += M3, C11 += M3
    # C01 is c_tl[2], C11 is c_tl[3].
    
    # M4
    add_rank1_term([(a_tl[3], 1)], [(b_11[2], 1), (b_11[0], -1)], [(c_tl[0], 1), (c_tl[1], 1)]) # C00 += M4, C10 += M4
    # C00 is c_tl[0], C10 is c_tl[1].
    
    # M5
    add_rank1_term([(a_tl[0], 1), (a_tl[1], 1)], [(b_11[3], 1)], [(c_tl[0], -1), (c_tl[2], 1)]) # C00 -= M5, C01 += M5
    # C00 is c_tl[0], C01 is c_tl[2].
    
    # M6
    add_rank1_term([(a_tl[2], 1), (a_tl[0], -1)], [(b_11[0], 1), (b_11[1], 1)], [(c_tl[3], 1)]) # C11 += M6
    
    # M7
    add_rank1_term([(a_tl[1], 1), (a_tl[3], -1)], [(b_11[2], 1), (b_11[3], 1)], [(c_tl[0], 1)]) # C00 += M7
    
    # Block 2: A_top_R x B_21 -> C_top_L
    a_tr = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    b_21 = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
    # C is same c_tl
    
    # M1
    add_rank1_term([(a_tr[0], 1), (a_tr[3], 1)], [(b_21[0], 1), (b_21[3], 1)], [(c_tl[0], 1), (c_tl[3], 1)])
    # M2
    add_rank1_term([(a_tr[2], 1), (a_tr[3], 1)], [(b_21[0], 1)], [(c_tl[1], 1), (c_tl[3], -1)])
    # M3
    add_rank1_term([(a_tr[0], 1)], [(b_21[1], 1), (b_21[3], -1)], [(c_tl[2], 1), (c_tl[3], 1)])
    # M4
    add_rank1_term([(a_tr[3], 1)], [(b_21[2], 1), (b_21[0], -1)], [(c_tl[0], 1), (c_tl[1], 1)])
    # M5
    add_rank1_term([(a_tr[0], 1), (a_tr[1], 1)], [(b_21[3], 1)], [(c_tl[0], -1), (c_tl[2], 1)])
    # M6
    add_rank1_term([(a_tr[2], 1), (a_tr[0], -1)], [(b_21[0], 1), (b_21[1], 1)], [(c_tl[3], 1)])
    # M7
    add_rank1_term([(a_tr[1], 1), (a_tr[3], -1)], [(b_21[2], 1), (b_21[3], 1)], [(c_tl[0], 1)])

    # Block 3: A_top_L x B_12 -> C_top_R
    b_12 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    c_tr = [2*3+0, 3*3+0, 2*3+1, 3*3+1] # k=2, i=0; k=2, i=1; k=3, i=0; k=3, i=1
    # C[0,2] -> k=2, i=0 -> 6
    # C[1,2] -> k=2, i=1 -> 7
    # C[0,3] -> k=3, i=0 -> 9
    # C[1,3] -> k=3, i=1 -> 10
    c_tr = [6, 7, 9, 10]
    
    # M1
    add_rank1_term([(a_tl[0], 1), (a_tl[3], 1)], [(b_12[0], 1), (b_12[3], 1)], [(c_tr[0], 1), (c_tr[3], 1)])
    # M2
    add_rank1_term([(a_tl[2], 1), (a_tl[3], 1)], [(b_12[0], 1)], [(c_tr[1], 1), (c_tr[3], -1)])
    # M3
    add_rank1_term([(a_tl[0], 1)], [(b_12[1], 1), (b_12[3], -1)], [(c_tr[2], 1), (c_tr[3], 1)])
    # M4
    add_rank1_term([(a_tl[3], 1)], [(b_12[2], 1), (b_12[0], -1)], [(c_tr[0], 1), (c_tr[1], 1)])
    # M5
    add_rank1_term([(a_tl[0], 1), (a_tl[1], 1)], [(b_12[3], 1)], [(c_tr[0], -1), (c_tr[2], 1)])
    # M6
    add_rank1_term([(a_tl[2], 1), (a_tl[0], -1)], [(b_12[0], 1), (b_12[1], 1)], [(c_tr[3], 1)])
    # M7
    add_rank1_term([(a_tl[1], 1), (a_tl[3], -1)], [(b_12[2], 1), (b_12[3], 1)], [(c_tr[0], 1)])

    # Block 4: A_top_R x B_22 -> C_top_R
    b_22 = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
    
    # M1
    add_rank1_term([(a_tr[0], 1), (a_tr[3], 1)], [(b_22[0], 1), (b_22[3], 1)], [(c_tr[0], 1), (c_tr[3], 1)])
    # M2
    add_rank1_term([(a_tr[2], 1), (a_tr[3], 1)], [(b_22[0], 1)], [(c_tr[1], 1), (c_tr[3], -1)])
    # M3
    add_rank1_term([(a_tr[0], 1)], [(b_22[1], 1), (b_22[3], -1)], [(c_tr[2], 1), (c_tr[3], 1)])
    # M4
    add_rank1_term([(a_tr[3], 1)], [(b_22[2], 1), (b_22[0], -1)], [(c_tr[0], 1), (c_tr[1], 1)])
    # M5
    add_rank1_term([(a_tr[0], 1), (a_tr[1], 1)], [(b_22[3], 1)], [(c_tr[0], -1), (c_tr[2], 1)])
    # M6
    add_rank1_term([(a_tr[2], 1), (a_tr[0], -1)], [(b_22[0], 1), (b_22[1], 1)], [(c_tr[3], 1)])
    # M7
    add_rank1_term([(a_tr[1], 1), (a_tr[3], -1)], [(b_22[2], 1), (b_22[3], 1)], [(c_tr[0], 1)])

    # Block 5: c_L x B_11 -> C_bot_L (Schoolbook)
    c_l = [2*4+0, 2*4+1]
    c_bl = [2*3+0, 2*3+1] # k=0, i=2 -> 2; k=1, i=2 -> 5
    c_bl = [2, 5]
    
    # Term (j,k)
    add_rank1_term([(c_l[0], 1)], [(b_11[0], 1)], [(c_bl[0], 1)])
    add_rank1_term([(c_l[0], 1)], [(b_11[1], 1)], [(c_bl[1], 1)])
    add_rank1_term([(c_l[1], 1)], [(b_11[2], 1)], [(c_bl[0], 1)])
    add_rank1_term([(c_l[1], 1)], [(b_11[3], 1)], [(c_bl[1], 1)])

    # Block 6: c_R x B_21 -> C_bot_L
    c_r = [2*4+2, 2*4+3]
    
    add_rank1_term([(c_r[0], 1)], [(b_21[0], 1)], [(c_bl[0], 1)])
    add_rank1_term([(c_r[0], 1)], [(b_21[1], 1)], [(c_bl[1], 1)])
    add_rank1_term([(c_r[1], 1)], [(b_21[2], 1)], [(c_bl[0], 1)])
    add_rank1_term([(c_r[1], 1)], [(b_21[3], 1)], [(c_bl[1], 1)])

    # Block 7: c_L x B_12 -> C_bot_R
    c_br = [2*3+2, 2*3+3] # k=2, i=2 -> 8; k=3, i=2 -> 11
    c_br = [8, 11]
    
    add_rank1_term([(c_l[0], 1)], [(b_12[0], 1)], [(c_br[0], 1)])
    add_rank1_term([(c_l[0], 1)], [(b_12[1], 1)], [(c_br[1], 1)])
    add_rank1_term([(c_l[1], 1)], [(b_12[2], 1)], [(c_br[0], 1)])
    add_rank1_term([(c_l[1], 1)], [(b_12[3], 1)], [(c_br[1], 1)])

    # Block 8: c_R x B_22 -> C_bot_R
    
    add_rank1_term([(c_r[0], 1)], [(b_22[0], 1)], [(c_br[0], 1)])
    add_rank1_term([(c_r[0], 1)], [(b_22[1], 1)], [(c_br[1], 1)])
    add_rank1_term([(c_r[1], 1)], [(b_22[2], 1)], [(c_br[0], 1)])
    add_rank1_term([(c_r[1], 1)], [(b_22[3], 1)], [(c_br[1], 1)])

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
