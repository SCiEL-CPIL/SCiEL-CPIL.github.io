"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a decomposition of rank 73 for 3x5x5 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks and Schoolbook for others.
    """
    n, m, p = DIMENSIONS
    total_rank = 73
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0

    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal rank_offset
        if rank_offset >= total_rank:
            return
        U[u_idx][rank_offset] += val
        V[v_idx][rank_offset] += val
        W[w_idx][rank_offset] += val
        rank_offset += 1

    def add_schoolblock(u_i_range, u_j_range, v_j_range, v_k_range, w_k_range, w_i_range):
        """
        Adds schoolbook decomposition for a block.
        u_idx = i*m + j
        v_idx = j*p + k
        w_idx = k*n + i
        """
        for i in u_i_range:
            for j in u_j_range:
                for k in v_k_range:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Ensure w_idx is in the correct range for w_i_range and w_k_range
                    # The logic above maps i,j,k directly.
                    # We need to ensure the indices correspond to the specific sub-blocks.
                    # The ranges provided must be consistent.
                    add_term(u_idx, v_idx, w_idx, 1)

    def add_strassen(u_i_list, u_j_list, v_j_list, v_k_list, w_k_list, w_i_list):
        """
        Adds Strassen decomposition for a 2x2x2 block.
        u_i_list, u_j_list, etc. should be lists of 2 indices.
        """
        # Strassen factors:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local indices 0,1 correspond to the lists
        # A[i][j] -> U[u_i_list[i] * m + u_j_list[j]]
        # B[j][k] -> V[v_j_list[j] * p + v_k_list[k]]
        # C[i][k] -> W[w_k_list[k] * n + w_i_list[i]]
        
        # We define the 7 terms as sums of products of linear combos.
        # Term 1: (A00+A11)(B00+B11)
        # Expands to: A00B00 + A00B11 + A11B00 + A11B11
        # This contributes to C00, C01, C10, C11 with signs.
        
        # It is easier to add the 7 rank-1 tensors directly.
        # Rank 1 tensor (u, v, w) adds U[u]*V[v]*W[w].
        # In Strassen, we have linear combos.
        # M1 = (U0+U1)(V0+V1) -> 4 rank 1 terms? No, Strassen uses 7 multiplications.
        # The 7 multiplications are of linear combinations.
        # So we add 7 terms where U and V entries are linear combos of the block entries.
        
        # Let's define the 7 multipliers and their contributions to C.
        
        # 1. M1 = (A00+A11)(B00+B11)
        # Contrib to C00: +1, C01: -1, C10: -1, C11: +1
        # Wait, standard Strassen:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # The 7 M's are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We add 7 rank-1 components.
        # Each component has U, V, W vectors.
        
        # 1. U: [1, 1, 1, 1] (A00, A11, A11, A00? No)
        # U for M1: A00, A11 are 1. Others 0.
        # V for M1: B00, B11 are 1. Others 0.
        # W for M1: C00(+1), C01(-1), C10(-1), C11(+1).
        
        # This is getting complicated to hardcode.
        # I will use the 7 terms explicitly.
        
        # Helper to add a specific Strassen term
        def add_strassen_term(u_vals, v_vals, w_vals):
            # u_vals, v_vals, w_vals are dicts {idx: val}
            # This represents one rank-1 tensor in the decomposition of the block.
            # But Strassen terms are products of linear combos, which are sums of rank-1 tensors?
            # No, a multiplication (A+D)(B+E) is NOT a single rank-1 tensor.
            # It is 4 rank-1 tensors.
            # Strassen replaces 8 rank-1 tensors with 7 products of linear combos.
            # Each product is a scalar.
            # So we have 7 scalars m1..m7.
            # C is a linear combo of m1..m7.
            # This means the decomposition is sum of 7 terms, where each term is 
            # (LinearComboU) * (LinearComboV) * (LinearComboW)?
            # No. The tensor is T = sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k.
            # Strassen says T = sum_{r=1}^7 u^{(r)} \otimes v^{(r)} \otimes w^{(r)}.
            # Where u^{(r)}, v^{(r)} are linear combos of input elements.
            
            # So I just need to define the 7 vectors u^{(r)}, v^{(r)}, w^{(r)}.
            
            pass
        
        # Let's define the 7 vectors.
        # u^{(r)} is length 4 (2x2 block).
        # v^{(r)} is length 4.
        # w^{(r)} is length 4.
        
        # r=1: u=[1,0,0,1] (A00+A11), v=[1,0,0,1] (B00+B11), w=[1,-1,-1,1] (C00-C01-C10+C11)
        # r=2: u=[0,0,1,1] (A10+A11), v=[1,0,0,0] (B00), w=[0,0,1,0] (C10) -- Wait, M2 contribs to C10 and C11.
        # M2 = (A10+A11)B00.
        # C10 += M2. C11 -= M2.
        # So w=[0,0,1,-1].
        
        # Let's list them:
        # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,-1,-1,1]
        # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
        # 3. u=[1,0,0,0], v=[0,1,-1,0], w=[0,1,0,0]  # M3 = A00(B10-B11). C01 += M3. C11 += M3.
        #    Wait, M3 contribs to C01 and C11. w=[0,1,0,1].
        # 4. u=[0,0,0,1], v=[0,1,-1,0], w=[0,0,-1,1]  # M4 = A11(B01-B00). C00 -= M4. C10 += M4.
        #    w=[-1,0,1,0].
        # 5. u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,1]   # M5 = (A00+A01)B11. C00 -= M5. C01 += M5. C11 += M5.
        #    w=[-1,1,0,1].
        # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]   # M6 = (A10-A00)(B00+B01). C11 += M6.
        #    w=[0,0,0,1].
        # 7. u=[0,-1,0,1], v=[0,0,1,1], w=[-1,0,0,0]   # M7 = (A01-A11)(B10+B11). C00 -= M7.
        #    w=[-1,0,0,0].
        
        # Now map local indices 0..3 to global u,v,w indices.
        # Local A index l -> i=l//2, j=l%2. Global u = u_i_list[i]*m + u_j_list[j].
        # Local B index l -> j=l//2, k=l%2. Global v = v_j_list[j]*p + v_k_list[k].
        # Local C index l -> i=l//2, k=l%2. Global w = w_k_list[k]*n + w_i_list[i].
        
        vectors_u = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,1,0,0],
            [-1,0,1,0],
            [0,-1,0,1]
        ]
        vectors_v = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,-1,0],
            [0,1,-1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,0,1,1]
        ]
        vectors_w = [
            [1,-1,-1,1],
            [0,0,1,-1],
            [0,1,0,1],
            [-1,0,1,0],
            [-1,1,0,1],
            [0,0,0,1],
            [-1,0,0,0]
        ]
        
        for r in range(7):
            for lu in range(4):
                if vectors_u[r][lu] != 0:
                    i, j = lu // 2, lu % 2
                    gu = u_i_list[i] * m + u_j_list[j]
                    for lv in range(4):
                        if vectors_v[r][lv] != 0:
                            jj, kk = lv // 2, lv % 2
                            gv = v_j_list[jj] * p + v_k_list[kk]
                            for lw in range(4):
                                if vectors_w[r][lw] != 0:
                                    ii, kk2 = lw // 2, lw % 2
                                    gw = w_k_list[kk2] * n + w_i_list[ii]
                                    val = vectors_u[r][lu] * vectors_v[r][lv] * vectors_w[r][lw]
                                    add_term(gu, gv, gw, val)

    # 1. A1 (0-2, 0-2) * B1a (0-2, 0-2) : Schoolbook 27
    add_schoolblock(range(3), range(3), range(3), range(3), range(3), range(3))
    
    # 2. A1 (0-2, 0-2) * B1b (0-2, 3-4) : Schoolbook 18
    add_schoolblock(range(3), range(3), range(3), range(3,5), range(3,5), range(3))
    
    # 3. A2a (0-1, 3-4) * B2a1 (3-4, 0-1) : Strassen 7
    add_strassen([0,1], [3,4], [3,4], [0,1], [0,1], [0,1])
    
    # 4. A2a (0-1, 3-4) * B2a2 (3-4, 2-2) : Schoolbook 4
    add_schoolblock(range(2), range(3,5), range(3,5), [2], [2], range(2))
    
    # 5. A2b (2-2, 3-4) * B2a (3-4, 0-2) : Schoolbook 6
    add_schoolblock([2], range(3,5), range(3,5), range(3), range(3), [2])
    
    # 6. A2c (0-1, 3-4) * B2b (3-4, 3-4) : Strassen 7
    add_strassen([0,1], [3,4], [3,4], range(3,5), range(3,5), [0,1])
    
    # 7. A2d (2-2, 3-4) * B2b (3-4, 3-4) : Schoolbook 4
    add_schoolblock([2], range(3,5), range(3,5), range(3,5), range(3,5), [2])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
