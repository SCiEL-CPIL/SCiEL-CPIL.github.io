"""Matrix multiplication 4,4,7 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 4x4x4 sub-problem (rank 49) combined with
    schoolbook for the remaining 4x4x3 sub-problem (rank 48), total rank = 97.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    rank = 97  # 49 (Strassen 4x4x4) + 48 (schoolbook 4x4x3)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # B columns 0-3, C columns 0-3
    
    # 2x2x2 Strassen patterns
    # Each pattern: (u_terms, v_terms, w_terms)
    # u_terms: list of (local_idx, sign)
    # v_terms: list of (local_idx, sign)  
    # w_terms: list of (local_idx, sign)
    
    s22 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,1),(2,1),(3,1)]),   # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),                            # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,-1)]),                    # M3
        ([(3,1)], [(2,1),(0,-1)], [(2,1),(0,-1)]),                    # M4
        ([(0,1),(1,1)], [(3,1)], [(3,1)]),                            # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(2,1),(1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(1,1),(3,1)]),               # M7
    ]
    
    # 4x4 Strassen: for each Mi, which 2x2 blocks of A and B
    # a_blocks: list of (block_row, block_col, sign)
    # b_blocks: list of (block_row, block_col, sign)
    
    s44 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),   # M1: (A00+A11)(B00+B11)
        ([(1,0,1),(1,1,1)], [(0,0,1)]),            # M2: (A10+A11)B00
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),           # M3: A00(B01-B11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),           # M4: A11(B10-B00)
        ([(0,0,1),(0,1,1)], [(1,1,1)]),            # M5: (A00+A01)B11
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),   # M6: (A10-A00)(B00+B01)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),   # M7: (A01-A11)(B10+B11)
    ]
    
    # C block mapping for Strassen results:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Mi, the 2x2x2 Strassen produces a 2x2 result.
    # We need to add this to the appropriate C blocks.
    
    # C block offsets in W (for 4x4x4 part, W index = k*4+i):
    # C00: k=0,1; i=0,1 -> W indices 0,1,4,5
    # C01: k=0,1; i=2,3 -> W indices 2,3,6,7
    # C10: k=2,3; i=0,1 -> W indices 8,9,12,13
    # C11: k=2,3; i=2,3 -> W indices 10,11,14,15
    
    # For each Mi, we know which C blocks it contributes to and with what sign.
    
    c_block_contribs = {
        0: [(0,0,1), (0,1,1), (1,0,1), (1,1,1)],   # M1 -> C00(+), C01(?), ... 
        # Actually, let me be more careful.
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
    }
    
    # For each Mi, list of (c_block_row, c_block_col, sign):
    mi_c_blocks = [
        [(0,0,1), (1,1,1)],   # M1: C00(+), C11(+)
        [(1,0,1), (1,1,-1)],  # M2: C10(+), C11(-)
        [(0,1,1), (1,1,1)],   # M3: C01(+), C11(+)
        [(0,0,1), (1,0,1)],   # M4: C00(+), C10(+)
        [(0,0,-1), (0,1,1)],  # M5: C00(-), C01(+)
        [(1,1,1)],            # M6: C11(+)
        [(0,0,1), (0,1,-1)],  # M7: C00(+), C01(-)
    ]
    
    # Now build the 49 terms.
    for mi in range(7):
        a_blocks, b_blocks = s44[mi]
        for si in range(7):
            u_terms, v_terms, w_terms = s22[si]
            
            # For this sub-multiplication, build U and V contributions.
            u_contribs = []
            for (ar, ac, sa) in a_blocks:
                for (lu, su) in u_terms:
                    # Full U index for A[i,j] where i = ar*2 + lu//2, j = ac*2 + lu%2
                    full_u = (ar * 2 + lu // 2) * 4 + (ac * 2 + lu % 2)
                    sign = sa * su
                    u_contribs.append((full_u, sign))
            
            v_contribs = []
            for (br, bc, sb) in b_blocks:
                for (lv, sv) in v_terms:
                    # Full V index for B[j,k] where j = br*2 + lv//2, k = bc*2 + lv%2
                    # For 4x4x4 part, B has 4 columns, so V index = j*4 + k
                    full_v = (br * 2 + lv // 2) * 4 + (bc * 2 + lv % 2)
                    sign = sb * sv
                    v_contribs.append((full_v, sign))
            
            # For each combination of U and V contributions, add to W.
            for (fu, su_total) in u_contribs:
                for (fv, sv_total) in v_contribs:
                    combined_sign = su_total * sv_total
                    for (lw, sw) in w_terms:
                        # lw is local index in 2x2 C block: lw = local_k*2 + local_i
                        local_k = lw // 2
                        local_i = lw % 2
                        for (cbr, cbc, csign) in mi_c_blocks[mi]:
                            # Full W index: k = cbr*2 + local_k, i = cbc*2 + local_i
                            # W index = k*4 + i (for 4x4x4 part)
                            full_k = cbr * 2 + local_k
                            full_i = cbc * 2 + local_i
                            full_w = full_k * 4 + full_i
                            w_sign = combined_sign * sw * csign
                            W[full_w, col] += w_sign
            
            col += 1
    
    # Part 2: 4x4x3 using schoolbook (rank 48)
    # B columns 4-6, C columns 4-6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
    
    # Convert to lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
