import time
import numpy as np

def construct():
    """
    Rank-77 exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: 
    - Outer: Strassen (rank 7) on 2x2 block structure of A (4x4) and B (4x6)
    - Inner: For each 2x2 * 2x3 block product:
        * Strassen (rank 7) for the 2x2 * 2x2 part (first 2 cols)
        * Schoolbook (rank 4) for the 2x2 * 2x1 part (3rd col)
    - Total rank: 7 * (7 + 4) = 77
    
    W convention: W[k*n + i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    Actually: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r]
    And W[k*n+i, r] = 1 for the output entry.
    """
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = 4, 4, 6
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    # Outer Strassen steps for 2x2 block structure
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11 (each 2x2)
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11 (each 2x3)
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11 (each 2x3)
    
    # Each step: (A_entries, B_entries, C_entries)
    # Each entry: (block_id, coeff)
    outer_steps = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        # M2 = (A10 + A11)B00 -> C10 - C11
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        # M3 = A00(B01 - B11) -> C01 + C11
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        # M4 = A11(B10 - B00) -> C00 + C10
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        # M5 = (A00 + A01)B11 -> C01 - C00
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        # M6 = (A10 - A00)(B00 + B01) -> C11
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        # M7 = (A01 - A11)(B10 + B11) -> C00
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    # Inner Strassen for 2x2 * 2x2 (first 2 columns of the 2x3 block)
    # Local A(r,s), B(s,k), C(r,k) with r,s,k in {0,1}
    # Each term: (A_entries, B_entries, C_entries)
    # Each entry: (row, col, coeff)
    inner_strassen = [
        # S1: (A00+A11)(B00+B11) -> C00+C11
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # S2: (A10+A11)B00 -> C10-C11
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # S3: A00(B01-B11) -> C01+C11
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # S4: A11(B10-B00) -> C00+C10
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # S5: (A00+A01)B11 -> C01-C00
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        # S6: (A10-A00)(B00+B01) -> C11
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # S7: (A01-A11)(B10+B11) -> C00
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    # Inner Schoolbook for 2x2 * 2x1 (3rd column, local k=2)
    # C(r,2) = A(r,0)*B(0,2) + A(r,1)*B(1,2)
    # Each term: (A_entries, B_entries, C_entries)
    inner_schoolbook = [
        # T1: A(0,0)*B(0,2) -> C(0,2)
        (([(0,0,1)], [(0,2,1)], [(0,2,1)])),
        # T2: A(0,1)*B(1,2) -> C(0,2)
        (([(0,1,1)], [(1,2,1)], [(0,2,1)])),
        # T3: A(1,0)*B(0,2) -> C(1,2)
        (([(1,0,1)], [(0,2,1)], [(1,2,1)])),
        # T4: A(1,1)*B(1,2) -> C(1,2)
        (([(1,1,1)], [(1,2,1)], [(1,2,1)]))
    ]
    
    # Global index mapping functions
    def a_global(block_id, r, s):
        """A block (block_id, local_row r, local_col s) -> global (I, J) -> U index I*m+J"""
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I * m + J
    
    def b_global(block_id, s, k):
        """B block (block_id, local_row s, local_col k) -> global (J, K) -> V index J*p+K"""
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J * p + K
    
    def c_global(block_id, r, k):
        """C block (block_id, local_row r, local_col k) -> global (I, K) -> W index K*n+I"""
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return K * n + I
    
    # Build the decomposition
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = i * 11
        
        # Inner Strassen (7 terms) for columns 0,1
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            col = base + j
            
            for (ar, as_, acoeff) in A_inner:
                for (ab_id, ab_coeff) in A_outer:
                    u_idx = a_global(ab_id, ar, as_)
                    U[u_idx][col] += acoeff * ab_coeff
            
            for (bs, bk, bcoeff) in B_inner:
                for (bb_id, bb_coeff) in B_outer:
                    v_idx = b_global(bb_id, bs, bk)
                    V[v_idx][col] += bcoeff * bb_coeff
            
            for (cr, ck, ccoeff) in C_inner:
                for (cb_id, cb_coeff) in C_outer:
                    w_idx = c_global(cb_id, cr, ck)
                    W[w_idx][col] += ccoeff * cb_coeff
        
        # Inner Schoolbook (4 terms) for column 2
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            col = base + 7 + j
            
            for (ar, as_, acoeff) in A_inner:
                for (ab_id, ab_coeff) in A_outer:
                    u_idx = a_global(ab_id, ar, as_)
                    U[u_idx][col] += acoeff * ab_coeff
            
            for (bs, bk, bcoeff) in B_inner:
                for (bb_id, bb_coeff) in B_outer:
                    v_idx = b_global(bb_id, bs, bk)
                    V[v_idx][col] += bcoeff * bb_coeff
            
            for (cr, ck, ccoeff) in C_inner:
                for (cb_id, cb_coeff) in C_outer:
                    w_idx = c_global(cb_id, cr, ck)
                    W[w_idx][col] += ccoeff * cb_coeff
        
        # Time guard
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Convert to proper integer format
    def clean(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                if isinstance(x, (int, np.integer)):
                    new_row.append(int(x))
                elif isinstance(x, (float, np.floating)):
                    val = int(x) if abs(x - round(x)) < 1e-10 else x
                    new_row.append(val)
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
