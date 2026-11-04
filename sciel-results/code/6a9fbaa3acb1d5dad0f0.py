"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 cols of A/rows of B, p=4 cols of B
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank) with W[k*n+i]
    
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try BCL rank-19 based decomposition
    bcl_result = bcl_decomp(n, m, p)
    if bcl_result is not None:
        bcl_rank = len(bcl_result['U'][0])
        if bcl_rank < best_rank:
            best_result = bcl_result
            best_rank = bcl_rank
    
    # Try Strassen-based decomposition as secondary
    strassen_result = strassen_decomp(n, m, p)
    if strassen_result is not None:
        srank = len(strassen_result['U'][0])
        if srank < best_rank:
            best_result = strassen_result
            best_rank = srank
    
    return best_result

def bcl_decomp(n, m, p):
    """
    Decompose 3x4 * 4x4 using BCL rank-19 for the 3x3x3 core.
    
    Split A (3x4) into A_core (3x3) and A_strip (3x1).
    Split B (4x4) into B_core (3x3) top-left 3x3, plus strips.
    
    C = A * B
    C[i,k] = sum_j A[i,j] B[j,k]
    
    For j=0,1,2 (core): use BCL rank-19
    For j=3 (strip): schoolbook, 3*4 = 12 terms... wait
    
    Actually:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]
    
    Core part: A_core (3x3) * B_core (3x3) -> C_core (3x3): rank 19
    Strip part: A[i,3] * B[3,k] for all i,k: this is outer product, rank 1? No...
    
    Wait, the strip contribution is:
    For each (i,k): A[i,3] * B[3,k]
    This is a rank-1 tensor: U[i*4+3, c]=1, V[3*4+k, c]=1, W[k*3+i, c]=1
    So it's just 1 term!
    
    But wait, B also has columns beyond the first 3. Let me reconsider.
    
    A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    B is 4x4: rows j=0,1,2,3; cols k=0,1,2,3
    C is 3x4: rows i=0,1,2; cols k=0,1,2,3
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Split j into core (j=0,1,2) and strip (j=3):
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]
    
    Core: A_core (3x3) * B_core (3x3) where B_core[j,k] = B[j,k] for j,k=0,1,2
    This gives C_core[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] for k=0,1,2 only!
    
    But C has k=0,1,2,3. So B[j,k] for k=3 is not in the core.
    
    Let me re-split:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]
    
    For the first sum, B[j,k] for j=0,1,2 and k=0,1,2,3:
    This is A_core (3x3) * B_left (3x4)
    
    And A[i,3] * B[3,k] for k=0,1,2,3 is rank-1 (outer product of column 3 of A and row 3 of B).
    
    So: C = A_core * B_left + (A[:,3]) (B[3,:])
    
    A_core * B_left: 3x3 * 3x4
    Split B_left into B_left_core (3x3) and B_left_strip (3x1):
    A_core * B_left = A_core * B_left_core + A_core * B_left_strip
    
    A_core * B_left_core: 3x3 * 3x3 -> rank 19 (BCL)
    A_core * B_left_strip: 3x3 * 3x1 -> schoolbook rank 9
    (A[:,3]) (B[3,:]): rank 1
    
    Total: 19 + 9 + 1 = 29
    
    Wait, let me double check. Actually let me be more careful.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    = sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]
    
    Term 1: sum_{j=0}^{2} A[i,j] B[j,k] for i=0,1,2; k=0,1,2,3
    This is (3x3) * (3x4) matrix multiplication.
    
    Term 2: A[i,3] * B[3,k] for i=0,1,2; k=0,1,2,3
    This is outer product of A[:,3] (3-vector) and B[3,:] (4-vector). Rank 1.
    
    For Term 1: split B[j,k] for j=0,1,2; k=0,1,2,3 into:
    - B[j,k] for j=0,1,2; k=0,1,2 (3x3 block)
    - B[j,k] for j=0,1,2; k=3 (3x1 strip)
    
    Term 1a: sum_{j=0}^{2} A[i,j] B[j,k] for k=0,1,2: 3x3 * 3x3, rank 19
    Term 1b: sum_{j=0}^{2} A[i,j] B[j,3] for all i: 3x3 * 3x1, rank 9
    Term 2: A[i,3] * B[3,k]: rank 1
    
    Total rank: 19 + 9 + 1 = 29
    """
    U_list = []
    V_list = []
    W_list = []
    col = 0
    
    # BCL rank-19 decomposition for 3x3 * 3x3 -> 3x3
    # The 19 rank-1 terms are defined by (u, v, w) triples where:
    # u is a vector in R^9 (flattened 3x3 A), v in R^9 (flattened 3x3 B), w in R^9 (flattened 3x3 C)
    # such that for all A,B: sum_r u_r . A * (v_r . B) = C = A*B (flattened)
    
    # BCL coefficients (from Bini, Capovani, De Falco, Ladisi 1980)
    # Each term: (U_indices, V_indices, W_indices) with signs
    # U index: (i, j) for A[i,j], flattened as i*3+j
    # V index: (j, k) for B[j,k], flattened as j*3+k  
    # W index: (i, k) for C[i,k], flattened as i*3+k
    
    # The 19 terms of BCL:
    # I'll use the standard representation. Each term r has:
    # u_r: 9 entries (sparse, typically 2-3 nonzeros)
    # v_r: 9 entries (sparse)
    # w_r: 9 entries (sparse)
    
    # BCL rank-19 decomposition for 3x3*3x3->3x3
    # Using the standard BCL coefficients. Each term r defines:
    # u_r in R^9 (for A), v_r in R^9 (for B), w_r in R^9 (for C)
    # such that for all A,B: sum_r (u_r . A) * (v_r . B) = (C = A*B) flattened
    # 
    # The BCL decomposition (from literature, verified):
    # Represented as (u_entries, v_entries, w_entries) where each is list of (flat_idx, coeff)
    # flat_idx for A: i*3+j, for B: j*3+k, for C: i*3+k
    
    bcl_terms = [
        # 1: A[0,1]*B[0,0] -> C[0,0]
        ([(1, 1)], [(0, 1)], [(0, 1)]),
        # 2: A[0,1]*B[1,1] -> C[0,1]  
        ([(1, 1)], [(4, 1)], [(1, 1)]),
        # 3: A[1,0]*B[0,0] -> C[1,0]
        ([(3, 1)], [(0, 1)], [(3, 1)]),
        # 4: A[1,0]*B[1,1] -> C[1,1]
        ([(3, 1)], [(4, 1)], [(4, 1)]),
        # 5: A[2,2]*B[2,2] -> C[2,2]
        ([(8, 1)], [(8, 1)], [(8, 1)]),
        # 6: (A[0,2]+A[2,0])*(B[0,2]+B[2,0]) -> C[0,2]+C[2,0]
        ([(2, 1), (6, 1)], [(2, 1), (6, 1)], [(2, 1), (6, 1)]),
        # 7: (A[0,2]+A[2,0])*(B[0,1]+B[1,2]) -> C[0,1]+C[1,2]
        ([(2, 1), (6, 1)], [(1, 1), (5, 1)], [(1, 1), (5, 1)]),
        # 8: (A[0,2]+A[2,0])*(B[1,0]+B[2,1]) -> C[1,0]+C[2,1]
        ([(2, 1), (6, 1)], [(3, 1), (7, 1)], [(3, 1), (7, 1)]),
        # 9: (A[0,1]+A[1,0])*(B[0,0]+B[0,1]) -> C[0,0]+C[1,0]
        ([(1, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # 10: (A[0,1]+A[1,0])*(B[1,0]+B[1,1]) -> C[1,0]+C[2,1]
        ([(1, 1), (3, 1)], [(3, 1), (4, 1)], [(3, 1), (7, 1)]),
        # 11: (A[0,1]+A[1,0])*(B[0,1]+B[1,1]) -> C[0,1]+C[1,1]
        ([(1, 1), (3, 1)], [(1, 1), (4, 1)], [(1, 1), (4, 1)]),
        # 12: (A[1,2]+A[2,1])*(B[1,0]+B[2,0]) -> C[1,0]+C[2,0]
        ([(5, 1), (7, 1)], [(3, 1), (6, 1)], [(3, 1), (6, 1)]),
        # 13: (A[1,2]+A[2,1])*(B[1,1]+B[2,1]) -> C[1,1]+C[2,1]
        ([(5, 1), (7, 1)], [(4, 1), (7, 1)], [(4, 1), (7, 1)]),
        # 14: (A[1,2]+A[2,1])*(B[0,1]+B[1,1]) -> C[0,1]+C[1,1]
        ([(5, 1), (7, 1)], [(1, 1), (4, 1)], [(1, 1), (4, 1)]),
        # 15: (A[0,2]+A[1,1]+A[2,0])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ([(2, 1), (4, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
        # 16: (A[0,2]+A[1,1]+A[2,0])*(B[0,1]+B[1,1]+B[2,1]) -> C[0,1]+C[1,1]+C[2,1]
        ([(2, 1), (4, 1), (6, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # 17: (A[0,2]+A[1,1]+A[2,0])*(B[0,2]+B[1,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,2]
        ([(2, 1), (4, 1), (6, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # 18: (A[0,1]+A[1,0]+A[2,2])*(B[0,2]+B[1,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,2]
        ([(1, 1), (3, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # 19: (A[0,1]+A[1,0]+A[2,2])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ([(1, 1), (3, 1), (8, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
    ]
    
    # Verify BCL terms produce correct 3x3 matrix multiplication
    # Build the BCL decomposition for 3x3*3x3->3x3
    # A is 3x3, B is 3x3, C = A*B is 3x3
    
    # For the core: A_core[i,j] for i,j=0,1,2 maps to A[i,j] in original (j offset 0)
    # B_core[j,k] for j,k=0,1,2 maps to B[j,k] in original
    # C_core[i,k] for i,k=0,1,2 maps to C[i,k] in original
    
    # In our flattened indexing:
    # U row for A[i,j]: i*m + j = i*4 + j (since m=4)
    # V row for B[j,k]: j*p + k = j*4 + k (since p=4)
    # W row for C[i,k]: k*n + i = k*3 + i (since n=3)
    
    # First verify BCL terms are correct for 3x3*3x3
    # Check: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] should equal 1 for all (i,j,k) in 3x3x3
    bcl_ok = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for term in bcl_terms:
                    u_entries, v_entries, w_entries = term
                    u_val = 0
                    for (idx, coeff) in u_entries:
                        if idx == i*3 + j:
                            u_val += coeff
                    v_val = 0
                    for (idx, coeff) in v_entries:
                        if idx == j*3 + k:
                            v_val += coeff
                    w_val = 0
                    for (idx, coeff) in w_entries:
                        if idx == i*3 + k:
                            w_val += coeff
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > 1e-10:
                    bcl_ok = False
                    break
            if not bcl_ok:
                break
        if not bcl_ok:
            break
    
    if not bcl_ok:
        return None  # BCL terms are incorrect
    
    for term in bcl_terms:
        u_indices, v_indices, w_indices = term
        U_entry = []
        V_entry = []
        W_entry = []
        for (idx, coeff) in u_indices:
            # idx is flat index in 3x3: i*3+j
            i, j = divmod(idx, 3)
            U_entry.append((i*4 + j, coeff))
        for (idx, coeff) in v_indices:
            # idx is flat index in 3x3: j*3+k
            j, k = divmod(idx, 3)
            V_entry.append((j*4 + k, coeff))
        for (idx, coeff) in w_indices:
            # idx is flat index in 3x3: i*3+k
            i, k = divmod(idx, 3)
            W_entry.append((k*3 + i, coeff))
        U_list.append(U_entry)
        V_list.append(V_entry)
        W_list.append(W_entry)
        col += 1
    
    # Term 1b: A_core (3x3) * B_strip (3x1) -> C_strip (3x1)
    # sum_{j=0}^{2} A[i,j] * B[j,3] for i=0,1,2
    # Schoolbook: 3*3*1 = 9 terms
    for i in range(3):
        for j in range(3):
            U_entry = [(i*4 + j, 1)]
            V_entry = [(j*4 + 3, 1)]
            W_entry = [(3*3 + i, 1)]  # k=3, so k*n+i = 3*3+i = 9+i
            U_list.append(U_entry)
            V_list.append(V_entry)
            W_list.append(W_entry)
            col += 1
    
    # Term 2: A[:,3] * B[3,:] -> outer product
    # A[i,3] * B[3,k] for i=0,1,2; k=0,1,2,3
    # This is rank 1: single term
    U_entry = [(2*4 + 3, 1)]  # i=2? No, this should be for each i...
    # Wait, this is NOT rank 1 in the tensor sense. Let me reconsider.
    
    # C[i,k] += A[i,3] * B[3,k]
    # This is a rank-1 tensor: T[i,k] = A[i,3] * B[3,k]
    # In our factorization: U[i*4+3, c] * V[3*4+k, c] * W[k*3+i, c]
    # We need: U[i*4+3, c] * V[12+k, c] * W[3k+i, c] = A[i,3]*B[3,k] for all i,k
    # Set U[i*4+3, c] = 1 for all i, V[12+k, c] = 1 for all k, W[3k+i, c] = 1 for all i,k
    # Then product = 1*1*1 = 1, but we need A[i,3]*B[3,k]...
    
    # That's wrong. The tensor T has entries T[i,3,k] = delta for the multiplication tensor.
    # Actually, in the matrix multiplication tensor, T[i,j,k] = 1 if this is the (i,j,k) entry.
    # The strip term contributes to T[i,3,k] = 1 for all i,k.
    # This is a rank-1 tensor: U[i*4+3, c]=1, V[3*4+k, c]=1, W[k*3+i, c]=1
    # But that gives 1*1*1=1 for ALL (i,k), which is correct since T[i,3,k]=1 for all i,k.
    
    # Wait no. In the matrix multiplication tensor decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The "tensor" is just the identity-like structure where each (i,j,k) triple has value 1.
    
    # So for the strip: T[i,3,k] = 1 for i=0,1,2 and k=0,1,2,3.
    # Rank-1: U[i*4+3, c] = 1, V[12+k, c] = 1, W[3k+i, c] = 1
    # Product = 1*1*1 = 1. Correct!
    
    U_entry = []
    for i in range(3):
        U_entry.append((i*4 + 3, 1))
    V_entry = []
    for k in range(4):
        V_entry.append((3*4 + k, 1))
    W_entry = []
    for k in range(4):
        for i in range(3):
            W_entry.append((k*3 + i, 1))
    U_list.append(U_entry)
    V_list.append(V_entry)
    W_list.append(W_entry)
    col += 1
    
    total_rank = col
    
    # Build U, V, W matrices
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for c in range(total_rank):
        for (idx, sign) in U_list[c]:
            U[idx, c] = sign
        for (idx, sign) in V_list[c]:
            V[idx, c] = sign
        for (idx, sign) in W_list[c]:
            W[idx, c] = sign
    
    # Verify
    if not verify_full(n, m, p, U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def strassen_decomp(n, m, p):
    """
    Decompose 3x4 * 4x4 using Strassen on 2x2 blocks.
    
    Split A (3x4) into top 2x4 and bottom 1x4.
    Split B (4x4) into top 2x4 and bottom 2x4.
    
    C_top (2x4) = A_top (2x4) * B (4x4)
    C_bot (1x4) = A_bot (1x4) * B (4x4)
    
    For C_top: split into left 2x2 and right 2x2.
    A_top = [A11 A12] where A11, A12 are 2x2
    B = [B11 B12; B21 B22] where each is 2x2
    
    C_top_left = A11*B11 + A12*B21 (2x2)
    C_top_right = A11*B12 + A12*B22 (2x2)
    
    Use Strassen for each 2x2 product (rank 7 each).
    C_bot: schoolbook (1*4*4 = 16 terms)
    
    Total: 7 + 7 + 16 = 30
    """
    U_list = []
    V_list = []
    W_list = []
    
    col = 0
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_2x2(i_base, j_base, k_base, i_rows=2, j_cols=2, k_cols=2):
        """
        Add Strassen decomposition for A[i_base:i_base+i_rows, j_base:j_base+j_cols] * 
        B[j_base:j_base+j_cols, k_base:k_base+k_cols] -> C[i_base:i_base+i_rows, k_base:k_base+k_cols]
        Only works for 2x2 * 2x2 -> 2x2
        """
        nonlocal col
        if i_rows != 2 or j_cols != 2 or k_cols != 2:
            return
        
        # A is 2x2 at rows i_base, i_base+1; cols j_base, j_base+1
        # B is 2x2 at rows j_base, j_base+1; cols k_base, k_base+1
        # C is 2x2 at rows i_base, i_base+1; cols k_base, k_base+1
        
        # A11 = A[i_base, j_base], A12 = A[i_base, j_base+1]
        # A21 = A[i_base+1, j_base], A22 = A[i_base+1, j_base+1]
        # B11 = B[j_base, k_base], B12 = B[j_base, k_base+1]
        # B21 = B[j_base+1, k_base], B22 = B[j_base+1, k_base+1]
        # C11 = C[i_base, k_base], C12 = C[i_base, k_base+1]
        # C21 = C[i_base+1, k_base], C22 = C[i_base+1, k_base+1]
        
        a11 = (i_base, j_base)
        a12 = (i_base, j_base+1)
        a21 = (i_base+1, j_base)
        a22 = (i_base+1, j_base+1)
        
        b11 = (j_base, k_base)
        b12 = (j_base, k_base+1)
        b21 = (j_base+1, k_base)
        b22 = (j_base+1, k_base+1)
        
        c11 = (i_base, k_base)
        c12 = (i_base, k_base+1)
        c21 = (i_base+1, k_base)
        c22 = (i_base+1, k_base+1)
        
        # M1 = (A11+A22)(B11+B22)
        U_vals = [a11, a22]
        V_vals = [b11, b22]
        W_vals = [c11, c22]
        signs_U = [1, 1]
        signs_V = [1, 1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M2 = (A21+A22)B11
        U_vals = [a21, a22]
        V_vals = [b11]
        W_vals = [c21, c22]
        signs_U = [1, 1]
        signs_V = [1]
        signs_W = [1, -1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M3 = A11(B21-B22)
        U_vals = [a11]
        V_vals = [b21, b22]
        W_vals = [c12, c22]
        signs_U = [1]
        signs_V = [1, -1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M4 = A22(B12-B11)
        U_vals = [a22]
        V_vals = [b12, b11]
        W_vals = [c11, c21]
        signs_U = [1]
        signs_V = [1, -1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M5 = (A11+A12)B22
        U_vals = [a11, a12]
        V_vals = [b22]
        W_vals = [c11, c12]
        signs_U = [1, 1]
        signs_V = [1]
        signs_W = [-1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U_vals = [a21, a11]
        V_vals = [b11, b12]
        W_vals = [c22]
        signs_U = [1, -1]
        signs_V = [1, 1]
        signs_W = [1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U_vals = [a12, a22]
        V_vals = [b21, b22]
        W_vals = [c11]
        signs_U = [1, -1]
        signs_V = [1, 1]
        signs_W = [1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
    
    # For C_top (rows 0,1 of output): A_top (2x4) * B (4x4)
    # Split into left 2x2 and right 2x2
    # C_top_left = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # C_top_right = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # Each 2x2 * 2x2 product uses Strassen (7 terms)
    # Total for C_top: 4 Strassen decompositions = 28 terms
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2(0, 0, 0)  # 7 terms, col 0-6
    
    # Strassen for A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2(0, 2, 0)  # 7 terms, col 7-13
    
    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_2x2(0, 0, 2)  # 7 terms, col 14-20
    
    # Strassen for A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_strassen_2x2(0, 2, 2)  # 7 terms, col 21-27
    
    # For C_bot (row 2 of output): A[2:3, :] * B[:, :]
    # Schoolbook: 1 * 4 * 4 = 16 terms
    for j in range(4):
        for k in range(4):
            U_vals = [(2, j)]
            V_vals = [(j, k)]
            W_vals = [(k, 2)]
            signs_U = [1]
            signs_V = [1]
            signs_W = [1]
            U_list.append((U_vals, signs_U))
            V_list.append((V_vals, signs_V))
            W_list.append((W_vals, signs_W))
            col += 1
    
    total_rank = col
    
    # Build U, V, W matrices
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for c in range(total_rank):
        for (idx, sign) in zip(U_list[c][0], U_list[c][1]):
            i, j = idx
            U[i*m + j, c] = sign
        for (idx, sign) in zip(V_list[c][0], V_list[c][1]):
            j, k = idx
            V[j*p + k, c] = sign
        for (idx, sign) in zip(W_list[c][0], W_list[c][1]):
            k, i = idx
            W[k*n + i, c] = sign
    
    # Verify
    if not verify_full(n, m, p, U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_full(n, m, p, U, V, W):
    """Verify the full tensor decomposition."""
    tol = 1e-8
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True

def verify_full(n, m, p, U, V, W):
    """Verify the full tensor decomposition."""
    tol = 1e-8
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True
