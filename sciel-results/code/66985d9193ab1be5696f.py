import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Start with schoolbook rank 64 as fallback
    rank_school = n * m * p  # 64
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to construct a rank-56 decomposition using Strassen on 2x2x2 blocks
    # Strategy: group j into 2 pairs, k into 4 pairs
    # For each (j_pair, k_pair), we have a 2x2x2 sub-tensor
    # Use Strassen rank-7 for each, but need consistent U,V,W
    
    # Actually, let's try a direct rank-56 construction:
    # View as 4 independent 2x2x4 computations (one per pair of k columns)
    # Each 2x2x4: split into two 2x2x2, each rank 7, total 14
    # 4 x 14 = 56
    
    # But the j indices are shared! So we can't just do independent Strassen.
    # 
    # Alternative: Use the "parallel" Strassen approach
    # The 2x4x8 tensor can be seen as having structure where we apply
    # Strassen to the (j,k) plane for each i.
    
    # Let me try a different approach: direct rank-56 via known decomposition
    # For 2x4x8, we can use the following:
    # Think of B (4x8) as 4 rows. A (2x4) as 2 rows.
    # Result[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j into j1 in {0,1} and j2 in {2,3}
    # Result[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2]*B[j2,k]
    #
    # For the first term: A1 (2x2) * B1 (2x8) -> 2x8
    # For the second term: A2 (2x2) * B2 (2x8) -> 2x8
    #
    # Each 2x2 * 2x8: split B into 4 blocks of 2x2
    # 2x2 * 2x2 uses rank 7 (Strassen)
    # So 2x2 * 2x8 uses 4 * 7 = 28
    # Two such: 2 * 28 = 56
    #
    # Total rank: 56
    #
    # But wait - the U matrix entries for A1 and A2 are different,
    # and V entries for B1 and B2 are different. So they don't share
    # any Strassen multiplications. Total = 56.
    
    # Let's construct this:
    # For each of the 2 A-blocks (A1, A2) and 4 B-blocks (B11,B12,B13,B14 for B1, similarly for B2)
    # We do a 2x2x2 Strassen multiplication
    
    # Strassen 2x2x2: 
    # Given C = A * B where A,B are 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M4
    # c21 = M2 + M5
    # c22 = M1 - M2 + M3 + M7
    
    # For a 2x2 * 2x2 multiplication, the tensor decomposition has rank 7.
    # U entries (for the 2x2 A matrix, flattened as [a11, a12, a21, a22]):
    # V entries (for the 2x2 B matrix, flattened as [b11, b12, b21, b22]):
    # W entries (for the 2x2 C matrix, flattened as [c11, c12, c21, c22]):
    
    # The 7 Strassen terms give us the rank-7 decomposition of the 2x2x2 tensor.
    
    # For our problem:
    # A is 2x4, B is 4x8, C is 2x8
    # We split into 8 sub-problems: 2 (A-blocks) x 4 (B-blocks)
    # Each sub-problem is 2x2 * 2x2 -> 2x2, rank 7
    # Total rank: 8 * 7 = 56
    
    rank_56 = 56
    U_56 = np.zeros((nm, rank_56))
    V_56 = np.zeros((mp, rank_56))
    W_56 = np.zeros((pn, rank_56))
    
    # Strassen coefficients for 2x2x2:
    # U coefficients for each of the 7 terms (row in 2x2 A matrix):
    # Term 1: (a11+a22) -> U: [1, 0, 0, 1]
    # Term 2: (a21+a22) -> U: [0, 0, 1, 1]
    # Term 3: a11 -> U: [1, 0, 0, 0]
    # Term 4: a22 -> U: [0, 0, 0, 1]
    # Term 5: (a11+a12) -> U: [1, 1, 0, 0]
    # Term 6: (a21-a11) -> U: [-1, 0, 1, 0]
    # Term 7: (a12-a22) -> U: [0, 1, 0, -1]
    
    U_strassen = np.array([
        [1, 0, 0, 1],   # term 1
        [0, 0, 1, 1],   # term 2
        [1, 0, 0, 0],   # term 3
        [0, 0, 0, 1],   # term 4
        [1, 1, 0, 0],   # term 5
        [-1, 0, 1, 0],  # term 6
        [0, 1, 0, -1]   # term 7
    ])  # shape (7, 4) - 7 terms, 4 entries in 2x2 A
    
    # V coefficients for each of the 7 terms (row in 2x2 B matrix):
    # Term 1: (b11+b22) -> V: [1, 0, 0, 1]
    # Term 2: b11 -> V: [1, 0, 0, 0]
    # Term 3: (b12-b22) -> V: [0, 1, 0, -1]
    # Term 4: (b21-b11) -> V: [-1, 0, 1, 0]
    # Term 5: b22 -> V: [0, 0, 0, 1]
    # Term 6: (b11+b12) -> V: [1, 1, 0, 0]
    # Term 7: (b21+b22) -> V: [0, 0, 1, 1]
    
    V_strassen = np.array([
        [1, 0, 0, 1],   # term 1
        [1, 0, 0, 0],   # term 2
        [0, 1, 0, -1],  # term 3
        [-1, 0, 1, 0],  # term 4
        [0, 0, 0, 1],   # term 5
        [1, 1, 0, 0],   # term 6
        [0, 0, 1, 1]    # term 7
    ])  # shape (7, 4) - 7 terms, 4 entries in 2x2 B
    
    # W coefficients for each of the 7 terms (row in 2x2 C matrix):
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M4
    # c21 = M2 + M5
    # c22 = M1 - M2 + M3 + M7
    
    W_strassen = np.array([
        [1, 0, 0, 0],   # M1 contributes to c11, c22
        [0, 0, 1, 0],   # M2 contributes to c21
        [0, 1, 0, 0],   # M3 contributes to c12
        [-1, 1, 0, 0],  # M4 contributes to c11(-), c12(+)
        [1, 0, 1, 0],   # M5 contributes to c11, c21
        [0, 0, 0, 0],   # M6 doesn't appear in standard Strassen output? Wait...
        [0, 0, 0, 1]    # M7 contributes to c11(-), c22(+)
    ])
    
    # Wait, I need to double-check. Let me redo:
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M4
    # c21 = M2 + M5
    # c22 = M1 - M2 + M3 + M7
    
    # W for each M term (column = c11, c12, c21, c22):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [-1, 1, 0, 0]
    # M5: [1, 0, 1, 0]
    # M6: [0, 0, 0, 0]  -- M6 is not used in the standard Strassen formula above!
    # M7: [-1, 0, 0, 1]
    
    # Hmm, M6 = (a21-a11)(b11+b12) is not in the standard 7-term Strassen for the output.
    # Let me recheck the standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M7
    
    # Yes, M6 is not used in the output! That's fine - it's still part of the 7 multiplications
    # but its contribution to C is zero. So W for M6 is [0,0,0,0].
    
    W_strassen = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [-1, 1, 0, 0],  # M4
        [1, 0, 1, 0],   # M5
        [0, 0, 0, 0],   # M6
        [-1, 0, 0, 1]   # M7
    ])  # shape (7, 4)
    
    # Now build the full 56-rank decomposition:
    # 2 A-blocks (j=0,1 and j=2,3) x 4 B-blocks (k=0,1; k=2,3; k=4,5; k=6,7)
    # = 8 sub-problems, each rank 7
    
    col = 0
    for a_block in range(2):  # a_block 0: j=0,1; a_block 1: j=2,3
        for b_block in range(4):  # b_block 0: k=0,1; 1: k=2,3; 2: k=4,5; 3: k=6,7
            for s in range(7):  # 7 Strassen terms
                # U: A is 2x4, we're using A[:, a_block*2 : a_block*2+2]
                # A flat index: i*m + j = i*4 + j
                # For a_block=0: j=0,1 -> flat indices 0,1,4,5 for i=0,1
                # For a_block=1: j=2,3 -> flat indices 2,3,6,7 for i=0,1
                for i in range(2):
                    for j_local in range(2):
                        j_global = a_block * 2 + j_local
                        flat_idx = i * 4 + j_global
                        U_56[flat_idx, col] = U_strassen[s, j_local]  # Wait, U_strassen[s] is for the 2x2 A block
                        
                        # U_strassen[s] has 4 entries for the 2x2 A: [a11, a12, a21, a22]
                        # a11 = A[0, a_block*2], a12 = A[0, a_block*2+1]
                        # a21 = A[1, a_block*2], a22 = A[1, a_block*2+1]
                        # So U_strassen[s, 0] -> A[0, a_block*2], flat = 0*4 + a_block*2
                        # U_strassen[s, 1] -> A[0, a_block*2+1], flat = 0*4 + a_block*2+1
                        # U_strassen[s, 2] -> A[1, a_block*2], flat = 1*4 + a_block*2
                        # U_strassen[s, 3] -> A[1, a_block*2+1], flat = 1*4 + a_block*2+1
                        
                # Let me redo this properly:
                # Reset U for this column
                U_56[:, col] = 0
                V_56[:, col] = 0
                W_56[:, col] = 0
                
                # U: 2x4 A matrix, using columns a_block*2 and a_block*2+1
                # U_strassen[s] = [coeff for A[0,a_block*2], coeff for A[0,a_block*2+1], 
                #                   coeff for A[1,a_block*2], coeff for A[1,a_block*2+1]]
                U_56[0 * 4 + a_block * 2 + 0, col] = U_strassen[s, 0]
                U_56[0 * 4 + a_block * 2 + 1, col] = U_strassen[s, 1]
                U_56[1 * 4 + a_block * 2 + 0, col] = U_strassen[s, 2]
                U_56[1 * 4 + a_block * 2 + 1, col] = U_strassen[s, 3]
                
                # V: 4x8 B matrix, using rows a_block*2 and a_block*2+1 (same j range)
                # and columns b_block*2 and b_block*2+1
                # V flat index: j*p + k = j*8 + k
                # V_strassen[s] = [coeff for B[a_block*2, b_block*2], coeff for B[a_block*2, b_block*2+1],
                #                  coeff for B[a_block*2+1, b_block*2], coeff for B[a_block*2+1, b_block*2+1]]
                V_56[(a_block * 2 + 0) * 8 + b_block * 2 + 0, col] = V_strassen[s, 0]
                V_56[(a_block * 2 + 0) * 8 + b_block * 2 + 1, col] = V_strassen[s, 1]
                V_56[(a_block * 2 + 1) * 8 + b_block * 2 + 0, col] = V_strassen[s, 2]
                V_56[(a_block * 2 + 1) * 8 + b_block * 2 + 1, col] = V_strassen[s, 3]
                
                # W: 2x8 C matrix, using rows 0,1 and columns b_block*2, b_block*2+1
                # W flat index: k*n + i = k*2 + i
                # W_strassen[s] = [coeff for C[0, b_block*2], coeff for C[0, b_block*2+1],
                #                  coeff for C[1, b_block*2], coeff for C[1, b_block*2+1]]
                W_56[(b_block * 2 + 0) * 2 + 0, col] = W_strassen[s, 0]
                W_56[(b_block * 2 + 0) * 2 + 1, col] = W_strassen[s, 1]
                W_56[(b_block * 2 + 1) * 2 + 0, col] = W_strassen[s, 2]
                W_56[(b_block * 2 + 1) * 2 + 1, col] = W_strassen[s, 3]
                
                col += 1
    
    # Verify the rank-56 decomposition
    U_t = U_56.T.reshape(n, m, rank_56)  # (2, 4, 56)
    V_t = V_56.T.reshape(m, p, rank_56)  # (4, 8, 56)
    W_t = W_56.T.reshape(p, n, rank_56)  # (8, 2, 56)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    target = np.ones((n, m, p))
    
    if np.allclose(result, target, atol=1e-10):
        return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
