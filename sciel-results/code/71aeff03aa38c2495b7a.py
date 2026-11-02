import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen decomposition on 2x2 blocks.
    - The 2x4 * 4x8 multiplication is partitioned into 4 independent 2x4 * 4x2 multiplications (column pairs).
    - Each 2x4 * 4x2 multiplication is A_L B_L + A_R B_R, computed via two Strassen 2x2 * 2x2 multiplications.
    - Each Strassen 2x2 * 2x2 uses rank 7.
    - Total rank per block: 14.
    - Total rank: 4 * 14 = 56.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    n, m, p = 2, 4, 8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    # Indexing:
    # U[i*m + j] corresponds to A[i, j]
    # V[j*p + k] corresponds to B[j, k]
    # W[k*n + i] corresponds to C[i, k]  (k is column, i is row)
    
    terms = []
    
    for j in range(4):
        # Block j handles output columns C[:, 2j] and C[:, 2j+1]
        
        # A_L = A[:, 0:2], A_R = A[:, 2:4]
        # B_Lj = B[0:2, 2j:2j+2], B_Rj = B[2:4, 2j:2j+2]
        
        # Strassen for A_L * B_Lj (left block)
        # A_L = [[A00, A01], [A10, A11]] where A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B_Lj = [[B00, B01], [B10, B11]] where B00=B[0,2j], B01=B[0,2j+1], B10=B[1,2j], B11=B[1,2j+1]
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        # M2 = (A10+A11)B00
        terms.append({
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1},
            'C': {}
        })
        # M3 = A00(B10-B11)
        terms.append({
            'A': {(0,0): 1},
            'B': {(1, 2*j): 1, (1, 2*j+1): -1},
            'C': {}
        })
        # M4 = A11(B00-B01)
        terms.append({
            'A': {(1,1): 1},
            'B': {(0, 2*j): 1, (0, 2*j+1): -1},
            'C': {}
        })
        # M5 = (A00+A01)B11
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1, 2*j+1): 1},
            'C': {}
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0, 2*j): 1, (0, 2*j+1): 1},
            'C': {}
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # Strassen for A_R * B_Rj (right block)
        # A_R = [[A02, A03], [A12, A13]] where A02=A[0,2], A03=A[0,3], A12=A[1,2], A13=A[1,3]
        # B_Rj = [[B20, B21], [B30, B31]] where B20=B[2,2j], B21=B[2,2j+1], B30=B[3,2j], B31=B[3,2j+1]
        
        # M8 = (A02+A13)(B20+B31)
        terms.append({
            'A': {(0,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        # M9 = (A12+A13)B20
        terms.append({
            'A': {(1,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1},
            'C': {}
        })
        # M10 = A02(B30-B31)
        terms.append({
            'A': {(0,2): 1},
            'B': {(3, 2*j): 1, (3, 2*j+1): -1},
            'C': {}
        })
        # M11 = A13(B20-B21)
        terms.append({
            'A': {(1,3): 1},
            'B': {(2, 2*j): 1, (2, 2*j+1): -1},
            'C': {}
        })
        # M12 = (A02+A03)B31
        terms.append({
            'A': {(0,2): 1, (0,3): 1},
            'B': {(3, 2*j+1): 1},
            'C': {}
        })
        # M13 = (A12-A02)(B20+B21)
        terms.append({
            'A': {(1,2): 1, (0,2): -1},
            'B': {(2, 2*j): 1, (2, 2*j+1): 1},
            'C': {}
        })
        # M14 = (A03-A13)(B30+B31)
        terms.append({
            'A': {(0,3): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        
        if time.time() > deadline:
            break
    
    # Compute C contributions
    # For a 2x2 * 2x2 multiplication:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For block j:
    # Left Strassen (A_L * B_Lj) contributes to C[:, 2j] and C[:, 2j+1]
    # Output of Left Strassen is 2x2:
    #   L00 -> C[0, 2j]
    #   L01 -> C[0, 2j+1]
    #   L10 -> C[1, 2j]
    #   L11 -> C[1, 2j+1]
    
    # Right Strassen (A_R * B_Rj) contributes to C[:, 2j] and C[:, 2j+1]
    # Output of Right Strassen is 2x2:
    #   R00 -> C[0, 2j]
    #   R01 -> C[0, 2j+1]
    #   R10 -> C[1, 2j]
    #   R11 -> C[1, 2j+1]
    
    # W index mapping: W[k*n + i] for C[i, k]
    # C[0, 2j] -> W[2j*2 + 0] = W[4j]
    # C[0, 2j+1] -> W[(2j+1)*2 + 0] = W[4j+2]
    # C[1, 2j] -> W[2j*2 + 1] = W[4j+1]
    # C[1, 2j+1] -> W[(2j+1)*2 + 1] = W[4j+3]
    
    for j in range(4):
        base = j * 14
        
        # Left Strassen Terms (indices 0-6 relative to base)
        # T0: M1, T1: M2, T2: M3, T3: M4, T4: M5, T5: M6, T6: M7
        
        # L00 = M1 + M4 - M5 + M7 -> C[0, 2j] -> W[4j]
        terms[base + 0]['C'][4*j] = 1
        terms[base + 3]['C'][4*j] = 1
        terms[base + 4]['C'][4*j] = -1
        terms[base + 6]['C'][4*j] = 1
        
        # L01 = M3 + M5 -> C[0, 2j+1] -> W[4j+2]
        terms[base + 2]['C'][4*j+2] = 1
        terms[base + 4]['C'][4*j+2] = 1
        
        # L10 = M2 + M4 -> C[1, 2j] -> W[4j+1]
        terms[base + 1]['C'][4*j+1] = 1
        terms[base + 3]['C'][4*j+1] = 1
        
        # L11 = M1 - M2 + M3 + M6 -> C[1, 2j+1] -> W[4j+3]
        terms[base + 0]['C'][4*j+3] = 1
        terms[base + 1]['C'][4*j+3] = -1
        terms[base + 2]['C'][4*j+3] = 1
        terms[base + 5]['C'][4*j+3] = 1
        
        # Right Strassen Terms (indices 7-13 relative to base)
        # T7: M8, T8: M9, T9: M10, T10: M11, T11: M12, T12: M13, T13: M14
        
        # R00 = M8 + M11 - M12 + M14 -> C[0, 2j] -> W[4j]
        terms[base + 7]['C'][4*j] = 1
        terms[base + 10]['C'][4*j] = 1
        terms[base + 11]['C'][4*j] = -1
        terms[base + 13]['C'][4*j] = 1
        
        # R01 = M10 + M12 -> C[0, 2j+1] -> W[4j+2]
        terms[base + 9]['C'][4*j+2] = 1
        terms[base + 11]['C'][4*j+2] = 1
        
        # R10 = M9 + M11 -> C[1, 2j] -> W[4j+1]
        terms[base + 8]['C'][4*j+1] = 1
        terms[base + 10]['C'][4*j+1] = 1
        
        # R11 = M8 - M9 + M10 + M13 -> C[1, 2j+1] -> W[4j+3]
        terms[base + 7]['C'][4*j+3] = 1
        terms[base + 8]['C'][4*j+3] = -1
        terms[base + 9]['C'][4*j+3] = 1
        terms[base + 12]['C'][4*j+3] = 1
        
        if time.time() > deadline:
            break
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for (i, j_idx), coeff in term['A'].items():
            U[i * m + j_idx][c] = coeff
        for (j_idx, k), coeff in term['B'].items():
            V[j_idx * p + k][c] = coeff
        for (idx, coeff) in term['C'].items():
            if coeff != 0:
                W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
