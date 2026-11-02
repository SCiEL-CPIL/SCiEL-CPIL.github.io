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
    deadline = start_time + 10.0 # Safety margin
    
    n, m, p = 2, 4, 8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    # Indexing:
    # U[i*m + j] corresponds to A[i, j]
    # V[j*p + k] corresponds to B[j, k]
    # W[k*n + i] corresponds to C[k, i]
    
    terms = []
    
    for j in range(4):
        # Block j handles output columns C[:, 2j] and C[:, 2j+1]
        # Input rows from B are 0:2 for left half of A, 2:4 for right half of A
        
        # A_L = A[:, 0:2], A_R = A[:, 2:4]
        # B_Lj = B[0:2, 2j:2j+2], B_Rj = B[2:4, 2j:2j+1]
        
        # Strassen for A_L * B_Lj
        # M1 = (A11+A22)(B11+B22)
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        # M2 = A22(B21-B11)
        terms.append({
            'A': {(1,1): 1},
            'B': {(1, 2*j): 1, (0, 2*j): -1},
            'C': {}
        })
        # M3 = (A11-A22)(B21+B22)
        terms.append({
            'A': {(0,0): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        # M4 = A11(B11-B22)
        terms.append({
            'A': {(0,0): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): -1},
            'C': {}
        })
        # M5 = (A11+A12)B22
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1, 2*j+1): 1},
            'C': {}
        })
        # M6 = (A21-A11)(B11+B12)
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0, 2*j): 1, (0, 2*j+1): 1},
            'C': {}
        })
        # M7 = (A12-A22)(B21+B22)
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # Strassen for A_R * B_Rj
        # Note: A_R indices are 0,2 and 1,3 in A matrix.
        # B_Rj indices are 2, 2j and 3, 2j+1 in B matrix.
        
        # M8 = (A11+A22)(B11+B22) for the right block
        terms.append({
            'A': {(0,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        # M9 = A22(B21-B11) for the right block
        terms.append({
            'A': {(1,3): 1},
            'B': {(3, 2*j): 1, (2, 2*j): -1},
            'C': {}
        })
        # M10 = (A11-A22)(B21+B22) for the right block
        terms.append({
            'A': {(0,2): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        # M11 = A11(B11-B22) for the right block
        terms.append({
            'A': {(0,2): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): -1},
            'C': {}
        })
        # M12 = (A11+A12)B22 for the right block
        terms.append({
            'A': {(0,2): 1, (0,3): 1},
            'B': {(3, 2*j+1): 1},
            'C': {}
        })
        # M13 = (A21-A11)(B11+B12) for the right block
        terms.append({
            'A': {(1,2): 1, (0,2): -1},
            'B': {(2, 2*j): 1, (2, 2*j+1): 1},
            'C': {}
        })
        # M14 = (A12-A22)(B21+B22) for the right block
        terms.append({
            'A': {(0,3): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        
        if time.time() > deadline:
            break

    # Compute C contributions
    # For a 2x2 * 2x2 multiplication:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For block j:
    # Left Strassen (A_L * B_Lj) contributes to C[:, 2j] and C[:, 2j+1]
    # Output of Left Strassen is 2x2:
    #   L11 -> C[0, 2j]
    #   L12 -> C[0, 2j+1]
    #   L21 -> C[1, 2j]
    #   L22 -> C[1, 2j+1]
    
    # Right Strassen (A_R * B_Rj) contributes to C[:, 2j] and C[:, 2j+1]
    # Output of Right Strassen is 2x2:
    #   R11 -> C[0, 2j]
    #   R12 -> C[0, 2j+1]
    #   R21 -> C[1, 2j]
    #   R22 -> C[1, 2j+1]
    
    # Indices in W:
    # C[0, k] -> W[k*n + 0] = W[k]
    # C[1, k] -> W[k*n + 1] = W[k+2] for n=2? No, W index is k*n + i.
    # n=2.
    # C[0, 2j] -> W[2j]
    # C[0, 2j+1] -> W[2j+1]
    # C[1, 2j] -> W[2j+2]
    # C[1, 2j+1] -> W[2j+3]
    
    for j in range(4):
        base = j * 14
        
        # Left Strassen Terms (indices 0-6 relative to base)
        # T0: M1, T1: M2, T2: M3, T3: M4, T4: M5, T5: M6, T6: M7
        
        # L11 = M1 + M4 - M5 + M7
        terms[base + 0]['C'][2*j] = 1
        terms[base + 3]['C'][2*j] = 1
        terms[base + 4]['C'][2*j] = -1
        terms[base + 6]['C'][2*j] = 1
        
        # L12 = M3 + M5
        terms[base + 2]['C'][2*j+1] = 1
        terms[base + 4]['C'][2*j+1] = 1
        
        # L21 = M2 + M4
        terms[base + 1]['C'][2*j+2] = 1
        terms[base + 3]['C'][2*j+2] = 1
        
        # L22 = M1 - M2 + M3 + M6
        terms[base + 0]['C'][2*j+3] = 1
        terms[base + 1]['C'][2*j+3] = -1
        terms[base + 2]['C'][2*j+3] = 1
        terms[base + 5]['C'][2*j+3] = 1
        
        # Right Strassen Terms (indices 7-13 relative to base)
        # T7: M8, T8: M9, T9: M10, T10: M11, T11: M12, T12: M13, T13: M14
        
        # R11 = M8 + M11 - M12 + M14
        terms[base + 7]['C'][2*j] = 1
        terms[base + 10]['C'][2*j] = 1
        terms[base + 11]['C'][2*j] = -1
        terms[base + 13]['C'][2*j] = 1
        
        # R12 = M10 + M12
        terms[base + 9]['C'][2*j+1] = 1
        terms[base + 11]['C'][2*j+1] = 1
        
        # R21 = M9 + M11
        terms[base + 8]['C'][2*j+2] = 1
        terms[base + 10]['C'][2*j+2] = 1
        
        # R22 = M8 - M9 + M10 + M13
        terms[base + 7]['C'][2*j+3] = 1
        terms[base + 8]['C'][2*j+3] = -1
        terms[base + 9]['C'][2*j+3] = 1
        terms[base + 12]['C'][2*j+3] = 1
        
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
