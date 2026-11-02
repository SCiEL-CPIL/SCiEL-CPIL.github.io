# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen decomposition on 2x2 blocks.
    - The 2x4 * 4x8 multiplication is partitioned into 8 independent 2x2 * 2x2 multiplications.
    - Each 2x2 * 2x2 multiplication uses Strassen's algorithm with rank 7.
    - Total rank: 8 * 7 = 56.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = 2, 4, 8
    
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (16, rank)
    
    terms = []
    
    for j in range(4):
        # B1j = B[0:2, 2j:2j+2]
        # B2j = B[2:4, 2j:2j+2]
        
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        
        # Strassen for A1 * B1j
        # X = A1, Y = B1j
        # X11 = A[0,0], X12 = A[0,1], X21 = A[1,0], X22 = A[1,1]
        # Y11 = B[0,2j], Y12 = B[0,2j+1], Y21 = B[1,2j], Y22 = B[1,2j+1]
        
        # M1 = (X11 + X22) * (Y11 + Y22)
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # M2 = (X21 + X22) * Y11
        terms.append({
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1},
            'C': {}
        })
        
        # M3 = X11 * (Y12 - Y22)
        terms.append({
            'A': {(0,0): 1},
            'B': {(0, 2*j+1): 1, (1, 2*j+1): -1},
            'C': {}
        })
        
        # M4 = X22 * (Y21 - Y11)
        terms.append({
            'A': {(1,1): 1},
            'B': {(1, 2*j): 1, (0, 2*j): -1},
            'C': {}
        })
        
        # M5 = (X11 + X12) * Y22
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1, 2*j+1): 1},
            'C': {}
        })
        
        # M6 = (X21 - X11) * (Y11 + Y12)
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0, 2*j): 1, (0, 2*j+1): 1},
            'C': {}
        })
        
        # M7 = (X12 - X22) * (Y21 + Y22)
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # Strassen for A2 * B2j
        # X = A2, Y = B2j
        # X11 = A[0,2], X12 = A[0,3], X21 = A[1,2], X22 = A[1,3]
        # Y11 = B[2,2j], Y12 = B[2,2j+1], Y21 = B[3,2j], Y22 = B[3,2j+1]
        
        # M8 = (X11 + X22) * (Y11 + Y22)
        terms.append({
            'A': {(0,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        
        # M9 = (X21 + X22) * Y11
        terms.append({
            'A': {(1,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1},
            'C': {}
        })
        
        # M10 = X11 * (Y12 - Y22)
        terms.append({
            'A': {(0,2): 1},
            'B': {(2, 2*j+1): 1, (3, 2*j+1): -1},
            'C': {}
        })
        
        # M11 = X22 * (Y21 - Y11)
        terms.append({
            'A': {(1,3): 1},
            'B': {(3, 2*j): 1, (2, 2*j): -1},
            'C': {}
        })
        
        # M12 = (X11 + X12) * Y22
        terms.append({
            'A': {(0,2): 1, (0,3): 1},
            'B': {(3, 2*j+1): 1},
            'C': {}
        })
        
        # M13 = (X21 - X11) * (Y11 + Y12)
        terms.append({
            'A': {(1,2): 1, (0,2): -1},
            'B': {(2, 2*j): 1, (2, 2*j+1): 1},
            'C': {}
        })
        
        # M14 = (X12 - X22) * (Y21 + Y22)
        terms.append({
            'A': {(0,3): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
    
    # Now compute the C contributions for each term.
    # For A1*B1j (terms 0-6 for each j):
    #   Cj11 = M1 + M4 - M5 + M7
    #   Cj12 = M3 + M5
    #   Cj21 = M2 + M4
    #   Cj22 = M1 - M2 + M3 + M6
    #
    # For A2*B2j (terms 7-13 for each j):
    #   Cj11 += M8 + M11 - M12 + M14
    #   Cj12 += M10 + M12
    #   Cj21 += M9 + M11
    #   Cj22 += M8 - M9 + M10 + M13
    
    for j in range(4):
        base = j * 14
        # M1 to M7 are at indices base+0 to base+6
        # M8 to M14 are at indices base+7 to base+13
        
        # Cj11: output index = (2j+0)*2 + 0 = 4j
        # Cj12: output index = (2j+1)*2 + 0 = 4j + 2
        # Cj21: output index = (2j+0)*2 + 1 = 4j + 1
        # Cj22: output index = (2j+1)*2 + 1 = 4j + 3
        
        # From A1*B1j:
        terms[base + 0]['C'] = {4*j: 1, 4*j+3: 1}  # M1: Cj11 += 1, Cj22 += 1
        terms[base + 1]['C'] = {4*j+1: 1, 4*j+3: -1}  # M2: Cj21 += 1, Cj22 -= 1
        terms[base + 2]['C'] = {4*j+2: 1, 4*j+3: 1}  # M3: Cj12 += 1, Cj22 += 1
        terms[base + 3]['C'] = {4*j: 1, 4*j+1: 1}  # M4: Cj11 += 1, Cj21 += 1
        terms[base + 4]['C'] = {4*j: -1, 4*j+2: 1}  # M5: Cj11 -= 1, Cj12 += 1
        terms[base + 5]['C'] = {4*j+3: 1}  # M6: Cj22 += 1
        terms[base + 6]['C'] = {4*j: 1}  # M7: Cj11 += 1
        
        # From A2*B2j:
        terms[base + 7]['C'] = {4*j: 1, 4*j+3: 1}  # M8: Cj11 += 1, Cj22 += 1
        terms[base + 8]['C'] = {4*j+1: 1, 4*j+3: -1}  # M9: Cj21 += 1, Cj22 -= 1
        terms[base + 9]['C'] = {4*j+2: 1, 4*j+3: 1}  # M10: Cj12 += 1, Cj22 += 1
        terms[base + 10]['C'] = {4*j: 1, 4*j+1: 1}  # M11: Cj11 += 1, Cj21 += 1
        terms[base + 11]['C'] = {4*j: -1, 4*j+2: 1}  # M12: Cj11 -= 1, Cj12 += 1
        terms[base + 12]['C'] = {4*j+3: 1}  # M13: Cj22 += 1
        terms[base + 13]['C'] = {4*j: 1}  # M14: Cj11 += 1
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for (i, j), coeff in term['A'].items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in term['B'].items():
            V[j * p + k][c] = coeff
        for (idx, coeff) in term['C'].items():
            if coeff != 0:
                W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
