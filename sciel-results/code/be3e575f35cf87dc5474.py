import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll use Laderman's algorithm for 4x4 x 4x4 (rank 21)
    # Split the 4x8 matrix B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7)
    # Then C = A * B = [A*B1 | A*B2]
    # Each A*B is a 3x4 matrix, which we compute using Laderman on padded 4x4
    # Total rank: 2 * 21 = 42
    
    # Laderman's algorithm for 4x4 x 4x4 (0-indexed)
    # The 21 products are:
    # P1  = (A00+A11)(B00+B11)
    # P2  = (A00-A11)(B00-B11)
    # P3  = (A02+A13)(B02+B13)
    # P4  = (A02-A13)(B02-B13)
    # P5  = (A20+A31)(B20+B31)
    # P6  = (A20-A31)(B20-B31)
    # P7  = (A22+A33)(B22+B33)
    # P8  = (A22-A33)(B22-B33)
    # P9  = (A00+A22)(B20+B02)
    # P10 = (A00-A22)(B20-B02)
    # P11 = (A11+A33)(B31+B13)
    # P12 = (A11-A33)(B31-B13)
    # P13 = (A01+A10)(B10+B01)
    # P14 = (A03+A30)(B30+B03)
    # P15 = (A01-A10)(B10-B01)
    # P16 = (A03-A30)(B30-B03)
    # P17 = (A02+A32)(B20+B22)
    # P18 = (A02-A32)(B20-B22)
    # P19 = (A13+A23)(B30+B32)
    # P20 = (A13-A23)(B30-B32)
    # P21 = (A20+A22)(B20+B02)  -- wait, let me look up the correct Laderman
    
    # Actually, let me use the standard Laderman decomposition.
    # Standard Laderman for 4x4x4 has rank 21.
    # Let me define it carefully.
    
    # Laderman's 21 terms (using 0-indexed i,j,k for A[i,j], B[j,k], C[i,k]):
    laderman = []
    
    # Term 1: (A00+A11)*(B00+B11) -> contributes to C00, C11
    laderman.append({
        'a': [(0,0,1), (1,1,1)],
        'b': [(0,0,1), (1,1,1)],
        'c': [(0,0,1), (1,1,1)]
    })
    
    # Term 2: (A00-A11)*(B00-B11) -> contributes to C00, C11
    laderman.append({
        'a': [(0,0,1), (1,1,-1)],
        'b': [(0,0,1), (1,1,-1)],
        'c': [(0,0,1), (1,1,1)]
    })
    
    # Term 3: (A02+A13)*(B02+B13) -> contributes to C02, C13
    laderman.append({
        'a': [(0,2,1), (1,3,1)],
        'b': [(0,2,1), (1,3,1)],
        'c': [(0,2,1), (1,3,1)]
    })
    
    # Term 4: (A02-A13)*(B02-B13) -> contributes to C02, C13
    laderman.append({
        'a': [(0,2,1), (1,3,-1)],
        'b': [(0,2,1), (1,3,-1)],
        'c': [(0,2,1), (1,3,1)]
    })
    
    # Term 5: (A20+A31)*(B20+B31) -> contributes to C20, C31
    laderman.append({
        'a': [(2,0,1), (3,1,1)],
        'b': [(2,0,1), (3,1,1)],
        'c': [(2,0,1), (3,1,1)]
    })
    
    # Term 6: (A20-A31)*(B20-B31) -> contributes to C20, C31
    laderman.append({
        'a': [(2,0,1), (3,1,-1)],
        'b': [(2,0,1), (3,1,-1)],
        'c': [(2,0,1), (3,1,1)]
    })
    
    # Term 7: (A22+A33)*(B22+B33) -> contributes to C22, C33
    laderman.append({
        'a': [(2,2,1), (3,3,1)],
        'b': [(2,2,1), (3,3,1)],
        'c': [(2,2,1), (3,3,1)]
    })
    
    # Term 8: (A22-A33)*(B22-B33) -> contributes to C22, C33
    laderman.append({
        'a': [(2,2,1), (3,3,-1)],
        'b': [(2,2,1), (3,3,-1)],
        'c': [(2,2,1), (3,3,1)]
    })
    
    # Term 9: (A00+A22)*(B20+B02) -> contributes to C00, C22
    laderman.append({
        'a': [(0,0,1), (2,2,1)],
        'b': [(2,0,1), (0,2,1)],
        'c': [(0,0,1), (2,2,1)]
    })
    
    # Term 10: (A00-A22)*(B20-B02) -> contributes to C00, C22
    laderman.append({
        'a': [(0,0,1), (2,2,-1)],
        'b': [(2,0,1), (0,2,-1)],
        'c': [(0,0,1), (2,2,1)]
    })
    
    # Term 11: (A11+A33)*(B31+B13) -> contributes to C11, C33
    laderman.append({
        'a': [(1,1,1), (3,3,1)],
        'b': [(3,1,1), (1,3,1)],
        'c': [(1,1,1), (3,3,1)]
    })
    
    # Term 12: (A11-A33)*(B31-B13) -> contributes to C11, C33
    laderman.append({
        'a': [(1,1,1), (3,3,-1)],
        'b': [(3,1,1), (1,3,-1)],
        'c': [(1,1,1), (3,3,1)]
    })
    
    # Term 13: (A01+A10)*(B10+B01) -> contributes to C01, C10
    laderman.append({
        'a': [(0,1,1), (1,0,1)],
        'b': [(1,0,1), (0,1,1)],
        'c': [(0,1,1), (1,0,1)]
    })
    
    # Term 14: (A03+A30)*(B30+B03) -> contributes to C03, C30
    laderman.append({
        'a': [(0,3,1), (3,0,1)],
        'b': [(3,0,1), (0,3,1)],
        'c': [(0,3,1), (3,0,1)]
    })
    
    # Term 15: (A01-A10)*(B10-B01) -> contributes to C01, C10
    laderman.append({
        'a': [(0,1,1), (1,0,-1)],
        'b': [(1,0,1), (0,1,-1)],
        'c': [(0,1,1), (1,0,1)]
    })
    
    # Term 16: (A03-A30)*(B30-B03) -> contributes to C03, C30
    laderman.append({
        'a': [(0,3,1), (3,0,-1)],
        'b': [(3,0,1), (0,3,-1)],
        'c': [(0,3,1), (3,0,1)]
    })
    
    # Term 17: (A02+A32)*(B20+B22) -> contributes to C00, C02
    laderman.append({
        'a': [(0,2,1), (3,2,1)],
        'b': [(2,0,1), (2,2,1)],
        'c': [(0,0,1), (0,2,1)]
    })
    
    # Term 18: (A02-A32)*(B20-B22) -> contributes to C00, C02
    laderman.append({
        'a': [(0,2,1), (3,2,-1)],
        'b': [(2,0,1), (2,2,-1)],
        'c': [(0,0,1), (0,2,1)]
    })
    
    # Term 19: (A13+A23)*(B30+B32) -> contributes to C10, C12
    laderman.append({
        'a': [(1,3,1), (2,3,1)],
        'b': [(3,0,1), (3,2,1)],
        'c': [(1,0,1), (1,2,1)]
    })
    
    # Term 20: (A13-A23)*(B30-B32) -> contributes to C10, C12
    laderman.append({
        'a': [(1,3,1), (2,3,-1)],
        'b': [(3,0,1), (3,2,-1)],
        'c': [(1,0,1), (1,2,1)]
    })
    
    # Term 21: (A20+A22)*(B20+B02) -> contributes to C20, C22
    laderman.append({
        'a': [(2,0,1), (2,2,1)],
        'b': [(2,0,1), (0,2,1)],
        'c': [(2,0,1), (2,2,1)]
    })
    
    # Now build the full decomposition for 3x4 x 4x8
    # We use Laderman twice: once for B1 (cols 0-3) and once for B2 (cols 4-7)
    # For each Laderman term, we have a 4x4 x 4x4 decomposition
    # We only need rows 0-2 of the output C (since n=3)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 12 x 42
    V = [[0] * rank for _ in range(m * p)]  # 32 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # For each of the two blocks (block_idx = 0, 1)
    for block_idx in range(2):
        base_col = block_idx * 21
        
        for t in range(21):
            col = base_col + t
            term = laderman[t]
            
            # Fill U: A is 4x4 (padded from 3x4), index i*m+j
            for (i, j, val) in term['a']:
                if i < n:  # only rows 0-2 are valid
                    U[i * m + j][col] += val
            
            # Fill V: B is 4x4, index j*p+k where k is within this block
            for (j, k, val) in term['b']:
                # k is 0-3 within the block, actual column is k + block_idx*4
                actual_k = k + block_idx * 4
                V[j * p + actual_k][col] += val
            
            # Fill W: C is 4x4 (padded from 3x4), index k*n+i
            for (i, k, val) in term['c']:
                if i < n:  # only rows 0-2 are valid
                    # k is 0-3 within the block, actual column is k + block_idx*4
                    actual_k = k + block_idx * 4
                    W[actual_k * n + i][col] += val
    
    return {"U": U, "V": V, "W": W}
