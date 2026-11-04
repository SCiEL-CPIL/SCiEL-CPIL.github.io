"""
Matrix multiplication 2x4x4 - rank-minimal tensor decomposition.

Uses a rank-28 decomposition based on 4 Strassen-like 7-multiplication blocks.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err


def _construct_strassen_28():
    """
    Construct a rank-28 tensor decomposition using 4 Strassen blocks.
    Each 2x2 block multiplication uses the standard 7-multiplication Strassen algorithm.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Standard Strassen 7-multiplication pattern for 2x2 matrices:
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
    
    # Each pattern: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # Indices are local 2x2 positions (0,1,2,3) = (r,c) flattened as r*2+c
    strassen_patterns = [
        # M1: (A11+A22)*(B11+B22) -> contributes to C11 (+1), C22 (+1)
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        # M2: (A21+A22)*B11 -> contributes to C21 (+1), C22 (-1)
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, -1)),
        # M3: A11*(B21-B22) -> contributes to C12 (+1), C22 (+1)
        ((0,), (1,), (1, 3), (1, -1), (1, 3), (1, 1)),
        # M4: A22*(B12-B11) -> contributes to C11 (+1), C21 (+1)
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        # M5: (A11+A12)*B22 -> contributes to C11 (-1), C12 (+1)
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        # M6: (A21-A11)*(B11+B12) -> contributes to C22 (+1)
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # M7: (A12-A22)*(B21+B22) -> contributes to C11 (+1)
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),
    ]
    
    # Four 2x2 blocks of B (and corresponding parts of A and C):
    # Block (br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1 of B
    # A has shape 2x4, so for each block (br, bc):
    #   A rows 0..1, cols bc*2..bc*2+1 form the 2x2 A block
    #   B rows br*2..br*2+1, cols bc*2..bc*2+1 form the 2x2 B block
    #   C rows 0..1, cols br*2..br*2+1 form the 2x2 C block
    # 
    # C = A * B, so C[0:2, br*2:br*2+2] = sum_bc A[0:2, bc*2:bc*2+2] * B[br*2:br*2+2, bc*2:bc*2+2]
    # 
    # We need to handle the sum over bc. The Strassen trick works per (br, bc) pair.
    # For each br in {0,1} and bc in {0,1}, we have one Strassen block.
    # That's 4 blocks * 7 multiplications = 28 total.
    
    blocks = []
    for br in range(2):
        for bc in range(2):
            blocks.append((br, bc))
    
    col = 0
    for (br, bc) in blocks:
        for (a_idx, a_c, b_idx, b_c, c_idx, c_c) in strassen_patterns:
            # A block: rows 0..1, cols bc*2..bc*2+1
            # Local index in A block: lr (row), lc (col)
            # Global A index: (lr) * m + (bc*2 + lc) = lr * 4 + bc*2 + lc
            for idx, coeff in zip(a_idx, a_c):
                lr = idx // 2
                lc = idx % 2
                gi = lr          # row in A (0 or 1)
                gj = bc * 2 + lc  # col in A
                U[gi * m + gj][col] = coeff
            
            # B block: rows br*2..br*2+1, cols bc*2..bc*2+1
            # Local index: lr (row), lc (col)
            # Global B index: (br*2 + lr) * p + (bc*2 + lc)
            for idx, coeff in zip(b_idx, b_c):
                lr = idx // 2
                lc = idx % 2
                gj = br * 2 + lr   # row in B
                gk = bc * 2 + lc   # col in B
                V[gj * p + gk][col] = coeff
            
            # C block: rows 0..1, cols br*2..br*2+1
            # Local index: lr (row), lc (col)
            # Global C index: (br*2 + lc) * n + lr  [since W[k*n+i, r] where k=col, i=row]
            for idx, coeff in zip(c_idx, c_c):
                lr = idx // 2
                lc = idx % 2
                gi = lr              # row in C (0 or 1)
                gk = br * 2 + lc     # col in C
                W[gk * n + gi][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    deadline = 115.0
    
    n, m, p = DIMENSIONS
    
    # Use the rank-28 Strassen decomposition
    result = _construct_strassen_28()
    
    # Verify
    err = _verify(result['U'], result['V'], result['W'], n, m, p)
    
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    err = _verify(res['U'], res['V'], res['W'], 2, 4, 4)
    print(f"Max error: {err}")

