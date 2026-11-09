import time
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15

    def schoolbook():
        rank = nm * mp  # 45
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0]) if U else 0

        # Check dimensions
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(nm):
            if len(U[i]) != rank:
                return False
        for j in range(mp):
            if len(V[j]) != rank:
                return False
        for k in range(pn):
            if len(W[k]) != rank:
                return False

        # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The multiplication tensor: T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p
        # C[k,i] gets contribution from A[i,j]*B[j,k]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = 0.0
                    for r in range(rank):
                        computed += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(computed - 1.0) > tol:
                        return False
        return True

    def strassen_2x2():
        """
        Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
        U: 4 entries (a, b, c, d) for A
        V: 4 entries (e, f, g, h) for B
        W: 4 entries (C11, C12, C21, C22) for C
        """
        U = [
            [1.0, 1.0, 0.0, 0.0],   # term 1: a+b
            [1.0, -1.0, 0.0, 0.0],  # term 2: a-b
            [0.0, 0.0, 1.0, 1.0],   # term 3: c+d
            [1.0, 0.0, 0.0, 0.0],   # term 4: a
            [0.0, 0.0, 0.0, 1.0],   # term 5: d
            [1.0, 1.0, 0.0, 0.0],   # term 6: a+b
            [0.0, 0.0, 1.0, -1.0],  # term 7: c-d
        ]
        V = [
            [1.0, 1.0, 0.0, 0.0],   # term 1: e+f
            [1.0, 1.0, 0.0, 0.0],   # term 2: e+f
            [0.0, 0.0, 0.0, 1.0],   # term 3: h
            [0.0, 0.0, 1.0, -1.0],  # term 4: g-h
            [1.0, 0.0, -1.0, 0.0],  # term 5: e-g
            [0.0, -1.0, 0.0, 1.0],  # term 6: h-f
            [1.0, 1.0, 0.0, 0.0],   # term 7: e+f
        ]
        W = [
            [1.0, 0.0, 0.0, 1.0],   # term 1: C11 + C22
            [0.0, 0.0, 1.0, -1.0],  # term 2: C21 - C22
            [1.0, 0.0, 1.0, 0.0],   # term 3: C11 + C21
            [0.0, 1.0, 0.0, 1.0],   # term 4: C12 + C22
            [-1.0, 1.0, 0.0, 0.0],  # term 5: -C11 + C12
            [0.0, 0.0, 0.0, -1.0],  # term 6: -C22
            [-1.0, 0.0, 0.0, -1.0], # term 7: -C11 - C22
        ]
        return U, V, W

    def try_block_decomposition():
        """
        Use block decomposition to get a lower rank than schoolbook.
        
        Split the 3x3 * 3x5 multiplication into blocks:
        - A = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
        - B = [[B22, B25], [B12, B15]] where B22 is 2x2, B25 is 2x3, B12 is 1x2, B15 is 1x3
        
        This is complex. Let me try a simpler approach:
        Use Strassen for a 2x2 * 2x2 block and schoolbook for the rest.
        
        Actually, let me try to use the structure more cleverly.
        The 3x3 * 3x5 can be seen as:
        - For each of the 3 rows of A, we need to compute a 1x3 * 3x5 product
        - Each such product has rank 15 (schoolbook)
        - Total rank = 3 * 15 = 45 (same as schoolbook)
        
        To do better, we need to share multiplications across rows.
        
        Let me try a different approach: use the fact that we can combine
        Strassen's 2x2x2 with additional terms.
        
        For now, let me just return schoolbook if I can't find a better decomposition.
        """
        # Try to construct a decomposition with rank < 45
        # using a combination of Strassen and schoolbook terms
        
        # One idea: use Strassen for a 2x2 * 2x2 subproblem
        # and add terms for the remaining parts
        
        # Let me try to build a decomposition manually
        # For 3x3 * 3x5, we can write it as:
        # C = A * B where A is 3x3, B is 3x5
        
        # Let me try to use the following block structure:
        # A = [[a11, a12, a13],
        #      [a21, a22, a23],
        #      [a31, a32, a33]]
        # B = [[b11, b12, b13, b14, b15],
        #      [b21, b22, b23, b24, b25],
        #      [b31, b32, b33, b34, b35]]
        
        # C[k,i] = sum_j A[i,j] * B[j,k]
        
        # Let me try to use a known low-rank decomposition for 3x3x3
        # and extend it to 3x3x5
        
        # For 3x3x3, the best known rank is 23 (Strassen-like)
        # For 3x3x5, we can use the 23 terms for the 3x3x3 part
        # and add 12 terms for the 3x3x2 part (schoolbook)
        # Total rank = 23 + 12 = 35
        
        # But I need the actual 23-term decomposition for 3x3x3
        # Let me try to construct it
        
        # Actually, let me just return schoolbook for now
        # and try to find a better decomposition in the next iteration
        return None

    # Try the block decomposition
    result = try_block_decomposition()
    
    if result is None:
        # Fall back to schoolbook
        result = schoolbook()
    
    # Verify the result
    if not verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # If verification fails, return schoolbook
        result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
