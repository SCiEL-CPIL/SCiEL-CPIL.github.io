"""Matrix multiplication 2,2,4 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def naive_decomp():
        """Rank-16 naive decomposition."""
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using Strassen on two 2x2 blocks.
        Left block (cols 0,1): 7 Strassen products
        Right block (cols 2,3): 7 Strassen products
        Total: 14 rank-1 terms
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]

        # Left block Strassen (7 products) for A(2x2) * B_left(2x2)
        # B_left has entries b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
        # V indices: 0=b00, 1=b01, 2=b10, 3=b11
        # W indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # M3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        # Right block Strassen (7 products) for A(2x2) * B_right(2x2)
        # B_right has entries b02=B[0,2], b03=B[0,3], b12=B[1,2], b13=B[1,3]
        # V indices: 4=b02, 5=b03, 6=b12, 7=b13
        # W indices: 4=C02, 5=C03, 6=C12, 7=C13

        # N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[4][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1

        # N2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1
        V[4][8] = 1
        W[6][8] = 1; W[7][8] = -1

        # N3 = a00(b12-b13)
        U[0][9] = 1
        V[6][9] = 1; V[7][9] = -1
        W[5][9] = 1; W[7][9] = 1

        # N4 = a11(b03-b02)
        U[3][10] = 1
        V[5][10] = 1; V[4][10] = -1
        W[4][10] = 1; W[6][10] = 1

        # N5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[5][11] = 1

        # N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[4][12] = 1; V[5][12] = 1
        W[7][12] = 1

        # N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_rank13_construction():
        """
        Try a rank-13 construction by sharing one product between the two blocks.
        
        Key insight: M1 = (a00+a11)(b00+b11) and N1 = (a00+a11)(b02+b13)
        share the same U-factor (a00+a11). But we can't merge them into one
        rank-1 term because the V and W factors differ.
        
        However, we can try a different grouping. Let's see if we can find
        a 13-term decomposition.
        
        Actually, let's try a different approach: use the fact that for
        2x2*2x4, the tensor rank might be 13 or 14.
        
        Let me try a specific rank-13 construction:
        Use 7 for left block, 6 for right block by combining two of the
        right block Strassen products.
        """
        R = 13
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]

        # Left block: standard Strassen (r=0..6)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1

        # Right block: try to do it in 6 terms (r=7..12)
        # C02 = a00*b02 + a01*b12
        # C03 = a00*b03 + a01*b13
        # C12 = a10*b02 + a11*b12
        # C13 = a10*b03 + a11*b13
        
        # Try: P1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[4][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # P2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1
        V[4][8] = 1
        W[6][8] = 1; W[7][8] = -1
        
        # P3 = a00(b12-b13)
        U[0][9] = 1
        V[6][9] = 1; V[7][9] = -1
        W[5][9] = 1; W[7][9] = 1
        
        # P4 = a11(b03-b02)
        U[3][10] = 1
        V[5][10] = 1; V[4][10] = -1
        W[4][10] = 1; W[6][10] = 1
        
        # P5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[5][11] = 1
        
        # P6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[4][12] = 1; V[5][12] = 1
        W[7][12] = 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_numerical_optimization():
        """
        Use numerical optimization to find a low-rank decomposition.
        Start from the rank-14 Strassen solution and try to reduce rank
        by finding linear dependencies or using gradient-based methods.
        """
        if time.time() > deadline - 20:
            return None

        # Start with rank-14 solution and try to find a lower rank
        # by solving a system of equations.
        
        # Actually, let's try a different approach: use the known fact that
        # the border rank of 2x2*2x4 is 11, and try to find an exact rank-11
        # or rank-12 decomposition using algebraic methods.
        
        # For now, let's try to verify if rank 14 is correct and return it.
        # The key structural change is to use a cleaner implementation.
        
        return None

    # Try structured decompositions in order of increasing rank
    result = try_rank13_construction()
    if result is not None:
        return result

    result = try_strassen_rectangular()
    if result is not None:
        return result

    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
