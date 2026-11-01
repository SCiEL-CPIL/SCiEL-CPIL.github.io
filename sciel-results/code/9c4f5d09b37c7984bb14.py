"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_rank14_factors():
        """
        Returns the rank-14 Strassen decomposition for 2x2 * 2x4.
        This is constructed by applying Strassen's algorithm independently
        to the two 2x2 blocks of the 2x4 matrix B.
        """
        R = 14
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]

        # Left block (cols 0,1 of B -> rows 0,1 of C)
        # Indices:
        # U: i*m + j (0..3)
        # V: j*p + k (0..7). Left block k=0,1 -> V indices 0,1 (j=0) and 2,3 (j=1)?
        # Wait, V index is j*p + k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 4. j=1, k=1 -> 5.
        # W index is k*n + i.
        # k=0, i=0 -> 0. k=0, i=1 -> 1.
        # k=1, i=0 -> 2. k=1, i=1 -> 3.

        # r=0: M1 = (a00+a11)(b00+b10) ? No, Strassen is (a00+a11)(b00+b11) for 2x2.
        # Here B is 2x2 for the block.
        # B_block = [[b00, b01], [b10, b11]]
        # Indices in V:
        # b00 is at V[0] (j=0, k=0)
        # b01 is at V[1] (j=0, k=1)
        # b10 is at V[4] (j=1, k=0)
        # b11 is at V[5] (j=1, k=1)
        
        # Standard Strassen for 2x2 * 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Map to our indices:
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01, 4=b10, 5=b11
        # W indices: 0=c00, 1=c10, 2=c01, 3=c11
        
        # r=0: m1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # r=1: m2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1

        # r=2: m3
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1

        # r=3: m4
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1

        # r=4: m5
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1

        # r=5: m6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # r=6: m7
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Right block (cols 2,3 of B -> rows 2,3 of C)
        # B_block2 = [[b02, b03], [b12, b13]]
        # V indices:
        # b02 is at V[2] (j=0, k=2)
        # b03 is at V[3] (j=0, k=3)
        # b12 is at V[6] (j=1, k=2)
        # b13 is at V[7] (j=1, k=3)
        
        # W indices:
        # k=2, i=0 -> 4 (c20)
        # k=2, i=1 -> 5 (c21)
        # k=3, i=0 -> 6 (c30)
        # k=3, i=1 -> 7 (c31)

        # r=7: n1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1

        # r=8: n2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1

        # r=9: n3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1

        # r=10: n4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1

        # r=11: n5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1

        # r=12: n6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1

        # r=13: n7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return U, V, W

    def naive_decomp():
        rank = n * m * p
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

    U, V, W = get_rank14_factors()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return naive_decomp()
# EVOLVE-BLOCK-END
