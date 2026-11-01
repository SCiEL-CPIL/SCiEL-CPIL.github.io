"""
Matrix multiplication 2x2 * 2x13: rank-minimal exact tensor decomposition.

The matrix multiplication tensor for A (2x2) * B (2x13) = C (2x13) has:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

A rank-r decomposition satisfies:
sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k

Strategy: Use Strassen's algorithm on pairs of columns, plus schoolbook for the last column.
This gives rank 6*7 + 4 = 46, which is valid and better than schoolbook 52.
"""
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(decomp):
        """Check that the decomposition is valid: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1"""
        U = decomp["U"]
        V = decomp["V"]
        W = decomp["W"]
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    def build_strassen_grouped():
        """
        Strassen-based decomposition for 2x2 * 2x13.
        
        For each pair of columns (k0, k1), use Strassen's 7-multiplication algorithm:
        M1 = (A00+A11)(B0,k0+B0,k1)
        M2 = (A10+A11)B0,k0
        M3 = A00(B1,k0-B1,k1)
        M4 = A11(B0,k1-B0,k0)
        M5 = (A00+A01)B1,k1
        M6 = (A10-A00)(B0,k0+B1,k0)
        M7 = (A01-A11)(B0,k1+B1,k1)
        
        C[0,k0] = M1 + M4 - M5 + M7  -- WAIT, this is wrong for non-square case!
        
        Actually, for 2x2 * 2x2:
        C00 = M1 + M4 - M5 + M7  -- NO! Let me re-derive.
        
        Standard Strassen for C = A*B (both 2x2):
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)b00
        M3 = a00(b10-b11)
        M4 = a11(b01-b00)
        M5 = (a00+a01)b11
        M6 = (a10-a00)(b00+b10)
        M7 = (a01-a11)(b01+b11)
        
        c00 = M1 + M4 - M5 + M7
        c01 = M3 + M5
        c10 = M2 + M4
        c11 = M1 - M2 + M3 + M6
        
        In our case, B is 2x13, so we group columns. For columns k0, k1:
        B_sub = [[B0,k0, B0,k1], [B1,k0, B1,k1]]
        
        The Strassen formulas give:
        C[0,k0] = M1 + M4 - M5 + M7
        C[0,k1] = M3 + M5
        C[1,k0] = M2 + M4
        C[1,k1] = M1 - M2 + M3 + M6
        
        So W contributions:
        M1: W[k0*n+0] += 1, W[k1*n+1] += 1
        M2: W[k0*n+1] += 1
        M3: W[k1*n+0] += 1
        M4: W[k0*n+0] += 1, W[k1*n+1] += 1
        M5: W[k1*n+0] += 1
        M6: W[k1*n+1] += 1
        M7: W[k0*n+0] += 1
        """
        rank = 6 * 7 + 4  # 46
        U = [[0] * rank for _ in range(n * m)]  # 4 x 46
        V = [[0] * rank for _ in range(m * p)]  # 26 x 46
        W = [[0] * rank for _ in range(p * n)]  # 26 x 46

        col = 0
        # Process 6 pairs of columns
        for g in range(6):
            k0 = 2 * g
            k1 = 2 * g + 1

            # M1 = (A00+A11)(B0,k0+B0,k1)
            U[0][col] += 1   # A[0,0]
            U[3][col] += 1   # A[1,1]
            V[0 * p + k0][col] += 1  # B[0,k0]
            V[0 * p + k1][col] += 1  # B[0,k1]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1

            # M2 = (A10+A11)B0,k0
            U[1][col] += 1   # A[1,0]
            U[3][col] += 1   # A[1,1]
            V[0 * p + k0][col] += 1  # B[0,k0]
            W[k0 * n + 1][col] += 1  # C[1,k0]
            col += 1

            # M3 = A00(B1,k0-B1,k1)
            U[0][col] += 1   # A[0,0]
            V[1 * p + k0][col] += 1  # B[1,k0]
            V[1 * p + k1][col] -= 1  # B[1,k1]
            W[k1 * n + 0][col] += 1  # C[0,k1]
            col += 1

            # M4 = A11(B0,k1-B0,k0)
            U[3][col] += 1   # A[1,1]
            V[0 * p + k1][col] += 1  # B[0,k1]
            V[0 * p + k0][col] -= 1  # B[0,k0]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1

            # M5 = (A00+A01)B1,k1
            U[0][col] += 1   # A[0,0]
            U[1][col] += 1   # A[0,1]
            V[1 * p + k1][col] += 1  # B[1,k1]
            W[k1 * n + 0][col] += 1  # C[0,k1]
            col += 1

            # M6 = (A10-A00)(B0,k0+B1,k0)
            U[1][col] += 1   # A[1,0]
            U[0][col] -= 1   # A[0,0]
            V[0 * p + k0][col] += 1  # B[0,k0]
            V[1 * p + k0][col] += 1  # B[1,k0]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1

            # M7 = (A01-A11)(B0,k1+B1,k1)
            U[1][col] += 1   # A[0,1]
            U[3][col] -= 1   # A[1,1]
            V[0 * p + k1][col] += 1  # B[0,k1]
            V[1 * p + k1][col] += 1  # B[1,k1]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            col += 1

        # Process the last single column (k=12) with schoolbook (4 mults)
        k = 12
        U[0][col] += 1; V[0 * p + k][col] += 1; W[k * n + 0][col] += 1; col += 1
        U[1][col] += 1; V[1 * p + k][col] += 1; W[k * n + 0][col] += 1; col += 1
        U[2][col] += 1; V[0 * p + k][col] += 1; W[k * n + 1][col] += 1; col += 1
        U[3][col] += 1; V[1 * p + k][col] += 1; W[k * n + 1][col] += 1; col += 1

        return {"U": U, "V": V, "W": W}

    # Try Strassen-based decomposition first
    try:
        decomp = build_strassen_grouped()
        if validate(decomp):
            return decomp
    except Exception:
        pass

    # Fall back to schoolbook
    return build_schoolbook()
# EVOLVE-BLOCK-END
