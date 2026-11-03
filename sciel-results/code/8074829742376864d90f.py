"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit

    def schoolbook(n, m, p):
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

    def strassen_2x2x2():
        """Strassen's algorithm for 2x2x2 matrix multiplication, rank 7."""
        U = [[0]*7 for _ in range(4)]  # n*m = 4
        V = [[0]*7 for _ in range(4)]  # m*p = 4
        W = [[0]*7 for _ in range(4)]  # p*n = 4

        # Strassen's 7 products for 2x2 matrices
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[c11,c12],[c21,c22]]
        #
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6

        # U indices: [i*m+j] for i in {0,1}, j in {0,1}
        #   U[0] = a, U[1] = b, U[2] = c, U[3] = d
        # V indices: [j*p+k] for j in {0,1}, k in {0,1}
        #   V[0] = e, V[1] = f, V[2] = g, V[3] = h
        # W indices: [k*n+i] for k in {0,1}, i in {0,1}
        #   W[0] = c11, W[1] = c12, W[2] = c21, W[3] = c22

        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: c11=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1

        # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: c21=1, c22=-1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: c12=1, c22=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: c11=1, c21=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: c11=-1, c12=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: c22=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: c11=1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return {"U": U, "V": V, "W": W}

    def combine_decomps(decomp_list):
        """Combine multiple decompositions by concatenating rank dimensions."""
        total_rank = sum(d["U"][0].__len__() for d in decomp_list)
        U = [[0] * total_rank for _ in range(decomp_list[0]["U"].__len__())]
        V = [[0] * total_rank for _ in range(decomp_list[0]["V"].__len__())]
        W = [[0] * total_rank for _ in range(decomp_list[0]["W"].__len__())]
        offset = 0
        for d in decomp_list:
            r = d["U"][0].__len__()
            for i in range(len(d["U"])):
                for j in range(r):
                    U[i][offset + j] = d["U"][i][j]
            for i in range(len(d["V"])):
                for j in range(r):
                    V[i][offset + j] = d["V"][i][j]
            for i in range(len(d["W"])):
                for j in range(r):
                    W[i][offset + j] = d["W"][i][j]
            offset += r
        return {"U": U, "V": V, "W": W}

    def block_multiply(n, m, p):
        """
        Recursively decompose matrix multiplication n x m by m x p.
        Uses Strassen for 2x2x2 blocks and schoolbook for smaller/odd cases.
        """
        # Base case: small dimensions, use schoolbook
        if n <= 1 and m <= 1 and p <= 1:
            return schoolbook(n, m, p)
        if n * m * p <= 8:
            return schoolbook(n, m, p)

        # Try to split into 2x2x2 blocks where possible
        # Split n, m, p into even parts + remainders
        n2 = n // 2
        m2 = m // 2
        p2 = p // 2
        nr = n % 2
        mr = m % 2
        pr = p % 2

        decomps = []

        if n2 > 0 and m2 > 0 and p2 > 0:
            # Main 2x2x2 block: multiply (n2 x m2) by (m2 x p2)
            # This is a sub-problem of size n2, m2, p2
            main = block_multiply(n2, m2, p2)
            decomps.append(main)

        # Handle remainders with schoolbook
        if nr > 0 and m2 > 0 and p2 > 0:
            # Row remainder: (1 x m2) by (m2 x p2)
            d = schoolbook(nr, m2, p2)
            decomps.append(d)
        if n2 > 0 and mr > 0 and p2 > 0:
            # Col remainder in A: (n2 x 1) by (1 x p2)
            d = schoolbook(n2, mr, p2)
            decomps.append(d)
        if n2 > 0 and m2 > 0 and pr > 0:
            # Col remainder in B: (n2 x m2) by (m2 x 1)
            d = schoolbook(n2, m2, pr)
            decomps.append(d)
        if nr > 0 and mr > 0 and p2 > 0:
            # (1 x 1) by (1 x p2)
            d = schoolbook(nr, mr, p2)
            decomps.append(d)
        if nr > 0 and m2 > 0 and pr > 0:
            # (1 x m2) by (m2 x 1)
            d = schoolbook(nr, m2, pr)
            decomps.append(d)
        if n2 > 0 and mr > 0 and pr > 0:
            # (n2 x 1) by (1 x 1)
            d = schoolbook(n2, mr, pr)
            decomps.append(d)
        if nr > 0 and mr > 0 and pr > 0:
            # (1 x 1) by (1 x 1)
            d = schoolbook(nr, mr, pr)
            decomps.append(d)

        if len(decomps) == 1:
            return decomps[0]
        return combine_decomps(decomps)

    # First, get a baseline with schoolbook
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p

    # Try block decomposition
    try:
        if time.time() < deadline:
            block_result = block_multiply(n, m, p)
            block_rank = len(block_result["U"][0])
            if block_rank < best_rank:
                best_result = block_result
                best_rank = block_rank
    except Exception:
        pass

    # Verify the result using numpy
    def verify(decomp, n, m, p):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]

        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check it equals the multiplication tensor
        # T[i,j,k] should be 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor for matrix mult has T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes)
        # Wait, the tensor decomposition of matrix multiplication:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor T[i,j,k] = delta (it's just 1 for all i,j,k in the standard basis)
        # Actually the multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # because C[i,k] = sum_j A[i,j] * B[j,k]

        # Check: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Verify our best result
    try:
        if not verify(best_result, n, m, p):
            # Fallback to schoolbook if verification fails
            best_result = schoolbook(n, m, p)
            best_rank = n * m * p
    except Exception:
        best_result = schoolbook(n, m, p)
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END
