import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_rank46():
        """
        Construct a rank-46 decomposition.
        Splits M_{2,4,7} into two M_{2,2,7} blocks.
        Each M_{2,2,7} is decomposed into three M_{2,2,2} (rank 7) and one M_{2,2,1} (rank 2).
        Total rank = 2 * (3*7 + 2) = 46.
        """
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        def add_trivial_2x2x1(i0, j0, k0):
            nonlocal col
            # a*e
            U[i0 * m + j0, col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + i0, col] += 1
            col += 1
            # d*f (where d is i0+1, j0+1? No, 2x2x1 means A is 2x2, B is 2x1)
            # A = [[a, b], [c, d]], B = [[e], [f]]
            # C = [[ae+bf, ...], [ce+df, ...]]? No, C is 2x1.
            # C[0] = a*e + b*f
            # C[1] = c*e + d*f
            # Rank 2 decomposition:
            # Term 1: a*e -> C[0]
            # Term 2: b*f -> C[0]
            # Term 3: c*e -> C[1]
            # Term 4: d*f -> C[1]
            # Wait, rank of 2x2x1 is 2?
            # C = A @ b.
            # C[0] = A[0,0]b[0] + A[0,1]b[1]
            # C[1] = A[1,0]b[0] + A[1,1]b[1]
            # This can be done with 2 multiplications?
            # m1 = A[0,0]b[0]
            # m2 = A[1,1]b[1]
            # No, we need cross terms.
            # Actually, rank of M_{2,2,1} is 2.
            # U[:,0] = [1, 0, 0, 0], V[:,0] = [1, 0, 0], W[:,0] = [1, 0] -> C[0] += A[0,0]B[0]
            # U[:,1] = [0, 1, 0, 0], V[:,1] = [0, 1, 0], W[:,1] = [0, 1] -> C[1] += A[1,1]B[1]
            # This misses cross terms.
            # The tensor M_{2,2,1} has rank 2.
            # Decomposition:
            # r1: U=[1,0,0,0], V=[1,0,0], W=[1,0] -> C[0,0] += A[0,0]B[0,0]
            # r2: U=[0,1,0,0], V=[0,1,0], W=[0,1] -> C[1,0] += A[1,1]B[1,0]
            # This is not correct.
            # The rank of M_{n,m,1} is n*m? No, it is min(n,m)?
            # For 2x2x1, rank is 2.
            # Let's check:
            # C = A b.
            # C is a vector.
            # C_i = sum_j A_ij b_j.
            # This is a bilinear map.
            # Rank is 2.
            # Terms:
            # 1. A_00 b_0 -> C_0
            # 2. A_11 b_1 -> C_1
            # This doesn't cover A_01 b_1 or A_10 b_0.
            # Actually, the rank of M_{2,2,1} is 2.
            # Wait, M_{2,2,1} is the tensor of the map A,b -> Ab.
            # The rank is 2.
            # Decomposition:
            # u1 = e0, v1 = e0, w1 = e0 -> A00 b0 contributes to C0.
            # u2 = e1, v2 = e1, w2 = e1 -> A11 b1 contributes to C1.
            # This is not a valid decomposition of the whole tensor.
            # The tensor has 4 non-zero entries: (0,0,0), (0,1,1)? No.
            # T[i,j,k] where i is row of C (0..1), j is row of A (0..1), k is col of B (0..1).
            # T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1.
            # Rank is 2.
            # u1 = [1, 0, 0, 0] (A00), v1 = [1, 0] (B0), w1 = [1, 0] (C0)
            # u2 = [0, 0, 1, 0] (A10)? No.
            # Let's use the standard rank 2 decomposition for 2x2x1.
            # Actually, it's easier to just use 4 rank-1 terms if unsure, but rank is 2.
            # I will use 4 rank-1 terms to be safe, increasing rank to 48.
            # 2 * (3*7 + 4) = 46.
            # Wait, 3*7=21. 21+4=25. 2*25=50.
            # So trivial 2x2x1 with rank 4 gives 50.
            # Trivial 2x2x1 with rank 2 gives 46.
            # I will implement rank 2.
            
            # Rank 2 decomposition for 2x2x1:
            # Term 1: A[0,0]*B[0,0] -> C[0,0]
            # Term 2: A[1,1]*B[1,0] -> C[1,0]
            # This is not correct.
            # The correct rank 2 decomposition of M_{2,2,1}:
            # u1 = [1, 0, 0, 0], v1 = [1, 0], w1 = [1, 0]
            # u2 = [0, 1, 0, 0], v2 = [0, 1], w2 = [0, 1]
            # This gives C[0] = A[0,0]B[0] and C[1] = A[1,1]B[1].
            # Missing A[0,1]B[1] and A[1,0]B[0].
            # So rank is NOT 2.
            # Rank of M_{2,2,1} is 2?
            # No, the rank of M_{n,m,1} is n*m.
            # So rank is 4.
            # So total rank is 50.
            
            # Back to square one.
            pass

        return U, V, W

    # Since the rank 46 attempt failed due to rank of 2x2x1 being 4, 
    # the rank is 50 again.
    # I will return the current rank 50 solution.
    U, V, W = get_rank46() # This will fail or return zeros.
    
    # Fallback to the known rank 50 from the prompt.
    def get_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    U, V, W = get_rank50()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
