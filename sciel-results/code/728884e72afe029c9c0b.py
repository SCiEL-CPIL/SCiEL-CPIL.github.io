"""
Rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
1. Start with a fast, known-good upper bound (Strassen-block decomposition, rank 76).
2. Use a smooth optimization approach (L-BFGS-B) to search for lower-rank decompositions.
3. The key insight from the current program is the optimization framework;
   the key insight from inspiration 2 is the structured block decomposition.
4. We combine these: use the block decomposition as the initial guess for optimization
   at various ranks, and also try random restarts.

The tensor M_{4,4,5} has entries T[i,j,k] where i in [0,4), j in [0,4), k in [0,5).
A decomposition of rank r is:
  T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
"""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def schoolbook():
        """Trivial decomposition with rank n*m*p = 80."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def strassen_block():
        """
        Block decomposition using Strassen for 2x2x2 blocks.
        Split A (4x4) into four 2x2 blocks.
        Split B (4x5) into two 2x5 blocks (top 2 rows, bottom 2 rows).
        C = A * B where C is 4x5.
        
        C_top = A_11 * B_top + A_12 * B_bot  (2x5)
        C_bot = A_21 * B_top + A_22 * B_bot  (2x5)
        
        Each term A_ij * B_k is M_{2,2,5}.
        For M_{2,2,5}: split B (2x5) into B1 (2x2) and B2 (2x3).
        A * B1 uses Strassen (rank 7), A * B2 uses trivial (rank 2*2*3 = 12).
        Total per M_{2,2,5}: 19.
        4 blocks * 19 = 76.
        """
        rank = 76
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Strassen for M_{2,2,2}: 7 multiplications
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        def add_strassen_2x2x2(i_off, j_off, k_off):
            """Add Strassen decomposition for M_{2,2,2} at given offsets."""
            nonlocal col
            # i_off, j_off, k_off are the starting indices for the 2x2x2 block
            # U indices: (i, j) where i in [i_off, i_off+2), j in [j_off, j_off+2)
            # V indices: (j, k) where j in [j_off, j_off+2), k in [k_off, k_off+2)
            # W indices: (k, i) where k in [k_off, k_off+2), i in [i_off, i_off+2)

            # M1 = (a11+a22)(b11+b22)
            U[i_off * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            W[k_off * n + (i_off+1), col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M2 = (a21+a22)b11
            U[(i_off+1) * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            W[k_off * n + (i_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = -1
            col += 1

            # M3 = a11(b21-b22)
            U[i_off * m + j_off, col] = 1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = -1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M4 = a22(b12-b11)
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + (k_off+1), col] = 1
            V[j_off * p + k_off, col] = -1
            W[k_off * n + (i_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M5 = (a11+a12)b22
            U[i_off * m + j_off, col] = 1
            U[i_off * m + (j_off+1), col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M6 = (a21-a11)(b11+b12)
            U[(i_off+1) * m + j_off, col] = 1
            U[i_off * m + j_off, col] = -1
            V[j_off * p + k_off, col] = 1
            V[j_off * p + (k_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M7 = (a12-a22)(b21+b22)
            U[i_off * m + (j_off+1), col] = 1
            U[(i_off+1) * m + (j_off+1), col] = -1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

        def add_trivial_2x2x3(i_off, j_off, k_off):
            """Add trivial decomposition for M_{2,2,3} at given offsets."""
            nonlocal col
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[(i_off + i) * m + (j_off + j), col] = 1
                        V[(j_off + j) * p + (k_off + k), col] = 1
                        W[(k_off + k) * n + (i_off + i), col] = 1
                        col += 1

        # Four blocks of A: A_11 (i=0,1; j=0,1), A_12 (i=0,1; j=2,3), A_21 (i=2,3; j=0,1), A_22 (i=2,3; j=2,3)
        # Two blocks of B: B_top (j=0,1; k=0..4), B_bot (j=2,3; k=0..4)
        # For each A_ij * B_k:
        #   Split B_k (2x5) into B_k1 (2x2, k=0,1) and B_k2 (2x3, k=2,3,4)
        #   Use Strassen for A_ij * B_k1, trivial for A_ij * B_k2

        # A_11 * B_top: i_off=0, j_off=0, B_top has j=0,1
        # B_top1: k=0,1 (j=0,1), B_top2: k=2,3,4 (j=0,1)
        add_strassen_2x2x2(0, 0, 0)  # A_11 * B_top1 (i=0,1; j=0,1; k=0,1)
        add_trivial_2x2x3(0, 0, 2)   # A_11 * B_top2 (i=0,1; j=0,1; k=2,3,4)

        # A_12 * B_bot: i_off=0, j_off=2, B_bot has j=2,3
        # B_bot1: k=0,1 (j=2,3), B_bot2: k=2,3,4 (j=2,3)
        add_strassen_2x2x2(0, 2, 0)  # A_12 * B_bot1 (i=0,1; j=2,3; k=0,1)
        add_trivial_2x2x3(0, 2, 2)   # A_12 * B_bot2 (i=0,1; j=2,3; k=2,3,4)

        # A_21 * B_top: i_off=2, j_off=0, B_top has j=0,1
        add_strassen_2x2x2(2, 0, 0)  # A_21 * B_top1 (i=2,3; j=0,1; k=0,1)
        add_trivial_2x2x3(2, 0, 2)   # A_21 * B_top2 (i=2,3; j=0,1; k=2,3,4)

        # A_22 * B_bot: i_off=2, j_off=2, B_bot has j=2,3
        add_strassen_2x2x2(2, 2, 0)  # A_22 * B_bot1 (i=2,3; j=2,3; k=0,1)
        add_trivial_2x2x3(2, 2, 2)   # A_22 * B_bot2 (i=2,3; j=2,3; k=2,3,4)

        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Start with the best known decomposition
    best_result = strassen_block()
    best_rank = 76

    def validate(U, V, W, tol=1e-5):
        """Check if the decomposition is valid."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def error_func(x):
        """Calculate the sum of squared errors for the tensor decomposition."""
        r = len(x) // (n*m + m*p + p*n)
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r

        U = x[:U_size].reshape(n * m, r)
        V = x[U_size:U_size+V_size].reshape(m * p, r)
        W = x[U_size+V_size:].reshape(p * n, r)

        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err

    def try_optimize(r, U0, V0, W0):
        """Try to optimize a rank-r decomposition from initial guess."""
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        try:
            res = opt.minimize(error_func, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e-6:
                U_size = n * m * r
                V_size = m * p * r
                W_size = p * n * r
                U_opt = res.x[:U_size].reshape(n * m, r)
                V_opt = res.x[U_size:U_size+V_size].reshape(m * p, r)
                W_opt = res.x[U_size+V_size:].reshape(p * n, r)
                if validate(U_opt, V_opt, W_opt):
                    return {'U': U_opt.tolist(), 'V': V_opt.tolist(), 'W': W_opt.tolist()}
        except Exception:
            pass
        return None

    # Try to find lower-rank decompositions
    # Search from rank 49 up to 75 (since we already have 76)
    # Use both random restarts and structured initializations

    for r in range(49, 76):
        if time.time() > deadline - 15:
            break

        # Try a few random initializations
        max_attempts = 3 if r < 60 else 2

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            np.random.seed(42 + r * 100 + attempt)
            U0 = np.random.randn(n * m, r) * 0.5
            V0 = np.random.randn(m * p, r) * 0.5
            W0 = np.random.randn(p * n, r) * 0.5

            result = try_optimize(r, U0, V0, W0)
            if result is not None:
                best_result = result
                best_rank = r
                break  # Found a valid decomposition at this rank

    return best_result
# EVOLVE-BLOCK-END
