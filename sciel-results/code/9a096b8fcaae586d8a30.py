import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    def verify_exact(U, V, W):
        """Verify exact decomposition with rational arithmetic using Fractions."""
        n_, m_, p_ = n, m, p
        rank = len(U[0])
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m_+j][c]) * Fraction(V[j*p_+k][c]) * Fraction(W[k*n_+i][c])
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification with floating point."""
        n_, m_, p_ = n, m, p
        rank = len(U[0])
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m_+j][c] * V[j*p_+k][c] * W[k*n_+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try to find lower rank decompositions
    # We'll try ranks from 62 down to a reasonable minimum
    # For each rank, try random U, V and solve for W

    np.random.seed(42)

    # Try various rank values
    for target_rank in range(62, 20, -1):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break

            # Random U, V with small integer entries
            # Use entries from {-1, 0, 1} to keep things simple
            U_rand = np.random.randint(-1, 2, size=(n * m, target_rank))
            V_rand = np.random.randint(-1, 2, size=(m * p, target_rank))

            # Solve for W: for each (i, k), we have m equations
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for j=0..m-1
            W_sol = np.zeros((p * n, target_rank))
            valid = True

            for i in range(n):
                for k in range(p):
                    # Build the m x target_rank matrix A
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U_rand[i * m + j, c] * V_rand[j * p + k, c]

                    # Solve A @ x = ones(m)
                    try:
                        # Check if system is consistent
                        # Use lstsq
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)

                        # Check residual
                        if len(residuals) > 0:
                            if residuals[0] > 1e-8:
                                valid = False
                                break
                        else:
                            # Underdetermined, check if A @ x is close to ones
                            if np.linalg.norm(A @ x - np.ones(m)) > 1e-8:
                                valid = False
                                break

                        W_sol[k * n + i, :] = x
                    except:
                        valid = False
                        break

                if not valid:
                    break

            if not valid:
                continue

            # Check if W_sol is close to half-integers
            W_2x = W_sol * 2
            W_rounded_2x = np.round(W_2x)
            if np.linalg.norm(W_2x - W_rounded_2x) > 1e-6:
                continue

            W_rounded = W_rounded_2x / 2.0

            # Convert to lists
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rounded.tolist()

            # Fast verify
            if not verify_fast(U_list, V_list, W_list):
                continue

            # Exact verify with fractions
            if verify_exact(U_list, V_list, W_list):
                # Found a valid decomposition!
                return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
