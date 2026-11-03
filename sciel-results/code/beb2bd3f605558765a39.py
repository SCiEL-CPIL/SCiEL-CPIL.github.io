import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def compute_full_tensor(U, V, W):
        """Compute T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] for all a,b,c."""
        r = U.shape[1]
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r_idx in range(r):
            u_col = U[:, r_idx]
            v_col = V[:, r_idx]
            w_col = W[:, r_idx]
            T += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        return T

    def compute_on_diagonal_error(U, V, W):
        """Compute error on the (i,j,k) = 1 terms."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += (val - 1.0) ** 2
        return err

    def compute_off_diagonal_error(U, V, W, sample_size=500):
        """Compute error on random off-diagonal terms."""
        err = 0.0
        count = 0
        for _ in range(sample_size):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this is an on-diagonal term
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue  # Skip on-diagonal
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err += val * val
            count += 1
        return err / max(count, 1)

    def is_valid_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        # Check all on-diagonal terms equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check all off-diagonal terms equal 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank

    # Try to find a lower-rank decomposition using ALS
    # The key challenge: we need both on-diagonal = 1 AND off-diagonal = 0

    # Strategy: Use ALS with combined objective
    # For each rank R, try multiple random initializations

    lower_bound = max(n * p, 27)  # Heuristic lower bound

    for R in range(trivial_rank - 1, max(lower_bound - 1, 20), -1):
        if time.time() > deadline - 15:
            break

        # Try multiple random initializations
        for trial in range(8):
            if time.time() > deadline - 15:
                break

            np.random.seed(trial * 1000 + R)

            # Random initialization with small values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1

            # ALS iterations
            converged = False
            prev_err = float('inf')

            for iteration in range(80):
                if time.time() > deadline - 10:
                    break

                # Update U: for each (i,j), solve linear system
                # For fixed (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: A @ x = ones(p) where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass

                # Update V: for each (j,k), solve linear system
                # For fixed (j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # This is: B @ x = ones(n) where B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass

                # Update W: for each (k,i), solve linear system
                # For fixed (k,i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # This is: C @ x = ones(m) where C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass

                # Compute error
                on_err = compute_on_diagonal_error(U, V, W)
                off_err = compute_off_diagonal_error(U, V, W, sample_size=300)
                total_err = on_err + 10 * off_err  # Weight off-diagonal more

                if abs(prev_err - total_err) < 1e-12:
                    converged = True
                    break
                prev_err = total_err

            # Check if we have a valid solution
            if converged or compute_on_diagonal_error(U, V, W) < 1e-8:
                # Round to half-integers
                U_r, V_r, W_r = round_to_half_integers(U, V, W)

                # Verify exactness
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-10):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break  # Found valid for this rank, try lower
                else:
                    # Try without rounding (maybe the solution is already rational)
                    if is_valid_decomposition(U, V, W, tol=1e-10):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = R
                        break

        if time.time() > deadline - 10:
            break

    # Final verification of the best solution found
    # If we found a better rank, use it; otherwise return schoolbook
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }
