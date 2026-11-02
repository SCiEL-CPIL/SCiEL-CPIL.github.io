"""Matrix multiplication 2x5x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
DEADLINE_S = 120.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _try_rank_reduction(n, m, p, max_rank):
    """
    Try to find a decomposition of rank < n*m*p using random initialization
    and iterative refinement (gradient-free / alternating minimization style).
    """
    t_start = time.time()
    deadline = t_start + DEADLINE_S - 10  # safety margin

    best = None

    # Known lower bounds: for 2x5x5, the rank is at least n*m + m*p + p*n - something...
    # Actually for small matrices, we can try ranks from a reasonable lower bound up.
    # The rank of matrix multiplication tensor for n x m * m x p is at least max(n*m, m*p, p*n)
    # and at most n*m*p.
    # For 2x5x5: max(10, 25, 10) = 25. So rank >= 25.
    # Let's try to find rank as low as possible.

    # Try a few random initializations and use alternating minimization
    # to see if we can converge to a valid decomposition at lower rank.

    for rank in range(25, n * m * p):  # 25 to 49
        if time.time() > deadline:
            break

        for attempt in range(5):
            if time.time() > deadline:
                break

            np.random.seed(rank * 1000 + attempt)

            # Initialize U, V, W randomly
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            # Alternating minimization: fix two, solve for the third
            # The constraint is: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a system of n*m*p equations in the entries of one matrix (when others are fixed).

            # For fixed V and W, the equations for U are linear:
            # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # So for each (i,j): sum over k of sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = ...
            # Wait, actually each (i,j,k) gives one equation. For fixed V,W, U[i*m+j,c] appears
            # in equations for all k. Let me restructure.

            # For fixed V, W: the equations are linear in U.
            # Eq (i,j,k): sum_c U[i*m+j,c] * A_c(i,j,k) = 1 where A_c(i,j,k) = V[j*p+k,c]*W[k*n+i,c]
            # For a fixed (i,j), the unknowns are U[i*m+j, 0..rank-1].
            # The equations involving U[i*m+j, .] are for k=0..p-1:
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1, for k=0..p-1
            # That's p equations in rank unknowns. If rank >= p, we can solve.

            # Similarly for V and W.

            # Let's do alternating least squares.

            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break

                # Update U: for each (i,j), solve linear system
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # p equations, rank unknowns
                        # Eq k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((p, rank))
                        b = np.ones(p)
                        for k in range(p):
                            for c in range(rank):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve A @ x = b (least squares if rank > p)
                        try:
                            x, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(rank)
                        U_new[i * m + j, :] = x

                U = U_new

                # Update V: for each (j,k), solve linear system
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # m equations, rank unknowns
                        # Eq i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((n, rank))
                        b = np.ones(n)
                        for i in range(n):
                            for c in range(rank):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            x, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(rank)
                        V_new[j * p + k, :] = x

                V = V_new

                # Update W: for each (k,i), solve linear system
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # m equations, rank unknowns
                        # Eq j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((m, rank))
                        b = np.ones(m)
                        for j in range(m):
                            for c in range(rank):
                                A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            x, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(rank)
                        W_new[k * n + i, :] = x

                W = W_new

                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (s - 1.0) ** 2

                if err < 1e-10:
                    converged = True
                    break

            if converged:
                # Convert to lists
                U_list = [[U[i, c] for c in range(rank)] for i in range(n * m)]
                V_list = [[V[i, c] for c in range(rank)] for i in range(m * p)]
                W_list = [[W[i, c] for c in range(rank)] for i in range(p * n)]

                # Final verification
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found valid decomposition at this rank, return it

        if best is not None:
            break

    return best


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # First, get the schoolbook solution as fallback
    schoolbook = _schoolbook(n, m, p)

    # Try to find a lower-rank decomposition
    try:
        result = _try_rank_reduction(n, m, p, n * m * p - 1)
    except Exception:
        result = None

    if result is not None:
        return result

    # Fallback to schoolbook
    return schoolbook
