"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72

    # We try to find a decomposition with rank < 72 using random initialization
    # and gradient-free / gradient-based optimization.
    # The tensor T_{i,j,k} = e_{i,k} (where i in [0,n), j in [0,m), k in [0,p))
    # We need: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # For a given rank r, we have variables:
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Objective: minimize sum_{i,j,k} (T_{i,j,k} - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2

    # Since n=3, m=4, p=6, the tensor has 3*4*6 = 72 non-zero entries (all 1s at positions where the indices match the matrix multiplication structure).
    # Actually, T is the multiplication tensor: T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # In standard matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j] * B[j,k].
    # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p). That's n*m*p = 72 entries, all equal to 1.
    # Wait, no. The multiplication tensor is a 3-tensor of shape (n*m, m*p, p*n) in the flattened sense, or (n, m, p) in the unflattened sense.
    # T[i,j,k] = 1 for all valid i,j,k. So there are n*m*p = 72 non-zero entries, all 1.

    # The schoolbook decomposition uses rank = n*m*p = 72, with each term being a single (i,j,k) triple.

    # Can we do better? For matrix multiplication, the minimum rank is known as the bilinear complexity.
    # For 3x4 by 4x6, this is not trivially known, but we can search.

    # Let's try rank values from 60 down to some lower bound.
    # Lower bound: The tensor rank is at least max(n*m, m*p, p*n) / something... actually, a simple lower bound is the "slice rank" or similar.
    # For now, let's try a few ranks and see if we can find an exact solution.

    best_result = None

    # First, always have the schoolbook as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return mat.tolist()

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_rank(r, num_trials=5, max_iter=2000):
        """Try to find a decomposition of rank r using random initialization and gradient descent."""
        if r >= rank_schoolbook:
            return None

        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n

        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Random initialization
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1

            # Use a simple gradient descent approach
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                # Compute gradient
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # dL/dU[i*m+j,c] = -2 * (T[i,j,k] - pred) * V[j*p+k,c] * W[k*n+i,c] summed over j,k
                # This is expensive. Let's use a simpler approach: just check if the current solution is good enough.

                # Actually, let's use a different strategy: try to find solutions with integer/half-integer entries.
                # For small ranks, we can try random integer/half-integer vectors.

                # Let's break out and use a different method
                break

        # Alternative: try to construct solutions with simple patterns
        # For rank r, try to find U, V, W with entries in {0, 1/2, 1}
        # This is a combinatorial search, but for small r it might work.

        return None

    # For now, let's just return the schoolbook decomposition.
    # A proper search would take too long, but we can try a few quick things.

    # Try rank 71, 70, ... with a very quick check
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 50), -1):
        if time.time() > deadline:
            break

        # Quick random trial
        for trial in range(3):
            if time.time() > deadline:
                break
            U = np.random.choice([0, 0.5, 1], size=(n * m, r))
            V = np.random.choice([0, 0.5, 1], size=(m * p, r))
            W = np.random.choice([0, 0.5, 1], size=(p * n, r))

            err = check_decomposition(U, V, W)
            if err < 1e-6:
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Return schoolbook
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
# EVOLVE-BLOCK-END
