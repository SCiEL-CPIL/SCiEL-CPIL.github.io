import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Fallback schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, r):
        """
        Verify that the decomposition satisfies the matrix multiplication tensor.
        The tensor T[i,j,k] should be 1 if and only if the element (i,k) of C = A@B
        is computed from A[i,:] and B[:,k].
        
        In the standard tensor decomposition for matrix multiplication:
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        For this to equal the matrix multiplication tensor, we need:
        T[i,j,k] = 1 for all valid (i,j,k) since each (i,j,k) corresponds to 
        one multiplication in the dot product.
        
        Wait, actually the matrix multiplication tensor has T[i,j,k] = 1 for all
        (i,j,k) in the range, because each entry C[i,k] = sum_j A[i,j] * B[j,k],
        and each term A[i,j]*B[j,k] contributes to C[i,k].
        
        So the tensor is indeed the all-ones tensor of shape (n, m, p).
        """
        T_comp = np.zeros((n, m, p))
        for c in range(r):
            u_col = np.array(U)[:, c]
            v_col = np.array(V)[:, c]
            w_col = np.array(W)[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        T_comp[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
        return np.allclose(T_comp, 1.0, atol=1e-9)

    def verify_strassen(U, V, W, r):
        """
        Alternative verification: check if the decomposition correctly computes
        C = A @ B for random matrices.
        """
        np.random.seed(42)
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(r):
                u_col = np.array(U)[:, c]
                v_col = np.array(V)[:, c]
                w_col = np.array(W)[:, c]
                # Reconstruct A, B, C components
                A_part = np.zeros((n, m))
                B_part = np.zeros((m, p))
                C_part = np.zeros((n, p))
                for i in range(n):
                    for j in range(m):
                        A_part[i, j] = u_col[i * m + j]
                for j in range(m):
                    for k in range(p):
                        B_part[j, k] = v_col[j * p + k]
                for i in range(n):
                    for k in range(p):
                        C_part[i, k] = w_col[k * n + i]
                C_computed += A_part @ B_part * C_part
            if not np.allclose(C_computed, C_expected, atol=1e-6):
                return False
        return True

    def try_find_decomposition(r, num_tries=200):
        """
        Try to find a rank-r decomposition.
        Strategy: Use the structure that the tensor is all-ones.
        We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
        
        One approach: set U and V to have simple patterns, then solve for W.
        """
        for attempt in range(num_tries):
            if time.time() > deadline:
                return None

            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))

            # Solve for W using the constraint that T[i,j,k] = 1
            W_sol = np.zeros((p * n, r))
            valid = True

            for i in range(n):
                for k in range(p):
                    # We need: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # This is: sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]

                    # We need A_ik @ w = ones(m) where w = W[k*n+i, :]
                    try:
                        w_vec, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        if np.linalg.norm(A_ik @ w_vec - np.ones(m)) > 1e-9:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w_vec
                    except:
                        valid = False
                        break

            if not valid:
                continue

            # Check if W is close to half-integers
            W_snap = np.round(W_sol * 2) / 2
            if verify(U, V, W_snap, r):
                return U, V, W_snap

        return None

    # Try to find a better decomposition
    # The theoretical minimum rank for 3x3x5 is unknown, but schoolbook is 45.
    # Known results: rank(3x3x3) <= 23 (Strassen-like), so rank(3x3x5) <= 23 + 2*3*2 = 35?
    # Actually, we can extend Strassen to larger dimensions.
    
    # First, try to find any improvement over schoolbook
    for r in range(44, 39, -1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r, num_tries=50)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # If no improvement found, return schoolbook
    return get_schoolbook()
