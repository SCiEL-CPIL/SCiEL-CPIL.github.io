import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    def verify(U_arr, V_arr, W_arr, r):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        tensor = np.zeros((n, m, p))
        for c in range(r):
            uc = U_arr[:, c].reshape(n, m)
            vc = V_arr[:, c].reshape(m, p)
            wc = W_arr[:, c].reshape(p, n)
            tensor += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        return np.allclose(tensor, 1.0, atol=1e-9)

    def build_rank28():
        """Build the known rank-28 Strassen-like decomposition."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
            U[i1*m+j1, col] = 1
            U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            W[k2*n+i2, col] = 1
            col += 1

            # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
            U[i2*m+j1, col] = 1
            U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1
            W[k2*n+i2, col] = -1
            col += 1

            # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1
            V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1
            W[k2*n+i2, col] = 1
            col += 1

            # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1
            V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1
            W[k1*n+i2, col] = 1
            col += 1

            # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
            U[i1*m+j1, col] = 1
            U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1
            W[k2*n+i1, col] = 1
            col += 1

            # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
            U[i2*m+j1, col] = 1
            U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1
            V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1

            # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
            U[i1*m+j2, col] = 1
            U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        add_schoolbook([0, 1], [0, 1], [4])
        add_schoolbook([0, 1], [2], [0, 1])
        add_schoolbook([0, 1], [2], [2, 3])
        add_schoolbook([0, 1], [2], [4])

        assert col == rank
        return U, V, W, rank

    def build_schoolbook():
        """Build the rank-30 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Start with rank-28 solution
    U, V, W, rank = build_rank28()
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U, V, W, rank = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Try to reduce rank by removing terms
    target = np.ones((n, m, p))

    # Single term removal
    for r in range(rank):
        if time.time() > deadline - 10:
            break

        # Compute contribution of term r
        uc = U[:, r].reshape(n, m)
        vc = V[:, r].reshape(m, p)
        wc = W[:, r].reshape(p, n)
        term_r = np.einsum('ij,jk,ki->ijk', uc, vc, wc)

        # Check if remaining terms sum to target
        # i.e., total - term_r == target
        # total should be target, so we need term_r == 0
        # But that's not the right check. We need to verify the remaining terms.
        remaining = target - term_r
        # Check if remaining can be formed by other terms
        # Actually, since total = target, removing term r gives target - term_r
        # We need target - term_r to be decomposable by the remaining rank-1 tensors
        # This is not straightforward to check without solving

        # Simpler check: if term_r is zero, we can remove it
        if np.allclose(term_r, 0.0, atol=1e-9):
            new_rank = rank - 1
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            if verify(U_new, V_new, W_new, new_rank):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank
                rank = new_rank
                U, V, W = U_new, V_new, W_new
                break

    # Try linear combination approach: can we express one term as a linear
    # combination of others? This requires solving a linear system.
    # For rank r, we have r rank-1 tensors. If they are linearly dependent
    # in the tensor space, we might be able to reduce rank.
    # Flatten all terms into vectors and check rank of the matrix.
    if time.time() < deadline - 15:
        dim = n * m * p
        # Build matrix where each column is a flattened rank-1 tensor
        tensor_matrix = np.zeros((dim, rank))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            tensor_matrix[:, c] = np.einsum('ij,jk,ki->ijk', uc, vc, wc).flatten()

        # Check numerical rank
        try:
            sv = np.linalg.svd(tensor_matrix, full_matrices=False)
            # Count significant singular values
            tol = 1e-8
            num_rank = np.sum(sv > tol)
            if num_rank < rank:
                # There's linear dependence, but exploiting it is complex
                pass
        except np.linalg.LinAlgError:
            pass

    # Try MPS-style factorization approach
    # Model the tensor as a sequence of local interactions
    # For 2x3x5, we can try bond dimensions 1, 2, 3
    if time.time() < deadline - 20:
        # Try to find a lower-rank solution using optimization
        # We'll try rank 24, 25, 26, 27
        for target_rank in [27, 26, 25, 24]:
            if time.time() > deadline - 10:
                break

            # Initialize with random values
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break

                U_try = np.random.randn(n * m, target_rank)
                V_try = np.random.randn(m * p, target_rank)
                W_try = np.random.randn(p * n, target_rank)

                # Optimize using gradient descent on the tensor residual
                # Loss = ||sum_c U_c ⊗ V_c ⊗ W_c - T||^2
                lr = 0.01
                for step in range(200):
                    if time.time() > deadline - 10:
                        break

                    # Compute residual
                    residual = np.zeros((n, m, p))
                    for c in range(target_rank):
                        uc = U_try[:, c].reshape(n, m)
                        vc = V_try[:, c].reshape(m, p)
                        wc = W_try[:, c].reshape(p, n)
                        residual += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
                    residual -= 1.0

                    # Compute gradients
                    grad_U = np.zeros_like(U_try)
                    grad_V = np.zeros_like(V_try)
                    grad_W = np.zeros_like(W_try)

                    for c in range(target_rank):
                        uc = U_try[:, c].reshape(n, m)
                        vc = V_try[:, c].reshape(m, p)
                        wc = W_try[:, c].reshape(p, n)

                        # d/dU_c = sum over k of (residual * wc) contracted
                        # grad_U_c[i,j] = sum_k residual[i,j,k] * wc[k,i]
                        grad_U[:, c] = np.einsum('ijk,ki->ij', residual, wc).flatten()

                        # grad_V_c[j,k] = sum_i residual[i,j,k] * uc[i,j]
                        grad_V[:, c] = np.einsum('ijk,ij->jk', residual, uc).flatten()

                        # grad_W_c[k,i] = sum_j residual[i,j,k] * vc[j,k]
                        grad_W[:, c] = np.einsum('ijk,jk->ki', residual, vc).flatten()

                    # Update
                    U_try -= lr * grad_U
                    V_try -= lr * grad_V
                    W_try -= lr * grad_W

                    # Check convergence
                    if np.linalg.norm(residual) < 1e-6:
                        break

                # Verify solution
                if verify(U_try, V_try, W_try, target_rank):
                    # Round to nearest half-integer
                    U_rounded = np.round(U_try * 2) / 2
                    V_rounded = np.round(V_try * 2) / 2
                    W_rounded = np.round(W_try * 2) / 2

                    if verify(U_rounded, V_rounded, W_rounded, target_rank):
                        best_U, best_V, best_W, best_rank = U_rounded, V_rounded, W_rounded, target_rank
                        break
                    else:
                        # Use the unrounded solution
                        best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                        break

            if best_rank == target_rank:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
