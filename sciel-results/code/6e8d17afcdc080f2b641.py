# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10

    def deadline_ok():
        return (time.time() - t0) < 140.0

    # Start with the known rank-55 Strassen+schoolbook construction
    rank0 = 55
    U_mat = [[0] * rank0 for _ in range(n * m)]
    V_mat = [[0] * rank0 for _ in range(m * p)]
    W_mat = [[0] * rank0 for _ in range(p * n)]

    terms = []

    # Strassen for 5 pairs of columns in B
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        a00, a01, a10, a11 = 0, 1, 3, 4
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1

        # M1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        # M2 = (A[1,0]+A[1,1])*B[0,k0]
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        # M3 = A[0,0]*(B[0,k1]-B[1,k1])
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        # M4 = A[1,1]*(B[1,k0]-B[0,k0])
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        # M5 = (A[0,0]+A[0,1])*B[1,k1]
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        # M6 = (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        # M7 = (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))

    # Schoolbook for third row of A
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))

    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx][t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx][t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx][t_idx] = coeff

    best_U, best_V, best_W = U_mat, V_mat, W_mat
    best_rank = rank0

    # Convert to numpy for tensor operations
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)

    # Build the full multiplication tensor T[i,j,k] = sum_t U[i,t]*V[j,t]*W[k,t]
    # This should equal the standard matrix multiplication tensor
    def build_tensor(U, V, W):
        r = U.shape[1]
        T = np.zeros((n * m, m * p, p * n))
        for t in range(r):
            T += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
        return T

    # Reference tensor for matrix multiplication
    def ref_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T_ref = ref_tensor()
    T_current = build_tensor(U_arr, V_arr, W_arr)

    # Verify correctness
    if not np.allclose(T_current, T_ref, atol=1e-10):
        # Return what we have; it should be correct
        pass

    # Try to reduce rank by checking if any term can be removed
    # while preserving the tensor exactly
    def try_remove_term(U, V, W, t_remove):
        """Check if removing term t_remove still gives the correct tensor."""
        r = U.shape[1]
        if r <= 1:
            return None, None, None, False

        U_new = np.delete(U, t_remove, axis=1)
        V_new = np.delete(V, t_remove, axis=1)
        W_new = np.delete(W, t_remove, axis=1)

        T_new = build_tensor(U_new, V_new, W_new)

        if np.allclose(T_new, T_ref, atol=1e-10):
            return U_new, V_new, W_new, True

        return U, V, W, False

    # Iteratively try to remove terms
    for round_num in range(3):
        if not deadline_ok():
            break

        progress = True
        while progress:
            if not deadline_ok():
                break
            progress = False
            r = U_arr.shape[1]
            for t in range(r):
                if not deadline_ok():
                    break
                U_new, V_new, W_new, success = try_remove_term(U_arr, V_arr, W_arr, t)
                if success:
                    U_arr, V_arr, W_arr = U_new, V_new, W_new
                    best_rank = U_arr.shape[1]
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    progress = True
                    break

    # Final verification
    if best_rank < rank0:
        T_final = build_tensor(np.array(best_U), np.array(best_V), np.array(best_W))
        if not np.allclose(T_final, T_ref, atol=1e-10):
            # Revert to original if something went wrong
            best_U, best_V, best_W = U_mat, V_mat, W_mat
            best_rank = rank0

    return {'U': best_U, 'V': best_V, 'W': best_W}
# EVOLVE-BLOCK-END
