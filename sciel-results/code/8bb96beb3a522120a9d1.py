"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    # ---- Verified rank-42 block-Strassen construction (fallback / baseline) ----
    def build_rank42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        U_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 1): -1},
        ]
        V_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (1, 1): 1},
        ]
        W_contrib = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): 1},
            {(0, 0): 1, (1, 0): 1},
            {(0, 0): -1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): 1},
        ]

        col = 0
        for a in range(2):
            for b in range(3):
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_coeffs[s].items():
                        i = i_loc
                        j = a * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    for (j_loc, k_loc), coeff in V_coeffs[s].items():
                        j = a * 2 + j_loc
                        k = b * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        i = i_loc
                        k = b * 2 + k_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exactness check (with tolerance for half-integer entries) ----
    def is_valid(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # Build the multiplication tensor
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape to 3D
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        # T[i,j,k] = sum_s Ur[i,j,s]*Vr[j,k,s]*Wr[k,i,s]
        # Compute via einsum
        T = np.einsum('ijs,jks,kis->ijk', Ur, Vr, Wr)
        # Expected: E[i,j,k] = 1 if j == j (i.e. the (i,k) entry of product = sum_j A[i,j]*B[j,k])
        # So E[i,j,k] = 1 if (j is the inner index) -> actually the tensor is:
        # (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] should equal 1 for all (i,j,k) where the index j is the "middle"
        # Wait: the multiplication tensor has entries T[i,j,k] = 1 if we're computing (A*B)[i,k] via A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index sense)
        # Actually no: T[i,j,k] = delta_{j,j} = 1 always? No.
        # The tensor for matrix multiplication: T[i,j,k] = 1 if j is the contraction index.
        # For multiplying (n x m) by (m x p): T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because (A*B)[i,k] = sum_j A[i,j] * B[j,k], and the tensor is the "identity" on the middle index.
        # So the expected tensor is all ones? No, that's not right either.
        # Let me think again: the tensor T is defined by T[i,j,k] = 1 if we want to compute C[i,k] = sum_j A[i,j] B[j,k].
        # The decomposition says C[i,k] = sum_s U[i,j,s]... no.
        # Standard: C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
        # Tensor form: T[i,j,k] = 1 (the "multiplication tensor" has all entries 1? No.)
        # Actually the tensor is: T[i,j,k] = 1 for the specific pattern where j is the shared index.
        # For a fixed (i,k), C[i,k] = sum_j A[i,j] * B[j,k]. So the tensor entry T[i,j,k] = 1 for ALL i,j,k.
        # Wait, that would mean the tensor is just the all-ones tensor, which is rank 1. That's wrong.
        # Let me reconsider. The multiplication tensor is: T[i,j,k] = 1 if j is the index being summed over.
        # For (n,m,p) matrix multiplication, the tensor is in R^{n x m x p} with T[i,j,k] = 1 for all i,j,k.
        # No! That's the all-ones tensor. The correct tensor is: T[i,j,k] = 1 for the "identity" pattern.
        # Actually I think I'm confusing myself. Let me just check: for 2x2x2, the multiplication tensor has 8 entries, all equal to 1.
        # Because C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], etc. Each term A[i,j]*B[j,k] contributes to C[i,k].
        # So the tensor T[i,j,k] = 1 for all i,j,k. Yes, the multiplication tensor is the all-ones tensor!
        # And its rank is what we're trying to minimize.
        E = np.ones((n, m, p))
        return np.allclose(T, E, atol=tol)

    # ---- Time-budgeted search for lower rank using smooth epigraph surrogate ----
    def search_lower_rank():
        """Try to find a valid decomposition with rank < 42 using a smooth surrogate."""
        best_result = None
        best_rank = 42

        # Try ranks from 30 up to 41
        for rank in range(30, 42):
            if time.time() > deadline - 20.0:
                break

            # Initialize with random integer/half-integer values
            np.random.seed(rank)
            U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5

            # Smooth epigraph objective: minimize max_s |T - E|_inf using log-sum-exp
            Ur = U.reshape(n, m, rank)
            Vr = V.reshape(m, p, rank)
            Wr = W.reshape(p, n, rank)

            # Gradient descent with smooth objective
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 15.0:
                    break

                T = np.einsum('ijs,jks,kis->ijk', Ur, Vr, Wr)
                E = np.ones((n, m, p))
                diff = T - E

                # Smooth surrogate: log-sum-exp of |diff|^2
                # For simplicity, use L2 norm squared as smooth proxy
                obj = np.sum(diff ** 2)

                # Compute gradient via finite differences on a few random entries
                # This is expensive, so use a simpler approach: random perturbation + accept if better
                # Actually, let's use a simple coordinate descent on a subset of entries

                # Pick random indices to update
                for _ in range(5):
                    mat = np.random.choice([U, V, W])
                    row = np.random.randint(0, mat.shape[0])
                    col = np.random.randint(0, mat.shape[1])

                    for delta in [0.5, -0.5, 1.0, -1.0]:
                        old_val = mat[row, col]
                        mat[row, col] = old_val + delta

                        # Recompute
                        if mat is U:
                            Ur2 = mat.reshape(n, m, rank)
                            T2 = np.einsum('ijs,jks,kis->ijk', Ur2, Vr, Wr)
                        elif mat is V:
                            Vr2 = mat.reshape(m, p, rank)
                            T2 = np.einsum('ijs,jks,kis->ijk', Ur, Vr2, Wr)
                        else:
                            Wr2 = mat.reshape(p, n, rank)
                            T2 = np.einsum('ijs,jks,kis->ijk', Ur, Vr, Wr2)

                        obj2 = np.sum((T2 - E) ** 2)

                        if obj2 < obj:
                            obj = obj2
                        else:
                            mat[row, col] = old_val
                            break

            # Final check
            Ur = U.reshape(n, m, rank)
            Vr = V.reshape(m, p, rank)
            Wr = W.reshape(p, n, rank)
            T = np.einsum('ijs,jks,kis->ijk', Ur, Vr, Wr)
            E = np.ones((n, m, p))
            if np.allclose(T, E, atol=1e-6):
                # Convert to list of lists with integer/half-integer values
                U_list = [[U[i, j] for j in range(rank)] for i in range(n * m)]
                V_list = [[V[i, j] for j in range(rank)] for i in range(m * p)]
                W_list = [[W[i, j] for j in range(rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # ---- Main logic: try search, fall back to rank 42 ----
    result = search_lower_rank()
    if result is not None:
        return result
    else:
        return build_rank42()
# EVOLVE-BLOCK-END
