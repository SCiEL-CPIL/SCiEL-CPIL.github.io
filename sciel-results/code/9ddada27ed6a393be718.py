import time
import numpy as np

_DEADLINE = time.time() + 140.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 2, 3, 5

    # Build the rank-28 decomposition via explicit block helpers.
    # Index conventions: U[i*m + j], V[j*p + k], W[k*n + i]
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def schoolbook(i0, i1, j0, j1, k0, k1):
        """Rank-(#i * #j * #k) schoolbook block."""
        nonlocal col
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                for kk in range(k0, k1 + 1):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1

    def strassen(i0, j0, k0):
        """Strassen rank-7 for 2x2x2 sub-block at (i0, j0, k0)."""
        nonlocal col
        # M1 = (A00+A11)*(B00+B11)
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M2 = (A10+A11)*B00
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        # M3 = A00*(B11-B01)
        U[i0 * m + j0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M4 = A11*(B00-B10)
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        # M5 = (A00+A01)*B11
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        # M6 = (A10-A00)*(B00+B01)
        U[(i0 + 1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M7 = (A01-A11)*(B10+B11)
        U[i0 * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    # Block A: j=0, all k -> 2x1x5 schoolbook = rank 10
    schoolbook(0, 1, 0, 0, 0, 4)

    # Block B: j in {1,2}, k in {0,1} -> 2x2x2 Strassen = rank 7
    strassen(0, 1, 0)

    # Block C: j in {1,2}, k=2 -> 2x2x1 schoolbook = rank 4
    schoolbook(0, 1, 1, 2, 2, 2)

    # Block D: j in {1,2}, k in {3,4} -> 2x2x2 Strassen = rank 7
    strassen(0, 1, 3)

    assert col == 28, f"col={col}, expected 28"

    # --- Optional rank-27 probe (guarded by deadline) ---
    # Fix a Kronecker-like sparsity pattern on U, V and solve for W.
    if _time_left() > 2.0:
        try:
            rank27 = 27
            # Sparse pattern: U has entries only where (i*m+j) bit-matches r in a structured way
            # Use a simple structured mask: U[i*m+j, r] nonzero iff (i*m+j + r) % 3 == 0
            # V[j*p+k, r] nonzero iff (j*p+k + r) % 4 == 0
            # This gives a constrained linear system for W.
            U27 = np.zeros((n * m, rank27))
            V27 = np.zeros((m * p, rank27))
            for ii in range(n * m):
                for r in range(rank27):
                    if (ii + r) % 3 == 0:
                        U27[ii, r] = 1.0
            for jj in range(m * p):
                for r in range(rank27):
                    if (jj + r) % 4 == 0:
                        V27[jj, r] = 1.0

            # Build linear system: for each (i,k) output, sum over j of U[i*m+j,r]*V[j*p+k,r] * W[k*n+i,r] = (A*B)[i,k]
            # Use a single test case to get the RHS (this is a necessary condition).
            # Actually we need the system to hold for ALL A, B, so we use the tensor equation.
            # The tensor T[i,j,k] = delta (trivial), so: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
            # This is a trilinear system. With U,V fixed, it's linear in W:
            # For each (i,k): sum_j sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1 for all j
            # Wait, the tensor is T_{ijk} = 1 (the multiplication tensor has all ones for the standard basis).
            # Actually for matrix mult, T_{(i,k),j} ... let me think carefully.
            # The tensor is M_{(i,k), j} where the multiplication tensor has entries:
            # <M, e_i ⊗ e_j ⊗ e_k> = 1 for all i,j,k (since (e_i e_j^T)(e_j e_k^T) ... no)
            # Actually for standard matrix mult: C_{ik} = sum_j A_{ij} B_{jk}
            # The tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
            #
            # With U, V fixed, this is linear in W. For each (i,k) and each j:
            # sum_r [U[i*m+j, r] * V[j*p+k, r]] * W[k*n+i, r] = 1
            #
            # This gives m equations per (i,k) pair, n*p = 10 pairs, so 30 equations in 10*27=270 unknowns.
            # Underdetermined! So we just need to find ANY solution.
            # Let's set up: unknowns are W_flat of size p*n*rank27 = 270.
            # Equations: for each (i,k,j): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1

            A_mat = np.zeros((n * p * m, p * n * rank27))
            b_vec = np.ones(n * p * m)
            eq = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank27):
                            coeff = U27[i * m + j, r] * V27[j * p + k, r]
                            if coeff != 0:
                                A_mat[eq, (k * n + i) * rank27 + r] = coeff
                        eq += 1

            # Solve least-squares
            if _time_left() > 1.0:
                W_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W27 = W_sol.reshape(p * n, rank27)
                # Check if it's an exact solution
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = np.sum(
                                U27[i * m + j, :] * V27[j * p + k, :] * W27[k * n + i, :]
                            )
                if np.allclose(recon, 1.0, atol=1e-8):
                    # Found a rank-27 solution! Convert to list format
                    U_list = U27.tolist()
                    V_list = V27.tolist()
                    W_list = W27.tolist()
                    # Round to nearest half-integer for cleanliness
                    for arr in (U_list, V_list, W_list):
                        for row in arr:
                            for idx in range(len(row)):
                                row[idx] = round(row[idx] * 2) / 2
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass

    # Return the guaranteed rank-28 decomposition
    return {"U": U, "V": V, "W": W}
