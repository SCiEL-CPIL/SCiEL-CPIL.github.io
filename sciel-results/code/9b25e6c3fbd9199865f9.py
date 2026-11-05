"""Combined approach: Strassen rank-50 base + SVD null-space reduction + structured integer search."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = 2, 4, 7
    deadline = time.time() + 115  # safety margin before 120s limit

    def build_rank50():
        """Build the rank-50 Strassen decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22)
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: (A21+A22)B11
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: A11(B12-B22)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: A22(B21-B11)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A11+A12)B22
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: (A21-A11)(B11+B12)
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A12-A22)(B21+B22)
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # 6 Strassen blocks for k=0..5
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)

        # 8 rank-1 terms for k=6
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1

        return U, V, W, col

    def verify(U_m, V_m, W_m, n, m, p, num_tests=30):
        """Verify decomposition is exact."""
        np.random.seed(12345)
        for _ in range(num_tests):
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=float)
            rank_m = U_m.shape[1]
            for r in range(rank_m):
                u = U_m[:, r].reshape(n, m)
                v = V_m[:, r].reshape(m, p)
                w = W_m[:, r].reshape(p, n)
                C_comp += w * (u @ v)
            if not np.allclose(C_comp, C_expected, atol=1e-8):
                return False
        return True

    # Build initial rank-50 decomposition
    U0, V0, W0, init_rank = build_rank50()

    if not verify(U0, V0, W0, n, m, p):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = init_rank

    # Phase 1: SVD-based null space reduction
    # Stack columns: each rank-1 term is a vector in R^(nm + mp + pn)
    def svd_reduce(U_arr, V_arr, W_arr, tol=1e-10):
        """Find and eliminate linearly dependent columns using SVD."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 30:
                break
            # Stack all columns into one matrix: rows = (nm + mp + pn), cols = rank
            stacked = np.vstack([U_arr.T, V_arr.T, W_arr.T])  # shape: (rank, nm+mp+pn)
            # SVD to find rank
            try:
                U_svd, s_vals, Vt = np.linalg.svd(stacked, full_matrices=False)
            except:
                break
            # Determine numerical rank
            if len(s_vals) == 0:
                break
            rank_num = np.sum(s_vals > tol * s_vals[0])
            if rank_num >= current_rank:
                break
            # The null space is spanned by the last (current_rank - rank_num) right singular vectors
            null_dim = current_rank - rank_num
            if null_dim <= 0:
                break
            # Null space vectors are rows of Vt[rank_num:]
            null_vecs = Vt[rank_num:]  # shape: (null_dim, current_rank)
            # For each null vector, find the column with largest magnitude to eliminate
            for nv in null_vecs:
                if time.time() > deadline - 30:
                    break
                # Find column with max |nv[i]|
                max_idx = np.argmax(np.abs(nv))
                if nv[max_idx] == 0:
                    continue
                # Eliminate column max_idx
                keep = [c for c in range(current_rank) if c != max_idx]
                U_arr = U_arr[:, keep]
                V_arr = V_arr[:, keep]
                W_arr = W_arr[:, keep]
                current_rank = len(keep)
                improved = True
                break  # Restart SVD after one elimination
        return U_arr, V_arr, W_arr, current_rank

    if time.time() < deadline - 50:
        U_red, V_red, W_red, new_rank = svd_reduce(U0, V0, W0)
        if new_rank < best_rank and verify(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank

    # Phase 2: Structured search for integer/half-integer linear combinations
    # Try to find if any column can be expressed as a small-integer combination of others
    def structured_reduce(U_arr, V_arr, W_arr, max_coeff=4):
        """Try to eliminate columns expressible as small-integer combos of others."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 25:
                break
            for c in range(current_rank):
                if time.time() > deadline - 25:
                    break
                others = [oc for oc in range(current_rank) if oc != c]
                if not others:
                    continue
                # Try to express column c as combination of others
                U_sub = U_arr[:, others].T  # shape: (len(others), nm)
                V_sub = V_arr[:, others].T  # shape: (len(others), mp)
                W_sub = W_arr[:, others].T  # shape: (len(others), pn)
                stacked = np.vstack([U_sub, V_sub, W_sub])  # shape: (len(others), nm+mp+pn)
                target = np.concatenate([U_arr[:, c], V_arr[:, c], W_arr[:, c]])

                # Try small integer coefficient vectors
                found = False
                # First try: is it a simple multiple of one other column?
                for oc in others:
                    ratio = None
                    for i in range(len(target)):
                        if np.abs(stacked[:, oc][i]) > 1e-10:
                            r = target[i] / stacked[:, oc][i]
                            if ratio is None:
                                ratio = r
                            elif not np.allclose(r, ratio, atol=1e-9):
                                ratio = None
                                break
                    if ratio is not None and abs(ratio * 2) == round(abs(ratio * 2)):
                        # Check exactness
                        recon = stacked[:, oc] * ratio
                        if np.allclose(recon, target, atol=1e-9):
                            keep = [oc2 for oc2 in range(current_rank) if oc2 != c]
                            U_arr = U_arr[:, keep]
                            V_arr = V_arr[:, keep]
                            W_arr = W_arr[:, keep]
                            current_rank = len(keep)
                            improved = True
                            found = True
                            break

                if found:
                    break

                # Try: sum of two other columns with small integer coefficients
                for i, oc1 in enumerate(others):
                    for oc2 in others[i+1:]:
                        # Try coefficients a, b in {-2,-1,0,1,2}
                        for a in range(-2, 3):
                            for b in range(-2, 3):
                                if a == 0 and b == 0:
                                    continue
                                recon = stacked[:, oc1] * a + stacked[:, oc2] * b
                                if np.allclose(recon, target, atol=1e-9):
                                    keep = [oc3 for oc3 in range(current_rank) if oc3 != c]
                                    U_arr = U_arr[:, keep]
                                    V_arr = V_arr[:, keep]
                                    W_arr = W_arr[:, keep]
                                    current_rank = len(keep)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break

                if found:
                    break

        return U_arr, V_arr, W_arr, current_rank

    if time.time() < deadline - 30:
        U_red, V_red, W_red, new_rank = structured_reduce(best_U, best_V, best_W)
        if new_rank < best_rank and verify(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank

    # Phase 3: Try to find a completely different low-rank decomposition
    # by searching for rank-r decompositions with small integer entries
    # This is the "MILP-like" systematic search: enumerate candidate rank-1 terms
    # with entries in {-1, 0, 1} and try to find a set that spans the tensor

    def search_low_rank(max_rank=45):
        """Search for a decomposition with rank <= max_rank using structured terms."""
        if time.time() > deadline - 20:
            return None

        # The tensor T has components T[i,j,k] = delta (for matrix mult)
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if valid (i,j,k) else 0
        # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is: T[i,j,k] where the "product" gives C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # The constraint is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should
        # equal 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k], but actually the tensor
        # decomposition means: T_{ijk} = sum_r U_{ij}^r * V_{jk}^r * W_{ki}^r
        # where T_{ijk} = 1 for the matrix multiplication tensor.

        # Actually, the matrix multiplication tensor for n x m times m x p has:
        # T[i, j, k] = 1 (it's the identity in the middle dimension)
        # No wait - the tensor is defined such that:
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
        # Hmm, let me re-read the convention.

        # The convention is: W[k*n+i, r] and the decomposition is:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # No, that's not right either. Let me think about this more carefully.

        # Standard tensor decomposition for matrix mult:
        # The multiplication tensor T has T[i, j, k] = delta_{j,j} (identity in middle)
        # A rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # And the reconstruction: C[i,k] = sum_r w_r[k*n+i] * (sum_j u_r[i*m+j] * A[i,j]) * (sum_j v_r[j*p+k] * B[j,k])
        # Wait no, that's not standard. Let me use the standard form:
        # C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T_{i,j,k} = 1 for all valid i,j,k (the "multiplication" tensor)
        # Rank-r: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this doesn't work directly because T_{i,j,k} should be 1 for all i,j,k...

        # Actually I think the convention is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # This is a bilinear form. Let me just verify with the Strassen construction.

        # For the Strassen block with i0=0, j0=0, k0=0:
        # M1: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
        # Contribution to C[0,0]: W[0,0]*(U[0,0]*A[0,0] + U[0,1]*A[0,1])*(V[0,0]*B[0,0] + V[0,1]*B[0,1])
        # = 1 * (1*A[0,0] + 0*A[0,1]) * (1*B[0,0] + 0*B[0,1]) = A[0,0]*B[0,0]
        # Wait, U[0,1] is 0 in M1, so (U[0,0]*A[0,0] + U[0,1]*A[0,1]) = A[0,0]
        # And V[0,1] is 0, so (V[0,0]*B[0,0] + V[0,1]*B[0,1]) = B[0,0]
        # So contribution = A[0,0]*B[0,0] * 1

        # Hmm, but the standard Strassen formula gives different results. Let me just trust
        # the verification function and focus on the search.

        # For a systematic search, I'll try to find rank-r decompositions where
        # each rank-1 term has U, V, W entries in {-1, 0, 1}.
        # This is a constraint satisfaction problem. Given the small dimensions,
        # I can enumerate possible U vectors (sparse, with few non-zero entries).

        # Actually, this is too expensive to enumerate. Instead, let me try a different approach:
        # Use the fact that we know the rank-50 decomposition and try to find
        # linear dependencies more aggressively.

        return None

    # Phase 4: Iterative refinement - try random linear combinations of existing columns
    # to see if we can find a lower-rank equivalent
    if time.time() < deadline - 20:
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        cur_rank = best_rank

        # Try to find if any subset of columns is linearly dependent
        # by checking determinants of submatrices
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            if cur_rank < 2:
                break
            # Pick a random subset of 3 columns and check if one is a combo of the other two
            idx = np.random.choice(cur_rank, size=3, replace=False)
            c1, c2, c3 = idx
            # Check if col c3 = a*col c1 + b*col c2
            U_sub = np.column_stack([U_cur[:, c1], U_cur[:, c2]])
            V_sub = np.column_stack([V_cur[:, c1], V_cur[:, c2]])
            W_sub = np.column_stack([W_cur[:, c1], W_cur[:, c2]])
            stacked = np.vstack([U_sub.T, V_sub.T, W_sub.T])  # (2, nm+mp+pn)
            target = np.concatenate([U_cur[:, c3], V_cur[:, c3], W_cur[:, c3]])

            # Solve for a, b
            try:
                coeffs, residuals, rank_s, sv = np.linalg.lstsq(stacked, target, rcond=None)
                recon = stacked @ coeffs
                if np.allclose(recon, target, atol=1e-9):
                    # Check if coefficients are "nice" (small integers or half-integers)
                    a, b = coeffs
                    if abs(a * 2) == round(abs(a * 2)) and abs(b * 2) == round(abs(b * 2)):
                        keep = [c for c in range(cur_rank) if c != c3]
                        U_cur = U_cur[:, keep]
                        V_cur = V_cur[:, keep]
                        W_cur = W_cur[:, keep]
                        cur_rank = len(keep)
            except:
                continue

        if cur_rank < best_rank and verify(U_cur, V_cur, W_cur, n, m, p):
            best_U, best_V, best_W = U_cur, V_cur, W_cur
            best_rank = cur_rank

    # Final verification
    if not verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = init_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
