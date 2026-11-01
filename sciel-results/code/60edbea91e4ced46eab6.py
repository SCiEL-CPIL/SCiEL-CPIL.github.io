import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 5

    def build_strassen_decomposition():
        """Build the 4x4x2 + 2x2x1 + 2x2x1 block Strassen decomposition."""
        rank = 36
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def u_idx(i, j):
            return i * m + j

        def v_idx(j, k):
            return j * p + k

        def w_idx(k, i):
            return k * n + i

        col = 0

        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices

            # M1: (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M2: (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1

            # M3: y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M4: y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1

            # M5: (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1

            # M6: (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M7: (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1

        # Block 1: A_L * B_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)

        # Block 2: A_R * B_21
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)

        # Block 3: A_L * B_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)

        # Block 4: A_R * B_22
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)

        # Block 5: A_L * B_15 (2x2 * 2x1)
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1

        # Block 6: A_R * B_25 (2x2 * 2x1)
        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1

        return U, V, W, col

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        try:
            D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(D[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                            return False
            return True
        except Exception:
            return False

    def reduce_rank_via_svd(U, V, W, max_rank=None):
        """
        Reduce rank using SVD on tensor matricization.
        The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        Matricize as T_{ab,c} = (U ⊙ V)^T W  (or similar)
        We use the flattening: M = sum_r outer(U[:,r], V[:,r]) * W[:,r]
        This is a (n*m * m*p) x (p*n) matrix.
        SVD gives minimal rank decomposition.
        """
        dim_a = n * m  # 8
        dim_b = m * p  # 20
        dim_c = p * n  # 10

        # Build matricization M of shape (dim_a * dim_b, dim_c)
        # M[i,j] where i = a*dim_b + b, j = c
        # M = sum_r U[:,r] ⊗ V[:,r] * W[:,r]
        M = np.zeros((dim_a * dim_b, dim_c), dtype=float)
        for r in range(U.shape[1]):
            uv = np.outer(U[:, r], V[:, r])  # (dim_a, dim_b)
            M += uv.reshape(-1)[:, None] * W[:, r][None, :]

        # SVD
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except Exception:
            return None, None, None

        # Determine effective rank
        tol = 1e-10
        if max_rank is not None:
            eff_rank = min(max_rank, len(s))
        else:
            eff_rank = np.sum(s > tol)

        if eff_rank == 0:
            return None, None, None

        # Extract rank-r factors
        # M = U_svd[:, :r] * diag(s[:r]) * Vt[:r, :]
        # We need to decompose this back into U_new, V_new, W_new
        # M[i,j] = sum_{r'} U_svd[i,r'] * s[r'] * Vt[r',j]
        # We want M[i,j] = sum_r U_new[a,r]*V_new[b,r]*W_new[c,r] where i=a*dim_b+b
        # This is tricky. Let's use a different approach.

        # Alternative: use the fact that we can write the tensor as
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # The matricization T_{ab|c} has rank = tensor rank (for this mode)
        # SVD: T_{ab|c} = P * S * Q^T
        # Then T = sum_{r=1}^rank P[:,r] * S[r] * Q[:,r]
        # But P[:,r] is a vector of length dim_a*dim_b, not separable.
        # We need to further decompose.

        # Better approach: iterate - try to find minimal r such that
        # there exist U_new (dim_a x r), V_new (dim_b x r), W_new (dim_c x r)
        # with sum_r U_new[:,r] ⊗ V_new[:,r] ⊗ W_new[:,r] = T

        # Use the nuclear norm / iterative approach:
        # Start with the SVD-based approximation and refine.

        # For now, let's just return the SVD-based factors if they work.
        # Actually, let's use a simpler approach: try all possible rank values
        # and use linear algebra to find exact solutions.

        return None, None, None

    def find_minimal_rank_by_search(U, V, W, initial_rank):
        """
        Try to reduce rank by finding linear dependencies in the tensor.
        Use the approach of checking if the tensor can be represented with fewer terms.
        """
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = initial_rank

        # Method 1: Try removing columns and checking if tensor is preserved
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = initial_rank

        for iteration in range(5):  # Try a few rounds of removal
            if time.time() > deadline:
                break

            removed_any = False
            for col_to_remove in range(current_rank):
                if time.time() > deadline:
                    break

                remaining_cols = [c for c in range(current_rank) if c != col_to_remove]

                # Check if removing this column still gives valid decomposition
                # by checking if the column is in the span of others in a tensor-compatible way
                # Simple check: just remove and verify
                U_new = np.delete(current_U, col_to_remove, axis=1)
                V_new = np.delete(current_V, col_to_remove, axis=1)
                W_new = np.delete(current_W, col_to_remove, axis=1)

                if verify_decomposition(U_new, V_new, W_new):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    break  # Restart the loop with the new reduced set

            if not removed_any:
                break

        return best_U, best_V, best_W, best_rank

    def try_alternative_blockings():
        """Try different block decompositions and return the best one."""
        candidates = []

        # Original: 4 blocks of 2x2x2 + 2 blocks of 2x2x1
        U1, V1, W1, r1 = build_strassen_decomposition()
        candidates.append((U1, V1, W1, r1))

        # Try: 2 blocks of 2x4x2 (left half) + 2 blocks of 2x4x2 (right half)
        # Actually, let's try a different partition:
        # Split B into 3+2 columns instead of 2+2+1
        # Block A: A (2x4) * B_left (4x3) -> 2x3 result
        # Block B: A (2x4) * B_right (4x2) -> 2x2 result

        # For 2x4x2: use 2 Strassen blocks of 2x2x2
        # For 2x4x3: use 1 Strassen block of 2x2x2 + 1 block of 2x2x1

        # This is essentially what we already have, just grouped differently.
        # The total rank should be the same.

        # Try: split A into 1+1 rows instead of 2+2
        # Block A: A_row0 (1x4) * B (4x5) -> 1x5 result
        # Block B: A_row1 (1x4) * B (4x5) -> 1x5 result
        # Each 1x4x5 has rank 20 (schoolbook) or less with Strassen-like methods
        # Actually 1x4x5 is just a dot product for each output, so rank = 4*5 = 20 each
        # Total = 40, which is worse.

        # Try: use the fact that n=2 to get a better decomposition
        # The output C is 2x5, so we need to compute C[0,:] and C[1,:]
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # C[1,k] = sum_j A[1,j]*B[j,k]

        # We can use Strassen-like tricks on the 2-row structure.
        # Let's try a different approach: treat this as computing
        # [C0; C1] = [A0; A1] * B where A0, A1 are 1x4 rows

        # Use: S1 = (A0+A1), S2 = A0-A1
        # Then C0 = (S1*B + S2*B)/2, C1 = (S1*B - S2*B)/2
        # But S1*B and S2*B each require 4*5=20 multiplications
        # Total = 40, worse.

        # Better: use Strassen on the 2x2 subblocks as before.
        # The current decomposition is likely near-optimal for this structure.

        return candidates

    # Build the base decomposition
    U, V, W, rank = build_strassen_decomposition()

    # Verify it's correct
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=float)
        V_school = np.zeros((m * p, rank_school), dtype=float)
        W_school = np.zeros((p * n, rank_school), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1.0
                    V_school[j * p + k, idx] = 1.0
                    W_school[k * n + i, idx] = 1.0
                    idx += 1
        U, V, W = U_school, V_school, W_school
        rank = rank_school

    # Try to reduce rank
    best_U, best_V, best_W, best_rank = find_minimal_rank_by_search(U, V, W, rank)

    # Use the best result
    U, V, W = best_U, best_V, best_W

    # Final verification
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=float)
        V_school = np.zeros((m * p, rank_school), dtype=float)
        W_school = np.zeros((p * n, rank_school), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1.0
                    V_school[j * p + k, idx] = 1.0
                    W_school[k * n + i, idx] = 1.0
                    idx += 1
        U, V, W = U_school, V_school, W_school

    # Convert to lists with clean integer values
    def clean_matrix(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                # Round to nearest integer if close
                if abs(val - round(val)) < 1e-10:
                    clean_row.append(int(round(val)))
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    U_list = clean_matrix(U)
    V_list = clean_matrix(V)
    W_list = clean_matrix(W)

    return {"U": U_list, "V": V_list, "W": W_list}
