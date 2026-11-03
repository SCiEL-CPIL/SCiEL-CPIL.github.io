import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 140

    def make_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, A, B, tol=1e-9):
        """Check that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = (A@B)[i,k] for all i,j,k."""
        C = A @ B
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for t in range(r):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(total - C[i,k]) > tol:
                    return False
        return True

    def compress_rank(U, V, W, max_iter=200):
        """
        Try to reduce rank by finding linear dependencies among columns.
        We work over rationals to preserve exactness.
        Strategy: iteratively check if any column can be expressed as a rational
        linear combination of others in all three matrices simultaneously.
        """
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        r = U.shape[1]

        # Use rational arithmetic for precision
        U_r = np.array([[Fraction(int(round(x*1e9)), 10**9) if abs(x) > 1e-15 else Fraction(0,1) for x in row] for row in U])
        V_r = np.array([[Fraction(int(round(x*1e9)), 10**9) if abs(x) > 1e-15 else Fraction(0,1) for x in row] for row in V])
        W_r = np.array([[Fraction(int(round(x*1e9)), 10**9) if abs(x) > 1e-15 else Fraction(0,1) for x in row] for row in W])

        # Try to find columns that are linear combinations of others
        # For each column t, check if it's in the span of the other columns
        # across all three matrices
        removed = set()
        for _iter in range(max_iter):
            if time.time() > deadline - 5:
                break
            progress = False
            for t in range(r):
                if t in removed:
                    continue
                # Build the matrix of other columns (concatenated U, V, W columns)
                other_cols = [c for c in range(r) if c != t and c not in removed]
                if not other_cols:
                    break
                # Check if column t is in span of other columns
                # Stack: for each row index, we have a value in U, V, W
                # We need: exists coefficients c_s such that for all rows:
                # U[row, t] = sum_s c_s * U[row, s]
                # V[row, t] = sum_s c_s * V[row, s]
                # W[row, t] = sum_s c_s * W[row, s]
                # This is a linear system in c_s.
                n_eq = U_dim + V_dim + W_dim
                n_unknowns = len(other_cols)
                # Build augmented system: A_sys * c = b
                A_sys = np.zeros((n_eq, n_unknowns))
                b = np.zeros(n_eq)
                for idx, s in enumerate(other_cols):
                    for u_row in range(U_dim):
                        A_sys[u_row, idx] = float(U_r[u_row, s])
                    for v_row in range(V_dim):
                        A_sys[U_dim + v_row, idx] = float(V_r[v_row, s])
                    for w_row in range(W_dim):
                        A_sys[U_dim + V_dim + w_row, idx] = float(W_r[w_row, s])
                for u_row in range(U_dim):
                    b[u_row] = float(U_r[u_row, t])
                for v_row in range(V_dim):
                    b[U_dim + v_row] = float(V_r[v_row, t])
                for w_row in range(W_dim):
                    b[U_dim + V_dim + w_row] = float(W_r[w_row, t])

                # Solve least squares
                try:
                    coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=1e-10)
                except Exception:
                    continue

                # Check if solution is exact (residuals near zero)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # Verify with rational check
                    valid = True
                    for u_row in range(U_dim):
                        total = sum(coeffs[idx] * float(U_r[u_row, other_cols[idx]]) for idx in range(n_unknowns))
                        if abs(total - float(U_r[u_row, t])) > 1e-7:
                            valid = False
                            break
                    if valid:
                        for v_row in range(V_dim):
                            total = sum(coeffs[idx] * float(V_r[v_row, other_cols[idx]]) for idx in range(n_unknowns))
                            if abs(total - float(V_r[v_row, t])) > 1e-7:
                                valid = False
                                break
                    if valid:
                        for w_row in range(W_dim):
                            total = sum(coeffs[idx] * float(W_r[w_row, other_cols[idx]]) for idx in range(n_unknowns))
                            if abs(total - float(W_r[w_row, t])) > 1e-7:
                                valid = False
                                break
                    if valid:
                        removed.add(t)
                        progress = True
                        break  # restart scanning
            if not progress:
                break

        if not removed:
            return U, V, W, r

        # Remove columns
        keep = [c for c in range(r) if c not in removed]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, len(keep)

    def make_kron_decomposition():
        """
        Build a Kronecker-product based decomposition.
        For 2x4 * 4x8:
        A is 2x4, B is 4x8, C is 2x8.
        
        We can write this as: C[i,k] = sum_j A[i,j] * B[j,k]
        
        Using the fact that 4 = 2*2 and 8 = 2*4, we can use a mixed approach.
        Let's use the structure: A[i, j1*2+j2] where j1 in {0,1}, j2 in {0,1}
        B[j1*2+j2, k1*4+k2] where k1 in {0,1}, k2 in {0,1,2,3}
        
        Actually, let's use a simpler Kron approach:
        Decompose using 2x2 Strassen on the "inner" 2 dimension and 
        schoolbook on the outer.
        
        Alternative: Use the full tensor and find a rank decomposition via
        the Kronecker product of smaller decompositions.
        
        For 2x4*4x8, we can view it as (2x2 * 2x4) for each pair of columns of B.
        But let's try: A is 2x4, B is 4x8.
        
        Write A = [A1 | A2] where A1, A2 are 2x2.
        Write B = [B1; B2] where B1, B2 are 2x8.
        C = A1*B1 + A2*B2.
        
        Each A1*B1 is 2x2 * 2x8, and A2*B2 is 2x2 * 2x8.
        
        For 2x2 * 2x8: use Strassen on the 2x2 part.
        A1*B1: C1[i,k] = sum_j A1[i,j]*B1[j,k]
        Using Strassen: 7 mults per 2x2 block of B1, and B1 has 4 such blocks (8 cols / 2).
        So 7*4 = 28 mults for A1*B1.
        Similarly 28 for A2*B2.
        Total: 56. Same as before.
        
        But wait - we can do better by sharing Strassen intermediates!
        
        Actually, let's try a different decomposition:
        2x4 * 4x8 = (2x2) * (2x8) + (2x2) * (2x8)  [splitting the 4 dimension]
        
        For each 2x2 * 2x8 product, use Strassen: 7 * 4 = 28.
        Total 56.
        
        To beat 56, we need a smarter approach.
        
        Let's try: 2x4 * 4x8.
        Split differently: 2x4 * 4x8 = (2x4) * (4x2) * (2x4) [no, that doesn't work]
        
        Alternative: Use the fact that rank(2x4*4x8) <= rank(2x2*2x2) * something.
        
        Actually, let me just try the direct approach: build the full 64-term schoolbook,
        then apply rank compression.
        """
        # Start with schoolbook rank 64
        U, V, W, r = make_schoolbook()
        
        # Now try to compress
        U_c, V_c, W_c, r_c = compress_rank(U, V, W, max_iter=100)
        
        return U_c, V_c, W_c, r_c

    def make_hybrid_strassen():
        """
        Hybrid approach: use Strassen for the 2x2 inner blocks but share
        computations across the 4 column blocks of B.
        
        A is 2x4: A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
        B is 4x8.
        
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        Group j into pairs: (j0, j1) = (0,1) and (j2, j3) = (2,3)
        
        For group 0: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
        For group 1: C[i,k] += A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        Each group is a 2x2 * 2x8 product.
        Using Strassen on each: 7 mults per 2x2 block of the 2x8, so 7*4=28 per group.
        Total: 56.
        
        But can we share? The Strassen intermediates for group 0 and group 1
        are different because they use different A entries and B rows.
        
        However, we can use a different grouping. What if we split the 8 columns
        of B into 4 groups of 2, and for each group use a 2x4 * 4x2 product?
        That's the same as before.
        
        Let me try yet another approach: use the rank of the tensor directly.
        The tensor for 2x4*4x8 has dimensions 8x32x16.
        Its rank is known to be at most 56 (from Strassen tiling) and at least...
        
        Actually, for small matrices, the rank is often less than the Strassen tiling.
        Let me try to find a rank < 56 decomposition by solving a system of equations.
        """
        # This would be computationally expensive, so let's stick with the compression approach
        pass

    # Strategy: Try multiple approaches and pick the best
    best_result = None
    best_rank = float('inf')

    # Approach 1: Schoolbook + compression
    if time.time() < deadline - 10:
        U1, V1, W1, r1 = make_kron_decomposition()
        # Verify
        rng = np.random.RandomState(42)
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        if verify(U1, V1, W1, A, B):
            if r1 < best_rank:
                best_rank = r1
                best_result = (U1, V1, W1)

    # Approach 2: Strassen tiling (known good, rank 56)
    U2, V2, W2 = None, None, None
    if time.time() < deadline - 10:
        # Rebuild Strassen tiling
        def make_strassen_block():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
            U[2,1]=1; U[3,1]=1; V[0,1]=1
            U[0,2]=1; V[1,2]=1; V[3,2]=-1
            U[3,3]=1; V[2,3]=1; V[0,3]=-1
            U[0,4]=1; U[2,4]=1; V[3,4]=1
            U[1,5]=1; U[3,5]=-1; V[0,5]=1; V[2,5]=1
            U[2,6]=1; U[0,6]=-1; V[1,6]=1; V[3,6]=1
            W[0,0]=1; W[0,4]=-1; W[0,3]=1; W[0,6]=1
            W[1,2]=1; W[1,4]=1
            W[2,1]=1; W[2,3]=1
            W[3,0]=1; W[3,2]=-1; W[3,5]=1; W[3,1]=-1
            return U, V, W

        U_block, V_block, W_block = make_strassen_block()
        rank_total = 8 * 7
        U_full = np.zeros((U_dim, rank_total))
        V_full = np.zeros((V_dim, rank_total))
        W_full = np.zeros((W_dim, rank_total))
        col_offset = 0
        for a_block in range(2):
            a_col_base = a_block * 2
            b_row_base = a_block * 2
            for c_block in range(4):
                c_col_base = c_block * 2
                for t in range(7):
                    for i in range(2):
                        for j in range(2):
                            U_full[i*4 + a_col_base + j, col_offset] += U_block[i*2+j, t]
                    for j in range(2):
                        for k in range(2):
                            V_full[(b_row_base+j)*8 + c_col_base + k, col_offset] += V_block[j*2+k, t]
                    for i in range(2):
                        for k in range(2):
                            W_full[(c_col_base+k)*2 + i, col_offset] += W_block[k*2+i, t]
                    col_offset += 1
        U2, V2, W2 = U_full, V_full, W_full
        r2 = rank_total
        # Verify
        rng = np.random.RandomState(42)
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        if verify(U2, V2, W2, A, B):
            if r2 < best_rank:
                best_rank = r2
                best_result = (U2, V2, W2)

    # Approach 3: Try to compress the Strassen tiling further
    if best_result is not None and time.time() < deadline - 10:
        U3, V3, W3, r3 = compress_rank(best_result[0], best_result[1], best_result[2], max_iter=50)
        rng = np.random.RandomState(42)
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        if verify(U3, V3, W3, A, B):
            if r3 < best_rank:
                best_rank = r3
                best_result = (U3, V3, W3)

    # Fallback: schoolbook
    if best_result is None:
        U0, V0, W0, r0 = make_schoolbook()
        best_result = (U0, V0, W0)

    U_final, V_final, W_final = best_result
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
