"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses linear dependency peeling on Strassen-based decomposition to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Starts with Strassen-based decomposition (rank 78) and applies linear
    dependency peeling to find and remove redundant rank-1 terms.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def build_strassen_decomp():
        """Build the Strassen-based decomposition (rank = 78)."""
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)

            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val

            U_vecs.append(u)
            V_vecs.append(v)
            W_vecs.append(w)

        def strassen_222_local(i0, j0, k0):
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])

            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])

            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])

            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])

            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])

            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])

            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])

        def m227_decomp(i0, j0, k0):
            strassen_222_local(i0, j0, k0)
            strassen_222_local(i0, j0, k0 + 2)
            strassen_222_local(i0, j0, k0 + 4)
            k6 = k0 + 6
            for i in range(2):
                for j in range(2):
                    ii = i0 + i
                    jj = j0 + j
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+k6], [1.0],
                                   [k6*n+ii], [1.0])

        def m127_decomp(i0, j0, k0):
            for j in range(2):
                for k in range(7):
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([i0*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+i0], [1.0])

        def m327_decomp(i0, j0, k0):
            m227_decomp(i0, j0, k0)
            m127_decomp(i0 + 2, j0, k0)

        # m347: split m=4 as 2+2
        m327_decomp(0, 0, 0)
        m327_decomp(0, 2, 0)

        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    def build_schoolbook():
        """Build the schoolbook decomposition (rank = n*m*p = 84)."""
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_vecs.append(u)
                    V_vecs.append(v)
                    W_vecs.append(w)
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    def peel_linear_dependencies(U, V, W, max_time=90.0):
        """
        Iteratively find and remove rank-1 terms that are linear combinations
        of others. A term r is redundant if there exist coefficients alpha_s
        (s != r) such that for all (i,j,k):
            U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] =
            sum_{s != r} alpha_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        
        This is equivalent to: the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        lies in the span of {U[:,s] ⊗ V[:,s] ⊗ W[:,s] : s != r} when viewed
        in the appropriate tensor space restricted to the support pattern.
        
        We use an iterative approach: try removing each column and check if
        the remaining decomposition still reproduces the tensor.
        """
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        rank = U.shape[1]
        start_peel = time.time()

        # Precompute the target tensor values
        # T[i,j,k] = 1 for all (i,j,k) in the matrix multiplication tensor
        # Actually, T[i,j,k] = 1 for all valid (i,j,k) since it's the
        # multiplication tensor where each entry is 1.
        
        def compute_tensor_value(i, j, k, U_mat, V_mat, W_mat):
            """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
            return np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])

        def quick_verify(U_mat, V_mat, W_mat, sample_frac=0.3, tol=1e-8):
            """Quick verification on a random subset of tensor entries."""
            rng = np.random.RandomState(42)
            # Test a subset of (i,j,k) triples
            n_test = max(20, int(0.3 * n * m * p))
            indices = rng.choice(n * m * p, size=n_test, replace=False)
            for idx in indices:
                k = idx % p
                ij = idx // p
                j = ij % m
                i = ij // m
                val = compute_tensor_value(i, j, k, U_mat, V_mat, W_mat)
                if abs(val - 1.0) > tol:
                    return False
            return True

        def full_verify(U_mat, V_mat, W_mat, tol=1e-8):
            """Full verification of all tensor entries."""
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = compute_tensor_value(i, j, k, U_mat, V_mat, W_mat)
                        if abs(val - 1.0) > tol:
                            return False
            return True

        # Iterative peeling: repeatedly try to remove one term
        while time.time() - start_peel < max_time and time.time() - start_time < deadline - 15.0:
            rank = U.shape[1]
            if rank <= 1:
                break
            
            found_removal = False
            
            # Try removing each column (from last to first for stability)
            for r in range(rank - 1, -1, -1):
                if time.time() - start_peel > max_time or time.time() - start_time > deadline - 15.0:
                    break
                
                # Remove column r
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                # Quick verification first
                if quick_verify(U_test, V_test, W_test, sample_frac=0.5, tol=1e-7):
                    # Full verification
                    if full_verify(U_test, V_test, W_test, tol=1e-8):
                        U = U_test
                        V = V_test
                        W = W_test
                        found_removal = True
                        break
            
            if not found_removal:
                # No more removable terms via simple deletion
                break
        
        return U, V, W, U.shape[1]

    # Build initial Strassen decomposition
    U, V, W = build_strassen_decomp()
    current_rank = U.shape[1]

    # Verify initial decomposition
    if not verify_decomp(U, V, W, tol=1e-8):
        # Fallback to schoolbook if Strassen is wrong
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    # Apply linear dependency peeling
    U, V, W, current_rank = peel_linear_dependencies(U, V, W, max_time=90.0)

    # Final verification
    if not verify_decomp(U, V, W, tol=1e-8):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
