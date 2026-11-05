import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 115.0

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 25
    P = p * n  # 20
    deadline = _START + _DEADLINE

    # Build tensor T[i,k,j] = 1
    # Tensor shape: (n, m, p) = (4, 5, 5)
    T = np.ones((n, m, p), dtype=np.float64)

    def term_tensor(u, v, w):
        """Compute the rank-1 tensor from u (N,), v (M,), w (P,)."""
        U = u.reshape(n, m)
        V = v.reshape(m, p)
        W = w.reshape(p, n)
        return np.einsum('ik,kj,ji->ijk', U, V, W)

    def tensor_diff(a, b):
        return a - b

    # Initialize with schoolbook
    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def add_term(i, k, j, cu=1.0, cv=1.0, cw=1.0):
        u = np.zeros(N)
        v = np.zeros(M)
        w = np.zeros(P)
        u[i * m + k] = cu
        v[k * p + j] = cv
        w[j * n + i] = cw
        add_rank1(u, v, w)

    # Schoolbook: 100 terms
    for i in range(n):
        for k in range(m):
            for j in range(p):
                add_term(i, k, j)

    # Verify initial
    current_tensor = np.zeros((n, m, p), dtype=np.float64)
    for idx in range(len(U_cols)):
        current_tensor += term_tensor(U_cols[idx], V_cols[idx], W_cols[idx])
    assert np.allclose(current_tensor, T, atol=1e-10), "Initial schoolbook verification failed"

    # Sparsification: try to merge pairs of rank-1 terms
    # A pair (a, b) can be merged if there exist alpha, beta (half-integers)
    # such that alpha*u_a + beta*u_b, alpha*v_a + beta*v_b, alpha*w_a + beta*w_b
    # form a valid rank-1 term (i.e., the tensor they produce equals
    # alpha*T_a + beta*T_b, and the result is a single rank-1 tensor).
    #
    # Actually, the key insight: two rank-1 tensors T_a and T_b can be combined
    # into ONE rank-1 tensor iff their sum is rank-1.
    # T_a + T_b = u_a v_a w_a^T + u_b v_b w_b^T is rank-1 iff
    # u_a, u_b are linearly dependent AND v_a, v_b are linearly dependent AND
    # w_a, w_b are linearly dependent (or appropriate degeneracies).
    #
    # More precisely: for the sum of two rank-1 tensors to be rank-1, we need
    # the combined u, v, w to still be rank-1 in the tensor sense.
    #
    # A simpler approach: try replacing pairs (i,j) with a single term by
    # solving for a new rank-1 term that equals T_i + T_j.
    # This is possible iff T_i + T_j has tensor rank 1.
    #
    # Even better: use Strassen-like identities. For the 2x2x2 case:
    # (a+b)(c+d) = ac + ad + bc + bd can be done in 3 multiplications.
    # Generalize: look for groups of rank-1 terms that can be replaced.

    # Let's try a direct approach: for each pair, check if their sum is rank-1
    def is_rank1_tensor(t):
        """Check if a 3D tensor of shape (n,m,p) has tensor rank 1."""
        # A tensor T has rank 1 iff T[i,k,j] = a[i]*b[k]*c[j] for some vectors
        # This is equivalent to: all 2x2 minors of any "slice" are zero
        # Simple check: flatten and check
        # For a rank-1 tensor: T[i1,k1,j1]*T[i2,k2,j2] == T[i1,k2,j1]*T[i2,k1,j2] etc.
        # More efficiently: check that T = outer(u, outer(v, w)) for some u,v,w
        # A necessary and sufficient condition:
        # Fix j=0: T[:,:,0] must be rank 1 (as a matrix)
        # And T[:,:,j] = T[:,:,0] * c[j] / c[0] for all j (if c[0] != 0)
        # Handle edge cases carefully.

        # Check if T is zero
        if np.allclose(t, 0, atol=1e-12):
            return True

        # Find a nonzero entry
        nz = np.nonzero(t)
        if len(nz[0]) == 0:
            return True

        # Try: T[i,k,j] = a[i]*b[k]*c[j]
        # From T[i,k,0] = a[i]*b[k]*c[0], if c[0] != 0:
        # T[i,k,j] / T[i,k,0] = c[j]/c[0] (should be independent of i,k)
        # Let's check using the first nonzero slice

        # Get a reference
        # Try to factor: T = U @ V @ W^T (in tensor sense)
        # Use the fact that for rank-1: T[i,k,j] = A[i,k] * C[j] where A is rank-1 matrix
        # OR T[i,k,j] = B[k,j] * D[i] where B is rank-1 matrix
        # OR T[i,k,j] = E[i,j] * F[k] where E is rank-1 matrix

        # Check via SVD of slices
        # T[:,:,j] for each j should be scalar multiples of a rank-1 matrix

        # Pick the first j where T[:,:,j] is not all zero
        j_ref = None
        for j in range(p):
            if np.any(np.abs(t[:, :, j]) > 1e-12):
                j_ref = j
                break
        if j_ref is None:
            return True

        ref_slice = t[:, :, j_ref]
        # Check ref_slice is rank 1
        sv = np.linalg.svd(ref_slice, compute_uv=False)
        if len(sv) > 1 and sv[1] > 1e-10 * (sv[0] if sv[0] > 0 else 1):
            return False

        # Now check all other slices are multiples
        for j in range(p):
            if j == j_ref:
                continue
            sl = t[:, :, j]
            if np.allclose(sl, 0, atol=1e-12):
                continue
            # sl should be a scalar multiple of ref_slice
            # Find the scalar
            mask = np.abs(ref_slice) > 1e-12
            if not np.any(mask):
                continue
            ratios = sl[mask] / ref_slice[mask]
            if not np.allclose(ratios, ratios[0], atol=1e-10, rtol=1e-8):
                return False

        return True

    # Try to merge pairs
    # For efficiency, we work with a list of terms and try to reduce
    terms = list(zip(U_cols, V_cols, W_cols))
    current_tensor = T.copy()

    merged = True
    while merged:
        if time.time() > deadline - 10:
            break
        merged = False
        n_terms = len(terms)
        for i in range(n_terms):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n_terms):
                # Check if terms[i] + terms[j] is rank-1
                combined = term_tensor(terms[i][0], terms[i][1], terms[i][2]) + \
                           term_tensor(terms[j][0], terms[j][1], terms[j][2])
                if is_rank1_tensor(combined):
                    # Find the rank-1 decomposition of combined
                    # combined[i,k,j] = a[i]*b[k]*c[j]
                    # Extract: pick nonzero entries
                    # a[i] = combined[i, k0, j0] / (b[k0]*c[j0])
                    # This is tricky; let's use a simpler extraction
                    nz = np.nonzero(combined)
                    if len(nz[0]) == 0:
                        # Zero tensor - just remove both
                        terms.pop(j)
                        terms.pop(i)
                        merged = True
                        break
                    # Extract rank-1 factors
                    # Pick i0, k0, j0 where combined is nonzero
                    i0, k0, j0 = nz[0][0], nz[1][0], nz[2][0]
                    val0 = combined[i0, k0, j0]
                    # a[i] * b[k] * c[j] = combined[i,k,j]
                    # Set c[j0] = 1, then a[i]*b[k] = combined[i,k,j0]
                    # And a[i0]*b[k] = combined[i0,k,j0]
                    # So b[k] = combined[i0,k,j0] / a[i0]
                    # And a[i] = combined[i,k0,j0] / b[k0]
                    # We need to be careful. Let's set:
                    # c[j] = combined[i0,k0,j] / val0
                    # Then a[i]*b[k] = combined[i,k,j] / c[j] (for j where c[j]!=0)
                    # Use j0: a[i]*b[k] = combined[i,k,j0] / c[j0] = combined[i,k,j0]
                    # (since c[j0] = 1 by construction... wait, c[j0] = val0/val0 = 1)
                    c_vec = np.zeros(p)
                    c_vec[j0] = 1.0
                    for j in range(p):
                        if abs(combined[i0, k0, j]) > 1e-14:
                            c_vec[j] = combined[i0, k0, j] / val0
                    # Now a[i]*b[k] = combined[i,k,j] / c[j] for any j with c[j] != 0
                    # Use j0: a[i]*b[k] = combined[i,k,j0]
                    # a[i] = combined[i,k0,j0] / b[k0]
                    # b[k] = combined[i0,k,j0] / a[i0]
                    # a[i0] * b[k0] = combined[i0,k0,j0] = val0
                    # Set a[i0] = 1, b[k0] = val0
                    # Then a[i] = combined[i,k0,j0] / val0
                    # b[k] = combined[i0,k,j0] / 1 = combined[i0,k,j0]
                    a_vec = np.zeros(n)
                    b_vec = np.zeros(m)
                    a_vec[i0] = 1.0
                    b_vec[k0] = val0
                    for i in range(n):
                        a_vec[i] = combined[i, k0, j0] / val0
                    for k in range(m):
                        b_vec[k] = combined[i0, k, j0] / 1.0  # a[i0]=1
                    # Verify
                    # Actually let me just verify the decomposition
                    recon = np.einsum('i,k,j->ijk', a_vec, b_vec, c_vec)
                    if not np.allclose(recon, combined, atol=1e-10):
                        continue
                    # Build the new term
                    u_new = np.zeros(N)
                    v_new = np.zeros(M)
                    w_new = np.zeros(P)
                    for i in range(n):
                        for k in range(m):
                            u_new[i * m + k] = a_vec[i] * b_vec[k]
                    # Wait, this isn't right. The rank-1 term has:
                    # U[i,k] = a[i]*b[k] (for the U factor)
                    # V[k,j] = ... hmm, I'm confusing myself.
                    #
                    # Let me reconsider. A rank-1 tensor T[i,k,j] = U[i,k]*V[k,j]*W[j,i]
                    # where U is n x m, V is m x p, W is p x n.
                    # For T to be rank-1 in the tensor sense: T[i,k,j] = x[i]*y[k]*z[j]
                    # Then U[i,k]*V[k,j]*W[j,i] = x[i]*y[k]*z[j] for all i,k,j.
                    # One valid factorization: U[i,k] = x[i]*y[k], V[k,j] = z[j] (independent of k), W[j,i] = 1
                    # But V must depend on k. Actually: U[i,k]=x[i], V[k,j]=y[k]*z[j], W[j,i]=1
                    # Or: U[i,k]=x[i]*y[k], V[k,j]=z[j], W[j,i]=1
                    # The last one: V[k,j] = z[j] for all k. That's valid.
                    # Or: U[i,k]=x[i], V[k,j]=y[k]*z[j], W[j,i]=1
                    # Let's use: U[i,k]=x[i]*y[k], V[k,j]=z[j], W[j,i]=1
                    # But V[k,j]=z[j] means V doesn't depend on k, which is fine.
                    # Actually, let's use: U[i,k]=x[i], V[k,j]=y[k]*z[j], W[j,i]=1
                    # This gives T[i,k,j] = x[i]*y[k]*z[j]*1 = x[i]*y[k]*z[j]. Correct.
                    # But we need integer/half-integer entries. x,y,z might not be.
                    # Let's just use the direct factorization.
                    u_new = np.zeros(N)
                    v_new = np.zeros(M)
                    w_new = np.zeros(P)
                    for i in range(n):
                        u_new[i * m + k0] = a_vec[i]  # U[i,k0] = a[i], U[i,k]=0 for k!=k0
                    for k in range(m):
                        for j in range(p):
                            v_new[k * p + j] = b_vec[k] * c_vec[j]  # V[k,j] = b[k]*c[j]
                    for i in range(n):
                        for j in range(p):
                            w_new[j * n + i] = 1.0  # W[j,i] = 1
                    # Verify this gives the right tensor
                    recon2 = term_tensor(u_new, v_new, w_new)
                    if not np.allclose(recon2, combined, atol=1e-10):
                        # Try another factorization
                        u_new2 = np.zeros(N)
                        v_new2 = np.zeros(M)
                        w_new2 = np.zeros(P)
                        for i in range(n):
                            for k in range(m):
                                u_new2[i * m + k] = a_vec[i] * b_vec[k]
                        for j in range(p):
                            for k in range(m):
                                v_new2[k * p + j] = c_vec[j]
                        for i in range(n):
                            for j in range(p):
                                w_new2[j * n + i] = 1.0
                        recon3 = term_tensor(u_new2, v_new2, w_new2)
                        if np.allclose(recon3, combined, atol=1e-10):
                            u_new, v_new, w_new = u_new2, v_new2, w_new2
                        else:
                            continue
                    # Replace terms i and j with the new term
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((u_new, v_new, w_new))
                    merged = True
                    break
            if merged:
                break

    # Final verification
    U_cols = [t[0] for t in terms]
    V_cols = [t[1] for t in terms]
    W_cols = [t[2] for t in terms]

    final_tensor = np.zeros((n, m, p), dtype=np.float64)
    for idx in range(len(U_cols)):
        final_tensor += term_tensor(U_cols[idx], V_cols[idx], W_cols[idx])
    if not np.allclose(final_tensor, T, atol=1e-10):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term(i, k, j)

    rank = len(U_cols)
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
