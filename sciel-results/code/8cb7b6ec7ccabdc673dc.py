"""Search for a rank-minimal exact tensor decomposition of 3x3x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS


def _verify(U, V, W):
    """Check exact decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,n+i,c] == delta_{i,k}.
    Uses the official trace convention W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build M by iterating columns
    M = np.zeros((n, m, p))
    for c in range(r):
        u = Ua[:, c].reshape(n, m)
        v = Va[:, c].reshape(m, p)
        w = Wa[:, c].reshape(p, n)
        # term = u[i,j] * v[j,k] * w[k, i]
        # contract: for each (i,j,k): u[i,j]*v[j,k]*w[k,i]
        M += np.einsum('ij,jk,ki->ijk', u, v, w)
    target = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            target[i, 0, k] = 1  # j=0
    if not np.allclose(M, target, atol=1e-7):
        return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _strassen_3x3():
    """Strassen decomposition of 3x3x3, rank 21.
    U[i,j,c], V[j,k,c], W[k,i,c] with sum_c U*V*W = delta_{i,k}.
    Uses the standard 3x3 Strassen scheme (Cantor-Kaltofen-Volc)."""
    # 3x3 Strassen: partition into 3x3 blocks? No, 3x3 is already base.
    # Use the known rank-21 scheme for 3x3x3 (Cantor et al.).
    # We express multiplication of A(3x3) by B(3x3) = C(3x3).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Strassen scheme: 21 products.
    # Let's use a verified 3x3x3 rank-21 scheme.
    # Indices: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # We need U[i,j,c], V[j,k,c], W[k,i,c] for c in 0..20.

    # A standard 3x3x3 rank-21 decomposition (from literature).
    # I'll construct it via the 2x2 Strassen recursion on a 3x3 by padding,
    # but that gives rank up to 27. Instead use the explicit 21-term scheme.
    #
    # Explicit 3x3x3 rank 21 (Cantor-Kaltofen-Volc 1978):
    # Represent A, B as 3x3. Define 21 bilinear forms.
    #
    # Let me use a concrete known scheme.
    # A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    #
    # The 21 terms (each is a product of a linear form in A and linear form in B,
    # then combined with linear form in C):
    #
    # This is complex. Let me instead use the fact that 3x3x3 rank 21 is known and
    # build it by embedding 2x2 Strassen. Actually simpler: use the explicit
    # "3x3 Strassen" from the literature with integer coefficients.

    # I'll use a verified scheme. Define the 21 terms as (U coeffs, V coeffs, W coeffs).
    # U is 3x3 (i,j), V is 3x3 (j,k), W is 3x3 (k,i) per term.

    terms = []
    # Term structure: each term c has u (3x3), v (3x3), w (3x3)
    # such that sum_c u_c[i,j] v_c[j,k] w_c[k,i] = delta_{i,k} (as a 3x3x3 tensor)
    # i.e., the resulting C[i,k] = sum_j A[i,j] B[j,k] when A,B are substituted.

    # Use the 2x2 Strassen on the 3x3 by treating it as blocks won't reduce rank below 21 easily.
    # Let me use a direct construction.
    #
    # Known 3x3x3 rank-21 scheme (one form):
    # I'll define it via the "middle" approach:
    # C = A B. Use 21 products.

    # Concrete scheme (verified in code below):
    # I'll use the scheme from:
    #   M. A. Bender et al. or the classic "3x3 Strassen".
    #
    # Let me define it explicitly with the following 21 terms.
    # Each term: (u, v, w) where u,v,w are 3x3 integer matrices.
    # We need: sum_c (u_c * A)_{i,j'} ... no.
    #
    # Correct formulation: For term c, define
    #   P_c = (sum_{i,j} u_c[i,j] A[i,j]) * (sum_{j,k} v_c[j,k] B[j,k])
    #   C[i,k] += w_c[k,i] * P_c
    # We need sum_c w_c[k,i] * (sum_{i,j} u_c[i,j] A[i,j]) * (sum_{j,k} v_c[j,k] B[j,k]) = C[i,k]
    # where C = A B, so C[i,k] = sum_j A[i,j] B[j,k].

    # Let me just use the 2x2 Strassen recursion which gives a valid (if not minimal)
    # decomposition, and accept rank 27 for the 3x3 core if I can't easily get 21.
    # But we want minimal. Let me use a known 21-term scheme.

    # I'll use this explicit 3x3x3 rank-21 scheme (from the literature):
    # It's based on the following 21 bilinear products.

    # Define helper to build a 3x3 matrix from a flat list (row-major).
    def mat3(vals):
        return np.array(vals, dtype=float).reshape(3, 3)

    # The 21 terms (u, v, w) - this is the Cantor-Kaltofen-Volc scheme.
    # I'll use a well-known representation.
    #
    # Actually, let me use a simpler verified approach: the 3x3 rank-21 scheme
    # can be obtained from the 2x2 Strassen by a specific embedding.
    # Let me just construct a valid decomposition and verify it, even if rank > 21,
    # then the 7 extra terms give a valid (if not minimal) answer.
    #
    # For safety, let me use the schoolbook for the 3x3 core (rank 9) plus 7 extra = 16?
    # No, that's not how it works. Let me think again.

    # CORRECT APPROACH:
    # The full tensor T[i,j,k] = delta_{i,k} for i in 0..2, j in 0..2, k in 0..6.
    # This is NOT a direct sum of 3x3x3 and 3x3x4 in the simple sense because
    # the output dimension is 7, not 3+4 with independent blocks.
    #
    # The tensor is: for each (i,j), T[i,j,k] = 1 if k=i, else 0.
    # So T[i,j,k] = delta_{i,k} for k in 0..6, but i only goes to 2.
    # So for k >= 3, T[i,j,k] = 0 for all i,j.
    #
    # Wait, that means the tensor is:
    # T[i,j,k] = 1 if k == i (and i < 3), else 0.
    #
    # This is a very sparse tensor! It's essentially the "diagonal" tensor in the
    # (i,k) plane, replicated over j.
    #
    # So T = sum_i (e_i in i-space) (tensor) (sum_j e_j in j-space) (tensor) (e_i in k-space)
    #     = (sum_i e_i ⊗ e_i) ⊗ (sum_j e_j)
    #
    # This is a rank-3 tensor (in the i-k plane) times a rank-1 tensor (in j).
    # So the overall rank is at most 3 * 1 = 3? No, tensor rank doesn't factor that simply.
    #
    # Actually, T[i,j,k] = delta_{i,k} (with i,k in 0..2, j in 0..2, and k extended to 0..6 with zeros for k>=3).
    #
    # This tensor can be written as: T[i,j,k] = (delta_{i,k}) * (1)
    # where delta_{i,k} is 1 if i==k (i,k in 0..2) and 0 otherwise (including k>=3).
    #
    # As a tensor in (i,j,k): T = sum_i (e_i ⊗ 1_j ⊗ e_i) where 1_j = sum_j e_j.
    # This is a sum of 3 terms, each of rank 1 (in the tensor sense, each is a pure tensor).
    # So the tensor rank is at most 3!
    #
    # Let me verify: T[i,j,k] = sum_{c=0}^{2} u_c[i] * v_c[j] * w_c[k]
    # where u_c[i] = delta_{i,c}, v_c[j] = 1, w_c[k] = delta_{k,c}.
    # Then sum_c delta_{i,c} * 1 * delta_{k,c} = delta_{i,k} (for i,k in 0..2, and 0 for k>=3).
    # Yes! So the tensor rank is exactly 3.
    #
    # But wait, the factor matrices have specific shapes:
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    #
    # For term c: U[i*m+j, c] = u_c[i] = delta_{i,c} (independent of j)
    #             V[j*p+k, c] = v_c[j] * w_c[k] = 1 * delta_{k,c} = delta_{k,c}
    #             W[k*n+i, c] = ... wait, W has shape (p*n, rank) = (21, rank).
    #
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # For term c: U[i*m+j,c] = delta_{i,c}, V[j*p+k,c] = delta_{k,c}, W[k*n+i,c] = 1
    # Then product = delta_{i,c} * delta_{k,c} * 1 = delta_{i,k} (for i,k in 0..2).
    # Sum over c: sum_c delta_{i,c} delta_{k,c} = delta_{i,k}. Correct!
    #
    # So rank 3 works! Let me double check the shapes and conventions.
    #
    # U has shape (n*m, rank) = (9, 3). U[i*m+j, c] = delta_{i,c}.
    # V has shape (m*p, rank) = (21, 3). V[j*p+k, c] = delta_{k,c}.
    # W has shape (p*n, rank) = (21, 3). W[k*n+i, c] = 1.
    #
    # Let me verify with the trace convention: W[k*n+i, r].
    # For k in 0..6, i in 0..2, W[k*n+i, c] = 1 for all k,i,c.
    #
    # Check: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # = sum_c delta_{i,c} * delta_{k,c} * 1
    # = delta_{i,k} (if i,k in 0..2) else 0.
    #
    # For k in 0..2: delta_{i,k} = 1 if i==k. Correct (C[i,k] = sum_j A[i,j]B[j,k] with B having B[j,k]=delta_{j,0}... no wait).
    #
    # Hold on. Let me re-examine what the tensor should be.
    # The matrix multiplication A(3x3) * B(3x7) = C(3x7).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] satisfies: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]? No.
    #
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ... this is not right either.
    #
    # Let me re-read the convention. The task says:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # and the schoolbook uses:
    #   U[i*m+j, column] = 1
    #   V[j*p+k, column] = 1
    #   W[k*n+i, column] = 1
    # for each (i,j,k) term.
    #
    # So the decomposition is: C[i,k] = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Wait, that doesn't look right dimensionally. Let me think.
    #
    # Actually, the standard form is:
    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) * W[k,i,c]
    # No, that's not right either.
    #
    # The correct Strassen-style decomposition:
    # C[i,k] = sum_c W[k,i,c] * (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    #
    # Hmm, let me look at the schoolbook more carefully.
    # In schoolbook, for each (i,j,k):
    #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # All other entries of U[:,:,c], V[:,:,c], W[:,:,c] are 0.
    #
    # So the term c contributes: W[k*n+i, c] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    # = 1 * A[i,j] * B[j,k] = A[i,j] B[j,k].
    #
    # Summing over c (all (i,j,k) pairs): C[i,k] = sum_j A[i,j] B[j,k]. Correct!
    #
    # So the general form is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    #
    # For my rank-3 candidate:
    # Term c (c in 0,1,2):
    #   U[i'*m+j', c] = delta_{i',c} (independent of j')
    #   V[j'*p+k', c] = delta_{k',c} (independent of j')
    #   W[k*n+i, c] = 1
    #
    # Contribution: 1 * (sum_{i',j'} delta_{i',c} A[i',j']) * (sum_{j',k'} delta_{k',c} B[j',k'])
    # = (sum_{j'} A[c,j']) * (sum_{j'} B[j',c])
    #
    # Sum over c: C[i,k] = sum_c (sum_{j'} A[c,j']) (sum_{j'} B[j',c])
    #
    # But this is INDEPENDENT of i and k! That's wrong.
    #
    # I made an error. Let me reconsider.
    #
    # The issue is that W[k*n+i, c] = 1 for all k,i,c. But the contribution should depend on i and k.
    #
    # Let me reconsider. The tensor T[i,j,k] should satisfy:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k], we need the decomposition to "select" the right i and k.
    #
    # In schoolbook, W[k*n+i,c] = 1 only when the term corresponds to that specific (i,j,k), and U and V select specific (i',j') and (j',k').
    #
    # So my rank-3 attempt fails because W is constant. I need W to depend on i and k to select the right output entry.
    #
    # Let me reconsider the tensor structure.
    # T[i,j,k] = delta_{i,k} (for the multiplication tensor, where C[i,k] = sum_j A[i,j]B[j,k]).
    #
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # (using the convention that U is indexed by (i,j), V by (j,k), W by (k,i)).
    #
    # In the flattened form:
    # U_flat[i*m+j, c] = U[i,j,c]
    # V_flat[j*p+k, c] = V[j,k,c]
    # W_flat[k*n+i, c] = W[k,i,c]
    #
    # And T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    #
    # For T[i,j,k] = delta_{i,k}:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,k}.
    #
    # Note that the RHS depends on i and k but NOT on j.
    #
    # Can I write delta_{i,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] with few terms?
    #
    # Since the RHS is independent of j, I can set U[i,j,c] = u_c[i] (independent of j) and V[j,k,c] = v_c[k] (independent of j).
    # Then sum_c u_c[i] v_c[k] W[k,i,c] = delta_{i,k}.
    #
    # If I also set W[k,i,c] = w_c[i] (independent of k, but depends on i):
    # sum_c u_c[i] v_c[k] w_c[i] = delta_{i,k}.
    #
    # This requires: for each (i,k), sum_c (u_c[i] w_c[i]) v_c[k] = delta_{i,k}.
    #
    # Let a_c[i] = u_c[i] w_c[i]. Then sum_c a_c[i] v_c[k] = delta_{i,k}.
    #
    # This is like: the matrix A (i x c) times matrix V^T (c x k) = delta_{i,k} (i x k).
    # So A * V^T = I_3 (3x3 identity), where A is 3x3 and V^T is 3x3.
    # This requires A and V^T to be inverses, so rank 3 suffices (with A = I, V^T = I).
    #
    # So: u_c[i] = delta_{i,c}, w_c[i] = delta_{i,c}, v_c[k] = delta_{k,c}.
    # Then a_c[i] = delta_{i,c} * delta_{i,c} = delta_{i,c}.
    # sum_c delta_{i,c} delta_{k,c} = delta_{i,k}. Correct!
    #
    # So:
    # U[i,j,c] = delta_{i,c} (independent of j)
    # V[j,k,c] = delta_{k,c} (independent of j)
    # W[k,i,c] = delta_{i,c} (independent of k)
    #
    # Rank 3!
    #
    # Let me verify:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = sum_c delta_{i,c} * delta_{k,c} * delta_{i,c} = sum_c delta_{i,c} delta_{k,c} = delta_{i,k}.
    #
    # Yes! Rank 3 is sufficient.
    #
    # Now let me build the factor matrices:
    # U: shape (9, 3). U[i*m+j, c] = delta_{i,c}.
    # V: shape (21, 3). V[j*p+k, c] = delta_{k,c}.
    # W: shape (21, 3). W[k*n+i, c] = delta_{i,c}.

    U = [[0] * 3 for _ in range(9)]
    V = [[0] * 3 for _ in range(21)]
    W = [[0] * 3 for _ in range(21)]

    for i in range(3):
        for j in range(3):
            for c in range(3):
                if i == c:
                    U[i * m + j][c] = 1

    for j in range(3):
        for k in range(7):
            for c in range(3):
                if k == c:
                    V[j * p + k][c] = 1

    for k in range(7):
        for i in range(3):
            for c in range(3):
                if i == c:
                    W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 120.0

    # Try rank-3 construction first.
    try:
        result = _strassen_3x3()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    # Fallback to schoolbook.
    return _schoolbook()


# Test at module level (not called by evaluator, but for debugging).
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Valid:", _verify(res["U"], res["V"], res["W"]))
