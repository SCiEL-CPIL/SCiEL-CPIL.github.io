"""Search for a rank-minimal exact tensor decomposition of 3x3 * 3x5 matmul."""
import time
import random
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 3, 5]
N, M, P = DIMENSIONS

# Index maps: A[i,m+j], B[j,p+k], C[k*n+i]
def aidx(i, j):
    return i * M + j

def bidx(j, k):
    return j * P + k

def cidx(k, i):
    return k * N + i

def build_T(r):
    """Return the target multiplication tensor as (n*m, m*p, p*n)."""
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[aidx(i, j), bidx(j, k), cidx(k, i)] = 1.0
    return T

def _to_rational(mat, tol=1e-7, max_den=1 << 14):
    """Convert a numeric matrix to exact rationals (Fractions) if close enough."""
    from fractions import Fraction
    out = []
    for row in mat:
        r = []
        for v in row:
            if abs(v) < tol:
                r.append(Fraction(0, 1))
                continue
            f = Fraction(v).limit_denominator(max_den)
            if abs(float(f) - v) > tol:
                return None
            r.append(f)
        out.append(r)
    return out

def verify(U, V, W, T, tol=1e-7):
    """Check sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    r = U.shape[1]
    S = np.zeros((N * M, M * P, P * N))
    for c in range(r):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return bool(np.max(np.abs(S - T)) < tol)

def schoolbook():
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[aidx(i, j), col] = 1.0
                V[bidx(j, k), col] = 1.0
                W[cidx(k, i), col] = 1.0
                col += 1
    return U, V, W

def search_rank15(T, time_budget, seed):
    """
    Fix C (15 output vectors) as a basis. Solve the linear system for A and B.
    The system: for each (i,j), the 15-vector T[i,j,:] is a linear combination
    of the 15 vectors (A[i,j,:] * B[:,k])_k = sum_r A[i,j,r]*B[r,k].
    With C fixed, unknowns are A (45) and B (45).
    """
    rng = np.random.default_rng(seed)
    # C: (15, 15), each column c is a vector over (k,i).
    # We need the 15 vectors C[:,c] (as functions of (k,i)) to be linearly independent
    # so that the map from (A,B) to the output tensor is well-posed.
    # Actually the constraint is: for each (i,j), T[i,j,k,n] = sum_c A[i,j,c] * B[j,k,c] * C[k,n,c].
    # This is bilinear in (A,B). For fixed C, it's linear in A and B jointly? No, it's
    # bilinear. We need a different approach.
    #
    # Alternate: fix A and B, solve for C linearly. For fixed A,B, the equation
    # T[i,j,k,n] = sum_c A[i,j,c] B[j,k,c] C[k,n,c] is linear in C.
    # Unknowns: C has 15*15 = 225 entries. Equations: 3*3*5*3 = 135.
    # Under-determined. We can fix A,B randomly and check if a solution C exists
    # (it always does, but we need C to have rank 15, i.e., the 15 vectors C[:,c]
    # must span). Actually we just need a valid C; any solution works.
    #
    # Better: fix B (45 entries) and A (45 entries) randomly, then solve for C.
    # The system for C: for each (i,j,k,n): sum_c (A[i,j,c]*B[j,k,c]) C[k,n,c] = T[i,j,k,n].
    # This is linear in C. Unknowns C[k,n,c], c=0..14, (k,n) = 15 combos => 225 unknowns.
    # Equations: (i,j) = 9, (k,n) = 15 => 135 equations.
    # Under-determined by 90. We can set 90 of them to 0 and solve.
    # But we need the solution to be exact (rational). Random A,B won't give rational C.
    #
    # Alternate approach: use the fact that for rank 15, we can set A and B to
    # specific structured values. Let's try: set A[i,j,c] = 1 if c == (i*3+j) else 0,
    # and B[j,k,c] = 1 if c == (j*5+k) else 0. Then A has 9 nonzeros, B has 15 nonzeros.
    # Then for each (i,j), only c = i*3+j is active in A. So the equation becomes:
    # T[i,j,k,n] = A[i,j,c0] * B[j,k,c0] * C[k,n,c0] where c0 = i*3+j.
    # But B[j,k,c0] is nonzero only if c0 == j*5+k. For c0 = i*3+j, this requires
    # i*3+j = j*5+k => k = i*3 - 4j. This is only valid for specific (i,j,k).
    # This doesn't work in general.
    #
    # Let me try a different structured approach. Set C to a specific form.
    # C[k,n,c]: let's set C[k,n,c] = 1 if c == k*3+n else 0. Then C has 15 nonzeros.
    # Equation: T[i,j,k,n] = sum_c A[i,j,c] B[j,k,c] C[k,n,c].
    # Only c = k*3+n is active in C. So:
    # T[i,j,k,n] = A[i,j,k*3+n] * B[j,k,k*3+n] * 1.
    # We need A[i,j,k*3+n] * B[j,k,k*3+n] = T[i,j,k,n] = 1 if (i,j,k) is valid.
    # So for each (i,j,k): A[i,j,k*3+0]*B[j,k,k*3+0] = T[i,j,k,0] = 1 (if k is in range, which it is)
    # Wait, T[i,j,k,n] = 1 for all n (since the matmul output has all n entries).
    # Actually T[i,j,k,n] = 1 for all (i,j,k,n) where the product is defined.
    # For matmul of (n x m) * (m x p), the output is (n x p), so T[i,j,k,n] = 1
    # means: the (i,n) entry of the output gets contribution 1 from input (i,j) and (j,k).
    # So T[i,j,k,n] = 1 for all i,j,k,n. That's 3*3*5*3 = 135 ones.
    #
    # With C[k,n,c] = 1 if c == k*3+n:
    # For each (i,j,k,n): A[i,j,k*3+n] * B[j,k,k*3+n] = 1.
    # Let c = k*3+n. Then A[i,j,c] * B[j,k,c] = 1 for all i,j,k,n.
    # For fixed (j,k), and all i, n: A[i,j,c] * B[j,k,c] = 1 where c = k*3+n.
    # As n varies, c = k*3+n takes values k*3, k*3+1, k*3+2.
    # So for fixed (j,k): A[i,j,k*3]*B[j,k,k*3] = 1, A[i,j,k*3+1]*B[j,k,k*3+1] = 1, A[i,j,k*3+2]*B[j,k,k*3+2] = 1, for all i.
    # This means A[i,j,k*3] must be the same for all i (since B[j,k,k*3] is fixed), and their product is 1.
    # Let A[i,j,k*3] = a_{j,k*3} (independent of i), B[j,k,k*3] = 1/a_{j,k*3}.
    # Similarly for the other two.
    # But A[i,j,c] for c = k*3+n: as (k,n) varies, c ranges over 0..14.
    # For c = k*3+n, A[i,j,c] must be independent of i. So A[i,j,c] = a_{j,c} for all i.
    # Then B[j,k,c] = 1/a_{j,c} where c = k*3+n.
    # But B[j,k,c] depends on (j,k,c). For fixed (j,k), c = k*3, k*3+1, k*3+2.
    # So B[j,k,k*3] = 1/a_{j,k*3}, B[j,k,k*3+1] = 1/a_{j,k*3+1}, B[j,k,k*3+2] = 1/a_{j,k*3+2}.
    # And A[i,j,c] = a_{j,c} for all i.
    # Check: A[i,j,c]*B[j,k,c] = a_{j,c} * (1/a_{j,c}) = 1. Good.
    # So A[i,j,c] = a_{j,c}, B[j,k,c] = 1/a_{j,c} (where c = k*3+n), C[k,n,c] = 1 if c=k*3+n else 0.
    # But wait, B[j,k,c] is only defined for c in {k*3, k*3+1, k*3+2}. For other c, B[j,k,c] = 0.
    # And A[i,j,c] = a_{j,c} for all c.
    # Let's verify: sum_c A[i,j,c] B[j,k,c] C[k,n,c] = sum_c a_{j,c} B[j,k,c] * (1 if c=k*3+n else 0)
    # = a_{j,k*3+n} * B[j,k,k*3+n] = a_{j,k*3+n} * (1/a_{j,k*3+n}) = 1.
    # So this works for any nonzero a_{j,c}!
    # But we need rational entries. Set a_{j,c} = 1 for all j,c. Then A[i,j,c] = 1, B[j,k,c] = 1 if c in {k*3,k*3+1,k*3+2} else 0.
    # Check: A[i,j,c] = 1 for all i,j,c. B[j,k,c] = 1 if c = k*3+n for some n, else 0.
    # C[k,n,c] = 1 if c = k*3+n else 0.
    # sum_c 1 * B[j,k,c] * C[k,n,c] = B[j,k,k*3+n] * C[k,n,k*3+n] = 1 * 1 = 1.
    # So A = all ones (9*15 = 135 ones), B[j,k,c] = 1 if c = k*3, k*3+1, or k*3+2 (i.e., c mod 3 == n where n = c - 3k... wait c = k*3+n, so for fixed (j,k), c takes values k*3, k*3+1, k*3+2, so B[j,k,c] = 1 for c in {k*3, k*3+1, k*3+2}).
    # C[k,n,c] = 1 if c = k*3+n.
    # Let me double check with a specific example. i=0,j=0,k=0,n=0:
    # sum_c A[0,0,c] B[0,0,c] C[0,0,c] = sum_c 1 * B[0,0,c] * C[0,0,c].
    # B[0,0,c] = 1 if c in {0,1,2}. C[0,0,c] = 1 if c = 0*3+0 = 0.
    # So only c=0 contributes: 1*1*1 = 1. T[0,0,0,0] = 1. Good.
    # i=0,j=0,k=0,n=1: C[0,1,c] = 1 if c = 0*3+1 = 1. B[0,0,1] = 1. So sum = 1. T[0,0,0,1] = 1. Good.
    # i=0,j=1,k=2,n=3: C[2,3,c] = 1 if c = 2*3+3 = 9. B[1,2,c] = 1 if c in {2*3, 2*3+1, 2*3+2} = {6,7,8}. c=9 is not in {6,7,8}. So sum = 0. But T[0,1,2,3] = 1. FAIL!
    # Ah, I see the issue. B[j,k,c] should be 1 if c = k*3+n for some n, but the n in C[k,n,c] must match. Let me re-examine.
    # The equation is: T[i,j,k,n] = sum_c A[i,j,c] B[j,k,c] C[k,n,c].
    # With A[i,j,c] = 1, C[k,n,c] = 1 if c = k*3+n.
    # Then T[i,j,k,n] = sum_c B[j,k,c] * (1 if c = k*3+n else 0) = B[j,k,k*3+n].
    # So we need B[j,k,k*3+n] = 1 for all j,k,n.
    # B[j,k,c] where c = k*3+n. As n ranges 0..2, c ranges k*3, k*3+1, k*3+2.
    # So B[j,k,c] = 1 for c in {k*3, k*3+1, k*3+2}, for all j.
    # So B[j,k,c] = 1 if c mod 3 == n where... no, c = k*3+n, so for fixed k, c takes values k*3, k*3+1, k*3+2.
    # So B[j,k,c] = 1 if c is in {k*3, k*3+1, k*3+2}.
    # For k=2: c in {6,7,8}. For n=3: c = k*3+n = 2*3+3 = 9. But 9 is not in {6,7,8}.
    # Wait, n ranges 0..2 (since N=3), not 0..3. Let me recheck. N=3, so n in {0,1,2}.
    # C[k,n,c] = 1 if c = k*3+n. For k=2, n=2: c = 2*3+2 = 8. B[j,2,8] = 1 (since 8 in {6,7,8}). Good.
    # I made an error: n is 0-indexed, so n in {0,1,2}.
    # Let me re-verify i=0,j=1,k=2,n=2: C[2,2,c] = 1 if c = 2*3+2 = 8. B[1,2,8] = 1. Sum = 1. T[0,1,2,2] = 1. Good.
    # So the decomposition is:
    # A[i,j,c] = 1 for all i,j,c.
    # B[j,k,c] = 1 if c in {k*3, k*3+1, k*3+2} else 0.
    # C[k,n,c] = 1 if c = k*3+n else 0.
    # Let me verify all entries quickly.
    # For any (i,j,k,n): sum_c A[i,j,c] B[j,k,c] C[k,n,c] = sum_c 1 * B[j,k,c] * (1 if c=k*3+n else 0) = B[j,k,k*3+n].
    # B[j,k,k*3+n] = 1 if k*3+n in {k*3, k*3+1, k*3+2}, which is true since n in {0,1,2}.
    # So the sum is always 1. And T[i,j,k,n] = 1 for all (i,j,k,n).
    # This is a valid rank-15 decomposition!
    U = np.ones((N * M, 15))
    V = np.zeros((M * P, 15))
    W = np.zeros((P * N, 15))
    for j in range(M):
        for k in range(P):
            for n in range(N):
                c = k * N + n
                V[bidx(j, k), c] = 1.0
                W[cidx(k, n), c] = 1.0
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s
    T = build_T(15)
    # Try the structured rank-15 solution
    U, V, W = search_rank15(T, deadline - time.time(), 0)
    if verify(U, V, W, T):
        # Try to convert to rationals
        U_r = _to_rational(U)
        V_r = _to_rational(V)
        W_r = _to_rational(W)
        if U_r is not None and V_r is not None and W_r is not None:
            return {"U": U_r, "V": V_r, "W": W_r}
        # Fall back to floats if rational conversion fails (shouldn't happen here)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
