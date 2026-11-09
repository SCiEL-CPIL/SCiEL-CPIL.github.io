"""Tensor rank search for 2x3x7 matrix multiplication (exact, integer)."""
import random
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _valid_triples(n, m, p):
    # A valid triple (i,j,k) is one where all three factor indices are in range.
    # For the multiplication tensor, every (i,j,k) with 0<=i<n,0<=j<m,0<=k<p is a
    # "natural" support, but a rank-1 term can also have a nonzero entry at any of
    # the three factor positions as long as the other two are also nonzero.
    # We allow any combination of one index per factor.
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]


def _make_term(i, j, k, n, m, p, val):
    # Returns (u_dict, v_dict, w_dict) for a rank-1 term with value `val`.
    u = {(i * m + j): val}
    v = {(j * p + k): val}
    w = {(k * n + i): val}
    return u, v, w


def _try_complete(terms, n, m, p, max_terms, deadline, rng):
    """Given a partial list of terms, try to complete to an exact decomposition.
    Returns (rank, U, V, W) or None."""
    # Current residual: need to add terms so that sum of outer products = T.
    # Represent current partial sum and target.
    # Target: T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p).
    # We track residual as dict (i,j,k)->int (target - current).
    residual = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                residual[(i, j, k)] = 1

    for (i, j, k), val in terms:
        residual[(i, j, k)] -= val

    # Now greedily add terms to zero out residual.
    # A term (i,j,k) with value v reduces residual[(i,j,k)] by v.
    # But we want exact: we can only use integer values, and each term touches
    # exactly one (i,j,k) in the residual (since each term's support is a single
    # triple).
    # So the problem reduces to: assign an integer value to each of the n*m*p
    # triples such that the sum of all assigned values (including the initial
    # terms) equals 1 at every triple, using at most max_terms nonzero assignments.
    #
    # Wait - that's not right. A rank-1 term in the tensor decomposition
    # T = sum_r u_r ⊗ v_r ⊗ w_r has T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r].
    # A "term" with a single nonzero in each factor contributes val to exactly
    # one (i,j,k). So yes, each term touches exactly one (i,j,k).
    #
    # Therefore an exact decomposition with rank R is equivalent to assigning
    # R integers (the term values) to R of the n*m*p triples such that the sum
    # over terms touching (i,j,k) equals 1 for every (i,j,k).
    #
    # Since each term touches exactly one triple, we need the sum of values
    # assigned to triple (i,j,k) to be 1. The minimum number of terms is
    # achieved by assigning value 1 to each triple, giving rank = n*m*p = 42.
    #
    # To get rank < 42, we would need some triples to have value 0 and others
    # to have value >1 or negative values, but the sum must still be 1.
    # E.g., triple A gets value 2, triple B gets value -1, and all others get 1.
    # But that still uses 42 terms (one per triple).
    #
    # Actually, we can have FEWER terms if some triples get value 0 (no term)
    # and we compensate... but no, each triple independently needs sum = 1.
    # If triple (i,j,k) has no term, its sum is 0, not 1. So every triple must
    # have at least one term. Hence rank >= 42.
    #
    # Wait, that can't be right. Let me reconsider.
    #
    # The tensor T is in the space of n x m x p arrays. The multiplication
    # tensor has T[i,j,k] = 1 for all (i,j,k). A rank-1 tensor is u ⊗ v ⊗ w
    # which has (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k]. This is NOT a single
    # (i,j,k) entry - it's a full n x m x p array!
    #
    # I made a fundamental error. Let me redo this.
    #
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, the factors are U of shape (n*m, rank), V of shape (m*p, rank),
    # W of shape (p*n, rank). The tensor is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # where U[i,j,r] = U[i*m+j, r], etc.
    #
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    #
    # For the multiplication tensor, T[i,j,k] = 1 for all (i,j,k).
    #
    # A rank-1 term with U[i*m+j, r]=a, V[j*p+k, r]=b, W[k*n+i, r]=c
    # contributes a*b*c to T[i,j,k] and 0 to all other (i',j',k') where
    # (i',j') != (i,j) or (j',k') != (j,k) or (k',i') != (k,i).
    #
    # No wait, that's also not right. U[i*m+j, r] is nonzero only for the
    # specific (i,j) pair, but V[j*p+k, r] is nonzero only for the specific
    # (j,k) pair, and W[k*n+i, r] is nonzero only for the specific (k,i) pair.
    #
    # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is nonzero only when
    # ALL THREE are nonzero, which requires (i,j) to be the specific pair,
    # (j,k) to be the specific pair, and (k,i) to be the specific pair.
    # These are consistent only for a specific (i,j,k).
    #
    # So yes, a rank-1 term with a single nonzero in each factor contributes
    # to exactly one (i,j,k). My original analysis was correct.
    #
    # But that means the rank is at least 42, which contradicts the possibility
    # of improvement. Let me reconsider the problem.
    #
    # AH WAIT. I think I'm confusing myself. Let me re-read the problem.
    #
    # "Matrix multiplication 2,3,7: Find a rank-minimal exact tensor
    # decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."
    #
    # The standard matrix multiplication tensor for multiplying an n x m
    # matrix by an m x p matrix has entries T[i,j,k] where i indexes rows of
    # the first matrix, j indexes columns of first / rows of second, k indexes
    # columns of second. The tensor is T[i,j,k] = 1 (it's the "all ones"
    # tensor in the appropriate indexing).
    #
    # But the RANK of this tensor is NOT n*m*p in general! The rank of the
    # matrix multiplication tensor is a deep open problem. For 2x3x7, the
    # rank is known to be less than 42.
    #
    # The key insight I was missing: a rank-1 term in the tensor decomposition
    # does NOT have to have a single nonzero in each factor. It can have
    # multiple nonzeros! For example, U could have nonzeros at (i1,j1) and
    # (i2,j2), V could have nonzeros at (j1,k1) and (j2,k2), etc. The product
    # then contributes to multiple (i,j,k) entries.
    #
    # So the search space is much larger. Let me redo the approach.

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # safety margin before 150s limit
    rng = random.Random(42)

    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p

    # For a rank-R decomposition, we need R columns in U, V, W.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    #
    # This is a system of n*m*p = 42 equations in R*(n*m + m*p + p*n) = R*(6+21+14) = 41R unknowns.
    # For R < 42, we have 41R unknowns and 42 equations. For R=1, 41 unknowns, 42 eqs.
    # The system is underdetermined for R >= 2 (41R >= 82 > 42).
    #
    # We need to find integer/half-integer solutions. Let's use a randomized
    # approach: fix most entries and solve for the rest.

    # Strategy: for a given rank R, try to find a solution by:
    # 1. Randomly initialize U, V, W with small integers/half-integers.
    # 2. Compute the residual T - sum of outer products.
    # 3. Try to fix the residual by adjusting entries.
    #
    # A more systematic approach: use the fact that the tensor is "all ones".
    # We can try to find a decomposition where each rank-1 term is a "block"
    # covering multiple (i,j,k) entries.

    # Let's try a structured approach. For rank R, we want:
    # T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This must hold for all (i,j,k). Consider fixing the structure of U, V, W.
    #
    # A known technique: use "border rank" or "tensor rank" algorithms.
    # For small cases, we can do an exhaustive or randomized search.

    # Let's try a randomized search over small integer/half-integer values.
    # For each rank R from some lower bound up to 41, try random initializations
    # and see if we can find an exact solution.

    # Lower bound on rank: The tensor has 42 nonzero entries. Each rank-1 term
    # can cover at most (number of nonzeros in U_col) * (number of nonzeros in V_col) *
    # (number of nonzeros in W_col) entries. If each column has at most 1 nonzero,
    # each term covers 1 entry, so rank >= 42. If each column has 2 nonzeros,
    # each term covers up to 8 entries, so rank >= ceil(42/8) = 6.
    #
    # In practice, the rank of the 2x3x7 multiplication tensor is known to be
    # around 12-15 or so. Let's search from rank 10 upward.

    for R in range(10, best_rank):
        if time.time() > deadline - 5:
            break

        # Try a few random initializations for this rank
        for trial in range(200):
            if time.time() > deadline - 5:
                break

            # Initialize U, V, W with random small values
            # Use a mix of 0, 1, -1, 0.5, -0.5
            vals = [0, 1, -1, 0.5, -0.5, 2, -2]

            U = [[rng.choice(vals) for _ in range(R)] for _ in range(n * m)]
            V = [[rng.choice(vals) for _ in range(R)] for _ in range(m * p)]
            W = [[rng.choice(vals) for _ in range(R)] for _ in range(p * n)]

            # Compute T_hat[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Check if T_hat == 1 for all (i,j,k)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                # Found a valid decomposition of rank R!
                best = {"U": U, "V": V, "W": W}
                best_rank = R
                # Try to go lower
                break

    return best
# EVOLVE-BLOCK-END
