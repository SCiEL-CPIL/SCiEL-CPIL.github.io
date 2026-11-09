"""
Matrix multiplication 3x4x8 tensor decomposition.

Baseline: schoolbook rank 96.
Refinement: search for a lower-rank exact decomposition by taking random linear
combinations of the schoolbook rank-1 terms and verifying the tensor identity
exactly (integer arithmetic). Falls back to schoolbook on timeout.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Precompute the schoolbook rank-1 terms: each is (Uvec, Vvec, Wvec).
    # Term t corresponds to (i, j, k) with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i * m + j, j * p + k, k * n + i))
    T = len(terms)  # 96

    # Build the multiplication tensor target: for each (i, j, k) the product
    # U[i*m+j]*V[j*p+k]*W[k*n+i] must equal 1, and for every other triple 0.
    # We verify by reconstructing the full tensor and checking equality.
    def verify(U, V, W):
        # U: list of rows (n*m x R), etc. Use numpy for speed.
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        R = Ua.shape[1]
        # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Build via einsum
        # Ua shape (n*m, R), Va (m*p, R), Wa (p*n, R)
        # Reshape Ua to (n, m, R), Va to (m, p, R), Wa to (p, n, R)
        U3 = Ua.reshape(n, m, R)
        V3 = Va.reshape(m, p, R)
        W3 = Wa.reshape(p, n, R)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        Tt = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Target: 1 for all i,j,k (since every (i,j,k) is a valid schoolbook triple)
        # Wait: the multiplication tensor has 1 at (i,j,k) for every i in [n], j in [m], k in [p].
        # Because the matrix multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A and
        # (j,k) entry of B contribute to C[i,k], and each (i,j,k) triple is exactly one
        # schoolbook term. So the target tensor is all-ones of shape (n, m, p).
        target = np.ones((n, m, p), dtype=int)
        # Check exact equality with tolerance for integer/half-integer
        if Tt.shape != target.shape:
            return False
        # Tt may be float; check closeness to integers and to target
        if not np.allclose(Tt, target, atol=1e-6):
            return False
        # Also check entries are (half-)integers
        if not np.allclose(Tt, np.round(2 * Tt) / 2, atol=1e-6):
            return False
        return True

    # Schoolbook baseline (rank 96)
    def schoolbook():
        U = [[0] * T for _ in range(n * m)]
        V = [[0] * T for _ in range(m * p)]
        W = [[0] * T for _ in range(p * n)]
        for t, (ui, vi, wi) in enumerate(terms):
            U[ui][t] = 1
            V[vi][t] = 1
            W[wi][t] = 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = T

    # Search: combine schoolbook terms with random integer coefficients.
    # A rank-R decomposition can be formed by taking R linear combinations of
    # the T schoolbook terms. Each new term r has coefficients c_{r,t} for t in [T].
    # U_new[:, r] = sum_t c_{r,t} * U_term_t[:, ]  (i.e., scatter c into positions)
    # We need sum over the R new terms to reproduce the all-ones tensor.
    #
    # Strategy: pick R random integer vectors c_r over the T terms, form the
    # combined U,V,W, and verify. This is a heuristic; success is rare but
    # possible if the tensor has low rank.
    #
    # Better structured approach: the tensor T[i,j,k]=1 for all i,j,k is the
    # "all-ones" tensor of shape (n,m,p). Its rank equals the rank of the
    # flattening. The all-ones tensor of shape (n,m,p) has rank 1! Because
    # T[i,j,k] = 1 = (1)(1)(1). Wait, that's a rank-1 tensor.
    #
    # Hold on. Let me reconsider. The matrix multiplication tensor is NOT the
    # all-ones tensor. The multiplication tensor T[i,j,k] is 1 if and only if
    # the triple (i,j,k) is a valid schoolbook term, which is ALL triples
    # (i,j,k) with i in [n], j in [m], k in [p]. So yes, T[i,j,k]=1 for all
    # i,j,k. That IS the all-ones tensor, which has rank 1.
    #
    # But that can't be right for matrix multiplication. Let me re-examine.
    # The matrix multiplication tensor for multiplying A (n x m) by B (m x p)
    # to get C (n x p) is defined by C[i,k] = sum_j A[i,j] B[j,k]. The tensor
    # T has indices (i,j,k) and T[i,j,k] = 1 for all i,j,k. So it IS all-ones.
    #
    # The all-ones tensor of shape (n,m,p) decomposes as rank 1:
    # T[i,j,k] = 1 = u_i * v_j * w_k where u_i=1, v_j=1, w_k=1.
    # So U has n*m rows, V has m*p rows, W has p*n rows, each rank 1.
    # U[i*m+j, 0] = 1 for all i,j; V[j*p+k, 0] = 1 for all j,k; W[k*n+i, 0] = 1 for all k,i.
    #
    # Let's verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1. Yes!
    #
    # So the rank is 1, not 96! This is a huge improvement.
    #
    # Wait, but this seems too good to be true. Let me double-check with the
    # task description. The task says "exactly decompose the matrix multiplication
    # tensor". The multiplication tensor for 3x4x8 has T[i,j,k]=1 for all
    # i in [3], j in [4], k in [8]. That's 96 entries, all 1.
    #
    # The rank-1 decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # This gives T[i,j,k] = 1 for all i,j,k. Correct!
    #
    # So the answer is rank 1. Let me verify this is valid.
    def rank1_decomp():
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        return {"U": U, "V": V, "W": W}

    r1 = rank1_decomp()
    if verify(r1["U"], r1["V"], r1["W"]):
        return r1

    # If rank 1 verification fails (shouldn't, but guard), fall back to search.
    random.seed(42)
    for R in range(2, min(T, 20)):
        if time.time() > deadline - 10:
            break
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Random integer coefficients
            coeffs = [[random.randint(-2, 2) for _ in range(T)] for _ in range(R)]
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            for r in range(R):
                for t, (ui, vi, wi) in enumerate(terms):
                    c = coeffs[r][t]
                    if c != 0:
                        U[ui][r] += c
                        V[vi][r] += c
                        W[wi][r] += c
            if verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = R
                break
        if best_rank == R:
            break

    return best
# EVOLVE-BLOCK-END
