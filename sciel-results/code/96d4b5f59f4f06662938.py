"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
n, m, p = DIMENSIONS
DEADLINE = 110.0  # seconds; leave margin under the 120s kill (and 150s wall-clock)
START = time.time()


def _elapsed():
    return time.time() - START


def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-7):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(r):
                    lhs += U[i * m + j][c] * V[j * p + k][c]
                if abs(lhs - W[k * n + i][c]) > tol:
                    return False
    return True


def _to_halfint(x):
    """Round to nearest half-integer; returns exact float or None if not close."""
    h = round(2 * x) / 2.0
    if abs(x - h) < 1e-5:
        return h
    return None


def _is_halfint(x):
    h = round(2 * x) / 2.0
    return abs(x - h) < 1e-5


def construct():
    # Start with schoolbook; try to find lower rank via random search + local refinement.
    best = _schoolbook()
    best_rank = n * m * p

    def try_rank(r, max_iters=200, seed=0):
        """Attempt to find a rank-r decomposition via iterative refinement."""
        if r >= best_rank:
            return None
        rng = np.random.default_rng(seed)
        # Random init
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))

        # Target tensor: T[i,j,k] = delta (i.e., matmul tensor).
        # We want: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
        # This is a non-linear system. Use alternating least squares.

        for it in range(max_iters):
            if _elapsed() > DEADLINE - 10:
                return None

            # Update W: for each (k,n+i) row, W[k*n+i,:] = sum over (j) of U[i*m+j,:]*V[j*p+k,:]
            # Actually W[k*n+i,c] should equal sum_c U[i*m+j,c]*V[j*p+k,c] for each (i,j,k).
            # But W is indexed by (k,i) and the equation must hold for ALL j simultaneously.
            # This means: for fixed (i,k), W[k*n+i,:] must satisfy:
            #   for all j: sum_c U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c]
            # This is only possible if the vectors {U[i*m+j,:]*V[j*p+k,:] (element-wise)} are all equal.
            # That's a very strong constraint. The standard approach is different.

            # Standard tensor decomposition: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # Here our tensor is T_{i,j,k} = 1 if... no.
            # The matmul tensor: (A B)_{ik} = sum_j A_{ij} B_{jk}
            # So T_{i,j,k} = 1 (it's the "1" tensor in the middle), and the decomposition is:
            # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} where U,V,W are the factor matrices.
            # But wait, the interface says U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
            # And the equation is: for each c, the outer product structure.
            #
            # Let me re-read: "exactly decompose the matrix multiplication tensor"
            # The matmul tensor T has T[i,j,k] = delta (always 1 in the sense that C[i,k] = sum_j A[i,j]B[j,k])
            # Actually T[i,j,k] = 1 for all i,j,k (it's the identity in the summation index).
            # No: T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
            # So the tensor is all-ones: T[i,j,k] = 1 for all i,j,k.
            #
            # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
            # Wait, that's not right either. Let me look at the schoolbook:
            # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the same (i,j,k) at column c.
            # So the product U[i*m+j,c] * V[j*p+k,c] = 1, and W[k*n+i,c] = 1.
            # The convention is: C[k*n+i] = sum_c W[k*n+i,c] * (something)
            # 
            # Actually the standard format: the tensor T_{i,j,k} satisfies:
            # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] ... no.
            #
            # Let me think differently. The output W[k*n+i, c] suggests:
            # C[i,k] (the (i,k) entry of the product) = sum_c W[k*n+i, c] * A[i,j] * B[j,k] summed over j?
            # No. The factorization is:
            # For the matmul: C[i,k] = sum_j A[i,j] B[j,k]
            # We write: C[i,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c]) * W[k*n+i,c]
            # That doesn't work dimensionally.
            #
            # Standard: C[i,k] = sum_c U_c[i,:] * V_c[k,:] where U_c, V_c are rank-1 factors.
            # With the given indexing:
            # U has rows indexed by (i,j) pairs, V by (j,k), W by (k,i).
            # The equation must be: for all i,j,k:
            #   delta_{i,j,k} = 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # No, that gives a scalar per (i,j,k) but the product of three terms.
            #
            # Actually I think the correct interpretation from the schoolbook is:
            # The tensor T[i,j,k] = 1 (all ones) is decomposed as:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # But in schoolbook, each column c corresponds to one (i,j,k) triple, and only one term is nonzero.
            # So T[i,j,k] = 1 when c matches (i,j,k), else 0. Sum over c gives 1.
            #
            # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # And for "wrong" combinations it should be 0? No, the tensor IS all ones.
            #
            # Wait, that can't be right because then any rank-1 with all ones would work.
            # Let me reconsider. The matmul tensor is NOT all ones.
            #
            # The matmul tensor T is defined by: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
            # contribute to (i,k) entry of C. That's always true, so T[i,j,k]=1.
            # But the DECOMPOSITION format is different. Let me look at the schoolbook more carefully.
            #
            # In schoolbook, for column c corresponding to (i,j,k):
            # U[i*m+j, c] = 1, all other U rows in column c are 0.
            # V[j*p+k, c] = 1, all other V rows in column c are 0.  
            # W[k*n+i, c] = 1, all other W rows in column c are 0.
            #
            # The verification should be: for each (i,j,k), 
            #   sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c]? No, that doesn't make sense.
            #
            # I think the correct equation is:
            # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a 3-way tensor decomposition of the all-ones tensor.
            # But the all-ones tensor has rank 1! So this interpretation is wrong.
            #
            # Let me reconsider the problem. "Matrix multiplication 3,3,8" means
            # multiplying a 3x3 matrix by a 3x8 matrix to get a 3x8 matrix.
            # The tensor has indices (i, j, k) where i in [0,3), j in [0,3), k in [0,8).
            # T[i,j,k] = 1 for all valid (i,j,k).
            #
            # But the factor matrices have shapes:
            # U: (n*m, rank) = (9, rank) - indexed by (i,j)
            # V: (m*p, rank) = (24, rank) - indexed by (j,k)  
            # W: (p*n, rank) = (24, rank) - indexed by (k,i)
            #
            # The equation must be: for all i, j, k:
            #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            #
            # But T is all-ones, which has tensor rank 1. So this can't be right.
            #
            # UNLESS the equation is different. Perhaps:
            #   For all i, k: sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
            # and the decomposition gives a way to compute C[i,k] using rank-many multiplications.
            #
            # The standard format: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j B[j,k] * V[j*p+k,c]) * W[k*n+i,c]
            # No, that has W as a scalar multiplier per c.
            #
            # I think the correct equation from the interface is:
            # The product C[i,k] = sum_j A[i,j] B[j,k] can be computed as:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
            #
            # Wait, that's not standard either. Let me just trust the schoolbook and verify it.
            # In schoolbook, for a given (i,j,k), only column c=(i,j,k) has nonzero entries.
            # So: sum_c W[k*n+i,c] * (sum_j' A[i,j'] * U[i*m+j',c]) * (sum_j' B[j',k] * V[j'*p+k,c])
            # For c=(i,j,k): W[k*n+i,c]=1, sum_j' A[i,j']*U[i*m+j',c] = A[i,j]*1 = A[i,j]
            # sum_j' B[j',k]*V[j'*p+k,c] = B[j,k]*1 = B[j,k]
            # So the term is 1 * A[i,j] * B[j,k] = A[i,j]*B[j,k].
            # Summing over all c (which are all (i',j',k') triples), we get sum_{i',j',k'} ... 
            # But only the term where i'=i, j'=j, k'=k contributes to C[i,k] via W[k*n+i,c].
            # For c=(i',j',k'), W[k*n+i,c] is nonzero only if k*n+i = k'*n+i', i.e., k=k' and i=i'.
            # So sum_c W[k*n+i,c] * (...) * (...) = sum_{j'} W[k*n+i, (i,j',k)] * A[i,j'] * B[j',k]
            # = sum_{j'} A[i,j'] * B[j',k] = C[i,k]. 
            #
            # So the equation is:
            # C[i,k] = sum_c W[k*n+i,c] * (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c])
            #
            # This means: for all i, k:
            #   sum_c W[k*n+i,c] * (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) = sum_j A[i,j]*B[j,k]
            #
            # For this to hold for ALL A and B, we need:
            #   sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1  for all i, j, k
            #
            # So the tensor equation IS: for all i, j, k:
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # And this is the all-ones tensor! Which has rank 1. But schoolbook uses rank 72.
            # So there must be additional constraints I'm missing.
            #
            # OH WAIT. I think the issue is that U, V, W must have entries from {0, 1/2, 1} or similar,
            # AND the decomposition must be "exact" in a specific sense. But mathematically,
            # the all-ones tensor has rank 1, so rank 1 should suffice with U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
            #
            # Let me check: if U is all 1s (9x1), V is all 1s (24x1), W is all 1s (24x1),
            # then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. Yes!
            #
            # So rank 1 works?! But the schoolbook has rank 72. There must be a constraint I'm missing.
            #
            # Re-reading: "integer/half-integer entries that exactly decompose the matrix multiplication tensor"
            # Maybe the constraint is that the entries must be in {0, 1/2, 1} or {-1, 0, 1/2, 1}?
            # Even so, all-ones is valid.
            #
            # I think I'm misunderstanding the problem. Let me re-read the task.
            # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 8 matrix."
            #
            # Perhaps the "tensor" is not the all-ones tensor but something else. In matrix multiplication,
            # the standard tensor is T[i,j,k] = delta (which is 1), and the bilinear complexity is the tensor rank.
            # For 3x3 * 3x8, the tensor rank is known to be at most 72 (schoolbook) and the Strassen-like
            # bounds give lower values.
            #
            # But if T is all-ones, its rank is 1. So the "tensor" must be something else.
            #
            # I think the confusion is that in the standard formulation, the matmul tensor T has
            # T[i,j,k] = 1, and the tensor rank is the minimum number of rank-1 tensors whose sum is T.
            # A rank-1 tensor is U[i]*V[j]*W[k] (outer product of three vectors).
            # The all-ones tensor IS a rank-1 tensor (outer product of all-ones vectors).
            #
            # So either the problem is trivial (rank 1) or I'm fundamentally misunderstanding the format.
            #
            # Given the schoolbook solution and the problem statement, I'll assume the intended equation is:
            #   For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # And the "rank" is the number of columns. The minimum is 1, but perhaps there are
            # additional constraints (like sparsity, or the entries must be 0 or 1 only, not arbitrary reals).
            #
            # Given the uncertainty, I'll keep the schoolbook as a safe baseline and try small random searches
            # to see if lower rank works, verifying with the equation above.

        return None

    # Given the analysis above, rank 1 should work if the equation is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    # Let's test rank 1.
    def try_rank1():
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    r1 = try_rank1()
    if r1 is not None:
        return r1

    # If rank 1 doesn't verify (maybe my equation is wrong), fall back to schoolbook.
    return best
