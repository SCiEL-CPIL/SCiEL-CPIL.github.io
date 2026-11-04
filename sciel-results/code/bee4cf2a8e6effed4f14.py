# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    # Double centering: B = -1/2 J M J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    # Eigendecomposition (B symmetric)
    w, V = np.linalg.eigh(B)
    # Take largest k eigenvalues
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    # X = V * sqrt(w)
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian-like greedy.
    Actually solve the assignment: minimize sum_{i} ||X[i] - Y[p[i]]||^2.
    This is a linear assignment problem on the cost matrix C[i,j] = ||X[i]-Y[j]||^2.
    Use scipy's linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment
    n = X.shape[0]
    # C[i,j] = ||X[i] - Y[j]||^2 = ||X[i]||^2 - 2 X[i].Y[j] + ||Y[j]||^2
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    # linear_sum_assignment minimizes sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _two_opt_search(a, b, p, time_limit, max_iter=500000):
    """Vectorized 2-opt local search. p is numpy int64 array (facility i -> location p[i]).
    We try swapping locations of two facilities i,j and accept if cost decreases."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # Precompute: for each facility i, its contribution depends on b[p[i],:]
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap locations of i and j (i.e., p[i]<->p[j]), the change is:
    #   delta = sum_k a[i,k]*(b[old_j, p[k]] - b[old_i, p[k]]) 
    #          + sum_k a[j,k]*(b[old_i, p[k]] - b[old_j, p[k]])
    #          + sum_k a[k,i]*(b[p[k], old_j] - b[p[k], old_i])
    #          + sum_k a[k,j]*(b[p[k], old_i] - b[p[k], old_j])
    # where old_i = p[i], old_j = p[j], and p[k] unchanged for k != i,j.
    # This is O(n) per pair, too slow for n=150 with many pairs.
    # Instead, use a smarter approach: precompute row/col contributions.

    # Let's use a different formulation. Define:
    #   r[i] = a[i, :]  (row i of a)
    #   c[i] = a[:, i]  (col i of a)
    #   For location l = p[i], the "interaction" of facility i with all others:
    #   S_i = sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    #        = r[i] @ (b[p[i], p])  +  (c[i] @ b) [p[i], :]  ... hmm not quite.
    # Actually: sum_j a[i,j]*b[p[i],p[j]] = r[i] @ b[p[i], p]
    # and sum_j a[j,i]*b[p[j],p[i]] = c[i] @ b[p, p[i]]
    # So total cost = sum_i [ r[i] @ b[p[i], p] + c[i] @ b[p, p[i]] ] / 2 ... no.
    # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
    # For fixed i, the terms involving p[i] are:
    #   sum_j a[i,j] b[p[i], p[j]]  (row i, location p[i] as first index)
    #   sum_j a[j,i] b[p[j], p[i]]  (col i, location p[i] as second index)
    # So cost = sum_i [ r[i] @ b[p[i], p] + c[i] @ b[p, p[i]] ]
    # where r[i] = a[i,:], c[i] = a[:,i].

    # When we swap p[i] <-> p[j], only terms involving p[i] and p[j] change.
    # Let li = p[i], lj = p[j].
    # Old contribution of i: r[i]@b[li,p] + c[i]@b[p,li]
    # New contribution of i: r[i]@b[lj,p] + c[i]@b[p,lj]
    # Old contribution of j: r[j]@b[lj,p] + c[j]@b[p,lj]
    # New contribution of j: r[j]@b[li,p] + c[j]@b[p,li]
    # Delta = (r[i]@b[lj,p] - r[i]@b[li,p]) + (c[i]@b[p,lj] - c[i]@b[p,li])
    #       + (r[j]@b[li,p] - r[j]@b[lj,p]) + (c[j]@b[p,li] - c[j]@b[p,lj])
    # = (r[i]-r[j]) @ (b[lj,p] - b[li,p])  ... not exactly, let me expand:
    # r[i]@b[lj,p] - r[i]@b[li,p] = r[i] @ (b[lj,p] - b[li,p])
    # c[i]@b[p,lj] - c[i]@b[p,li] = c[i] @ (b[:,lj] - b[:,li]) ... wait b[p,lj] is vector over p
    # Actually b[p, lj] means the vector [b[p[0],lj], b[p[1],lj], ...], which is b[:,lj][p]
    # So c[i] @ b[p,lj] = c[i] @ b[:,lj][p] = sum_k c[i][k] * b[p[k], lj]
    # Similarly c[i] @ b[p,li] = sum_k c[i][k] * b[p[k], li]
    # So c[i]@(b[p,lj]-b[p,li]) = sum_k c[i][k]*(b[p[k],lj]-b[p[k],li])
    #
    # This is still O(n) per swap. For n=150, that's 150 ops per pair, 150*149/2 = ~11k pairs,
    # so ~1.6M ops per full pass. In Python that's too slow. Need vectorization.

    # Let me try a different approach: compute the delta for ALL pairs at once using matrix ops.
    # Define for each location l:
    #   R_l = sum_i a[i,:] ... no.
    # Let me define:
    #   For location l, define vector u_l of length n: u_l[k] = b[l, p[k]]  (row l of b indexed by p)
    #   For location l, define vector v_l of length n: v_l[k] = b[p[k], l]  (col l of b indexed by p)
    # Then cost = sum_i [ a[i,:] @ u_{p[i]} + a[:,i] @ v_{p[i]} ]
    #
    # When swapping p[i]=li and p[j]=lj:
    # delta_i = a[i,:] @ (u_{lj} - u_{li}) + a[:,i] @ (v_{lj} - v_{li})
    # delta_j = a[j,:] @ (u_{li} - u_{lj}) + a[:,j] @ (v_{li} - v_{lj})
    # total_delta = delta_i + delta_j
    #
    # Note u_{lj} - u_{li} = b[lj, p] - b[li, p]  (row difference)
    #       v_{lj} - v_{li} = b[p, lj] - b[p, li]  (col difference)
    #
    # Let Drow[l1,l2] = b[l1,p] - b[l2,p]  (n-vector)
    # Let Dcol[l1,l2] = b[p,l1] - b[p,l2]  (n-vector)
    #
    # delta for swapping i,j (li=p[i], lj=p[j]):
    # = a[i,:] @ (b[lj,p]-b[li,p]) + a[:,i] @ (b[p,lj]-b[p,li])
    # + a[j,:] @ (b[li,p]-b[lj,p]) + a[:,j] @ (b[p,li]-b[p,lj])
    # = (a[i,:]-a[j,:]) @ (b[lj,p]-b[li,p]) + (a[:,i]-a[:,j]) @ (b[p,lj]-b[p,li])
    #
    # This is still per-pair. Let me just do it in a loop but vectorize the inner part.
    # For each pair (i,j), compute delta in O(1) using precomputed quantities?
    #
    # Alternative: use the fact that for a swap of locations li and lj (assigned to i and j),
    # the change is:
    # delta = sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + sum_k a[k,i]*(b[p[k],lj] - b[p[k],li])
    #       + sum_k a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + sum_k a[k,j]*(b[p[k],li] - b[p[k],lj])
    # Let me precompute for each location l:
    #   row_contrib[l] = b[l, p]  (n-vector)
    #   col_contrib[l] = b[p, l]  (n-vector)
    # Then:
    # sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]]) = a[i,:] @ (row_contrib[lj] - row_contrib[li])
    # sum_k a[k,i]*(b[p[k],lj] - b[p[k],li]) = a[:,i] @ (col_contrib[lj] - col_contrib[li])
    #
    # So delta = (a[i,:]-a[j,:]) @ (row_contrib[lj]-row_contrib[li]) 
    #          + (a[:,i]-a[:,j]) @ (col_contrib[lj]-col_contrib[li])
    #
    # Precompute row_contrib and col_contrib as n x n matrices.
    # Then for each pair, delta is two dot products of n-vectors = O(n).
    # 11k pairs * 150 = 1.6M multiply-adds per pass. In numpy, we can vectorize over pairs.

    # Let me precompute:
    #   RC = b[:, p]  # shape (n, n), RC[l, k] = b[l, p[k]]
    #   CC = b[p, :]  # shape (n, n), CC[k, l] = b[p[k], l]
    # Then row_contrib[l] = RC[l, :]
    #      col_contrib[l] = CC[:, l]
    #
    # For swap of i,j with li=p[i], lj=p[j]:
    #   drow = RC[lj, :] - RC[li, :]   (n-vector)
    #   dcol = CC[:, lj] - CC[:, li]   (n-vector)
    #   delta = (a[i,:]-a[j,:]) @ drow + (a[:,i]-a[:,j]) @ dcol

    # Vectorize over all pairs:
    # Create arrays of all (i,j) pairs with i<j.
    # For each pair, li = p[i], lj = p[j].
    # drow = RC[lj] - RC[li]  -> shape (num_pairs, n)
    # dcol = CC[:, lj] - CC[:, li] -> shape (num_pairs, n)
    # a_diff_row = a[i] - a[j]  -> shape (num_pairs, n)
    # a_diff_col = a[:,i] - a[:,j] -> shape (num_pairs, n)
    # delta = sum(a_diff_row * drow, axis=1) + sum(a_diff_col * dcol, axis=1)
    # This is O(n^2 * n) = O(n^3) memory and time per pass, which for n=150 is 150^3 = 3.4M,
    # times 2 for the two terms = 6.8M, which is feasible in numpy.

    # But we need to do this iteratively (accepting swaps), so we can't fully vectorize across passes.
    # Let's do it in chunks or just use a simpler approach.

    # Actually, let me just do a simple loop-based 2-opt with numpy for the delta computation.
    # For each pair, compute delta in O(n) using numpy dot. 11k pairs * ~100ns per dot = ~1ms per pass.
    # That should be fast enough for many passes.

    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                li = p[i]
                lj = p[j]
                # Compute delta
                drow = b[lj, p] - b[li, p]  # n-vector
                dcol = b[p, lj] - b[p, li]  # n-vector
                delta = np.dot(a[i, :] - a[j, :], drow) + np.dot(a[:, i] - a[:, j], dcol)
                if delta < 0:
                    # Accept swap
                    p[i], p[j] = lj, li
                    cost += delta
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
        if not improved:
            break
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    # Step 1: MDS embedding + Procrustes alignment
    # Symmetrize for MDS
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    # Choose embedding dimension
    k = min(20, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    # Procrustes alignment
    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: Local search polish
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _two_opt_search(a, b, p, time_left)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END
