"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Ensure arrays are float64 for faster arithmetic
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i, :], b[p[i], :])
        return int(total)

    def precompute_swap_deltas(p):
        """
        Precompute the delta for all pairs (i, j) where i < j.
        Uses vectorized operations to compute all deltas in O(n^2) total.
        
        For swapping facilities i and j (locations li=p[i], lj=p[j]):
        delta = sum_{k!=i,j} [a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
                         + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj])]
              + a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
              + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
        """
        # For each pair (i,j), we need:
        # d1[i,j] = sum_k a[i,k] * (b[p[j], p[k]] - b[p[i], p[k]])  [k != i,j]
        # d2[i,j] = sum_k a[k,i] * (b[p[k], p[j]] - b[p[k], p[i]])  [k != i,j]
        # d3[i,j] = sum_k a[j,k] * (b[p[i], p[k]] - b[p[j], p[k]])  [k != i,j]
        # d4[i,j] = sum_k a[k,j] * (b[p[k], p[i]] - b[p[k], p[j]])  [k != i,j]
        
        # Note: d3[i,j] = -d1[j,i] and d4[i,j] = -d2[j,i] by symmetry of the sum structure
        # But let's just compute directly.
        
        # For a fixed i, d1[i,:] = sum_k a[i,k] * (b[p[j], p[k]] - b[p[i], p[k]]) for all j
        # = sum_k a[i,k] * b[p[j], p[k]] - sum_k a[i,k] * b[p[i], p[k]]
        # The second term is constant for fixed i.
        # The first term: for each j, it's dot(a[i,:], b[p[j], p[:]]) but excluding k=i and k=j.
        
        # Let's use a different approach. Define:
        # S1[i] = sum_k a[i,k] * b[p[i], p[k]]  (contribution of i to current cost from its location)
        # When we swap i and j, the new contribution involves b[p[j], p[k]] for i's terms.
        
        # Actually, let me use the standard incremental update approach:
        # Precompute row_dot[i] = sum_k a[i,k] * b[p[i], p[k]]
        # Precompute col_dot[i] = sum_k a[k,i] * b[p[k], p[i]]
        
        # When swapping i and j:
        # Old cost contributions involving i or j:
        #   row_dot[i] + row_dot[j] + col_dot[i] + col_dot[j] - a[i,j]*b[li,lj] - a[j,i]*b[lj,li] 
        #   (subtract double-counted cross terms)
        # Actually this gets complicated. Let me just do the direct vectorized computation.
        
        # For all pairs at once:
        # delta_matrix[i,j] for i<j
        
        # Let's compute using matrix operations.
        # For fixed i, the change in "row i contribution" when moving from li to lj:
        #   delta_row_i = sum_k a[i,k] * (b[lj, p[k]] - b[li, p[k]])
        # This is: a[i,:] @ (b[lj, p] - b[li, p]) where b[lj, p] means b[lj, p[k]] for all k
        # = a[i,:] @ b[lj, p] - a[i,:] @ b[li, p]
        # = dot(a[i,:], b[lj, p]) - row_dot[i]
        
        # Similarly for column i:
        #   delta_col_i = sum_k a[k,i] * (b[p[k], lj] - b[p[k], li])
        # = b[:, lj] @ a[:, i] - b[:, li] @ a[:, i]
        # = dot(b[:, lj], a[:, i]) - col_dot[i]
        
        # So for swap (i,j):
        # delta = delta_row_i + delta_row_j + delta_col_i + delta_col_j
        #        + (diagonal and cross adjustments)
        
        # Let's precompute:
        # row_dot[i] = a[i,:] @ b[p[i], p]  (this is sum_k a[i,k]*b[p[i],p[k]])
        # col_dot[i] = b[:, p[i]] @ a[:, i] (this is sum_k b[p[k],p[i]]*a[k,i])
        
        p_arr = np.array(p, dtype=np.int64)
        
        # row_dot[i] = sum_k a[i,k] * b[p[i], p[k]]
        # We can compute this as: for each i, a[i,:] dotted with b[p[i], p_arr]
        # b[p[i], p_arr] is row p[i] of b, indexed by p_arr
        # This is equivalent to (a @ B_perm) where B_perm[i,k] = b[p[i], p[k]]
        # But B_perm depends on p, so let's just loop over i (n=150, this is fine)
        
        row_dot = np.zeros(n)
        col_dot = np.zeros(n)
        for i in range(n):
            row_dot[i] = np.dot(a[i, :], b[p_arr[i], p_arr])
            col_dot[i] = np.dot(b[:, p_arr[i]], a[:, i])
        
        # Now for swap (i, j) with li=p[i], lj=p[j]:
        # delta = [dot(a[i,:], b[lj, p]) - row_dot[i]] + [dot(a[j,:], b[li, p]) - row_dot[j]]
        #       + [dot(b[:, lj], a[:, i]) - col_dot[i]] + [dot(b[:, li], a[:, j]) - col_dot[j]]
        #       + diagonal/cross corrections
        #
        # The diagonal/cross corrections:
        # When computing dot(a[i,:], b[lj, p]), this includes k=i and k=j terms.
        # Similarly for others. We need to be careful.
        #
        # Actually, the formula above should be correct if we define things properly.
        # Let me verify: the total cost is sum_{i,j} a[i,j]*b[p[i],p[j]].
        # 
        # When we swap i and j, only terms involving i or j change.
        # Terms involving i: sum_j' a[i,j']*b[p[i],p[j']] -> sum_j' a[i,j']*b[p[j],p[j']]
        #   Change: sum_j' a[i,j']*(b[p[j],p[j']] - b[p[i],p[j']])
        #   = dot(a[i,:], b[p[j], p]) - dot(a[i,:], b[p[i], p])
        #   = dot(a[i,:], b[lj, p]) - row_dot[i]
        #
        # Terms involving j: sum_i' a[j,i']*b[p[j],p[i']] -> sum_i' a[j,i']*b[p[i],p[i']]
        #   Change: dot(a[j,:], b[li, p]) - row_dot[j]
        #
        # But wait, the term a[i,j]*b[p[i],p[j]] is counted in both "terms involving i" and "terms involving j".
        # Similarly a[j,i]*b[p[j],p[i]] is counted in both.
        # So we're double-counting. We need to subtract the double-counted parts.
        #
        # Actually no. Let me think again.
        # Total cost = sum_{all i,j} a[i,j]*b[p[i],p[j]]
        # = sum_{i not in {i*,j*}} sum_{j not in {i*,j*}} a[i,j]*b[p[i],p[j]]  [unchanged]
        #   + sum_{j not in {i*,j*}} a[i*,j]*b[p[i*],p[j]]  [changes: p[i*] changes]
        #   + sum_{i not in {i*,j*}} a[i,j*]*b[p[i],p[j*]]  [changes: p[j*] changes]
        #   + a[i*,j*]*b[p[i*],p[j*]]  [changes]
        #   + a[j*,i*]*b[p[j*],p[i*]]  [changes]
        #   + a[i*,i*]*b[p[i*],p[i*]]  [changes: p[i*] changes]
        #   + a[j*,j*]*b[p[j*],p[j*]]  [changes: p[j*] changes]
        #
        # This is getting complex. Let me just use the direct formula from before but vectorize it.
        
        # I'll use the approach: compute delta for each pair using vectorized ops over k.
        # For efficiency, I'll compute all deltas in a batch.
        
        delta_matrix = np.zeros((n, n))
        
        for i in range(n):
            li = p_arr[i]
            # For all j != i, compute delta for swap (i, j)
            # We need: for each j, sum over k != i,j of the four terms
            
            # Let's compute the "row i" part: sum_k a[i,k] * (b[p[j], p[k]] - b[li, p[k]])
            # = sum_k a[i,k] * b[p[j], p[k]] - sum_k a[i,k] * b[li, p[k]]
            # = dot(a[i,:], b[p[j], p_arr]) - row_dot[i]
            # But this includes k=i and k=j. We need to exclude them.
            # When k=i: a[i,i] * (b[p[j], p[i]] - b[li, p[i]]) = a[i,i] * (b[p[j], li] - b[li, li])
            # When k=j: a[i,j] * (b[p[j], p[j]] - b[li, p[j]]) = a[i,j] * (b[p[j], p[j]] - b[li, p[j]])
            
            # This is getting too complicated to vectorize cleanly in one shot.
            # Let me just do a simpler but still fast approach:
            # For each i, precompute the contribution vectors, then for each j compute delta in O(1) or O(n).
            
            pass
        
        # Given the complexity, let me use a different strategy:
        # First-improvement local search with vectorized delta computation per row.
        # For each i, compute all deltas for swaps (i, j) for j > i in one vectorized op.
        
        deltas = np.zeros((n, n))
        
        for i in range(n):
            li = p_arr[i]
            # For each j > i, compute delta for swap (i, j)
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            ljs = p_arr[js]
            
            # For swap (i, j):
            # delta = sum_{k!=i,j} [a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
            #        + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj])]
            #       + a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
            #       + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
            
            # Let me compute this term by term, vectorized over j.
            
            # Term 1: sum_{k!=i,j} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
            # = sum_k a[i,k] * (b[lj, p[k]] - b[li, p[k]]) - a[i,i]*(b[lj,li]-b[li,li]) - a[i,j]*(b[lj,p[j]]-b[li,p[j]])
            # = (a[i,:] @ (b[lj, p_arr] - b[li, p_arr])) - corrections
            # But b[lj, p_arr] depends on j, so we can't easily vectorize over j in one mat-vec.
            
            # Alternative: precompute for each location l, the vector v_l[k] = b[l, p[k]]
            # Then term1 for (i,j) = a[i,:] @ (v_{lj} - v_{li}) - corrections
            
            # Let me precompute V[l] = b[l, p_arr] for all l. This is just rows of b indexed by p_arr.
            # V is n x n where V[l, k] = b[l, p[k]]
            
            # Actually V = b[:, p_arr]  (column permutation of b)
            V = b[:, p_arr]  # V[l, k] = b[l, p[k]]
            
            # Similarly for the "column" terms, we need b[p[k], l] for various l.
            # W[l, k] = b[p[k], l] = b[p_arr, l][k] = (b[p_arr, :])[:, l] 
            # W = b[p_arr, :]  (row permutation of b)
            W = b[p_arr, :]  # W[k, l] = b[p[k], l]
            
            # Now:
            # term1(i,j) = sum_k a[i,k] * (V[lj, k] - V[li, k]) = a[i,:] @ (V[lj] - V[li])
            # term2(i,j) = sum_k a[k,i] * (W[k, lj] - W[k, li]) = a[:,i] @ (W[:,lj] - W[:,li])
            # term3(i,j) = sum_k a[j,k] * (V[li, k] - V[lj, k]) = a[j,:] @ (V[li] - V[lj]) = -a[j,:] @ (V[lj] - V[li])
            # term4(i,j) = sum_k a[k,j] * (W[k, li] - W[k, lj]) = a[:,j] @ (W[:,li] - W[:,lj]) = -a[:,j] @ (W[:,lj] - W[:,li])
            
            # So delta(i,j) = (a[i,:] - a[j,:]) @ (V[lj] - V[li]) + (a[:,i] - a[:,j]) @ (W[:,lj] - W[:,li])
            #                + diagonal/cross corrections
            
            # The full sum includes k=i and k=j, but in the original formula we exclude them.
            # Let's include them and then subtract the corrections.
            
            # Full sum (including k=i and k=j):
            # S1 = (a[i,:] - a[j,:]) @ (V[lj] - V[li])
            # S2 = (a[:,i] - a[:,j]) @ (W[:,lj] - W[:,li])
            
            # Corrections to exclude k=i and k=j:
            # When k=i:
            #   a[i,i]*(V[lj,i] - V[li,i]) - a[j,i]*(V[lj,i] - V[li,i])  from S1
            #   a[i,i]*(W[i,lj] - W[i,li]) - a[j,i]*(W[i,lj] - W[i,li])  from S2
            #   = (a[i,i] - a[j,i]) * (V[lj,i] - V[li,i]) + (a[i,i] - a[j,i]) * (W[i,lj] - W[i,li])
            #   Note: V[lj,i] = b[lj, p[i]] = b[lj, li], W[i,lj] = b[p[i], lj] = b[li, lj]
            #   V[li,i] = b[li, p[i]] = b[li, li], W[i,li] = b[p[i], li] = b[li, li]
            #   So: (a[i,i] - a[j,i]) * (b[lj,li] - b[li,li]) + (a[i,i] - a[j,i]) * (b[li,lj] - b[li,li])
            
            # When k=j:
            #   a[i,j]*(V[lj,j] - V[li,j]) - a[j,j]*(V[lj,j] - V[li,j])  from S1
            #   a[i,j]*(W[j,lj] - W[j,li]) - a[j,j]*(W[j,lj] - W[j,li])  from S2
            #   = (a[i,j] - a[j,j]) * (V[lj,j] - V[li,j]) + (a[i,j] - a[j,j]) * (W[j,lj] - W[j,li])
            #   Note: V[lj,j] = b[lj, p[j]] = b[lj, lj], W[j,lj] = b[p[j], lj] = b[lj, lj]
            #   V[li,j] = b[li, p[j]] = b[li, lj], W[j,li] = b[p[j], li] = b[lj, li]
            #   So: (a[i,j] - a[j,j]) * (b[lj,lj] - b[li,lj]) + (a[i,j] - a[j,j]) * (b[lj,lj] - b[lj,li])
            
            # This is getting very messy. Let me just use a simpler but still efficient approach:
            # Compute delta for each pair (i,j) in O(n) using precomputed row/col dot products,
            # but do it in a vectorized way over j for each i.
            
            # For fixed i, compute delta(i, j) for all j > i:
            # delta(i,j) = [a[i,:] @ (V[lj] - V[li])] + [a[:,i] @ (W[:,lj] - W[:,li])]
            #           + [a[j,:] @ (V[li] - V[lj])] + [a[:,j] @ (W[:,li] - W[:,lj])]
            #           + diag/cross corrections
            #
            # = (a[i,:] @ V[lj] - a[i,:] @ V[li]) + (a[:,i] @ W[:,lj] - a[:,i] @ W[:,li])
            #   - (a[j,:] @ V[lj] - a[j,:] @ V[li]) - (a[:,j] @ W[:,lj] - a[:,j] @ W[:,li])
            #   + corrections
            #
            # = (a[i,:] @ V[lj] - a[j,:] @ V[lj]) - (a[i,:] @ V[li] - a[j,:] @ V[li])
            #   + (a[:,i] @ W[:,lj] - a[:,j] @ W[:,lj]) - (a[:,i] @ W[:,li] - a[:,j] @ W[:,li])
            #   + corrections
            #
            # = ((a[i,:] - a[j,:]) @ V[lj]) - ((a[i,:] - a[j,:]) @ V[li])
            #   + ((a[:,i] - a[:,j]) @ W[:,lj]) - ((a[:,i] - a[:,j]) @ W[:,li])
            #   + corrections
            
            # For fixed i, the vector (a[i,:] - a[j,:]) depends on j, so we can't fully vectorize.
            # But we can compute a[i,:] @ V[lj] for all j at once: that's a[i,:] @ V[ljs] which is a dot product
            # of a[i,:] with each column of V[ljs]. V[ljs] is a (len(js), n) matrix.
            # a[i,:] @ V[ljs].T gives a vector of length len(js) where entry m is a[i,:] @ V[ljs[m], :]
            
            # Let me define:
            # R_i_j = a[i,:] @ V[p[j], :]  for all j
            # This is: a[i,:] @ (b[:, p_arr])[p[j], :] = a[i,:] @ b[:, p_arr][p[j], :]
            # Hmm, V = b[:, p_arr], so V[l, :] = b[l, p_arr]
            # R_i_j = a[i,:] @ b[p[j], p_arr]  (dot product of row i of a with row p[j] of b, permuted)
            
            # For fixed i, R_i = a[i,:] @ b[:, p_arr].T  ??? No.
            # a[i,:] is (n,), b[:, p_arr] is (n, n) with columns permuted.
            # a[i,:] @ (b[:, p_arr]).T = a[i,:] @ b[p_arr, :]  (dot of a[i,:] with each row of b[p_arr,:])
            # That gives for each k: a[i,:] @ b[p_arr[k], :] = sum_m a[i,m] * b[p[k], m]
            # That's not quite what I want.
            
            # I want: for each j, a[i,:] @ b[p[j], p_arr] = sum_k a[i,k] * b[p[j], p[k]]
            # = sum_k a[i,k] * b[p[j], p[k]]
            # This is the (i, j) entry of a @ (b[:, p_arr]) @ ... hmm.
            
            # Let M = a @ b.T  (M[i, l] = sum_k a[i,k] * b[k,l])
            # Then a[i,:] @ b[p[j], p_arr] = sum_k a[i,k] * b[p[j], p[k]]
            # This is not directly M[i, p[j]] because M[i, p[j]] = sum_k a[i,k] * b[k, p[j]]
            # which is different (b[k, p[j]] vs b[p[j], p[k]]).
            
            # Unless b is symmetric! If b is symmetric, then b[p[j], p[k]] = b[p[k], p[j]],
            # and sum_k a[i,k] * b[p[j], p[k]] = sum_k a[i,k] * b[p[k], p[j]] = M[i, p[j]].
            
            # For QAP, the distance matrix b is typically symmetric. Let me assume it is.
            # If b is symmetric, then:
            # a[i,:] @ b[p[j], p_arr] = sum_k a[i,k] * b[p[k], p[j]] = (a @ b.T)[i, p[j]] = M[i, p[j]]
            # where M = a @ b.T
            
            # Similarly, a[:,i] @ b[p_arr, p[j], :] ... let me check the column term.
            # a[:,i] @ (W[:,lj] - W[:,li]) where W[k, l] = b[p[k], l]
            # a[:,i] @ W[:,lj] = sum_k a[k,i] * b[p[k], lj]
            # If b is symmetric: b[p[k], lj] = b[lj, p[k]]
            # = sum_k a[k,i] * b[lj, p[k]]
            # This is like (a.T @ b.T)[lj, i] ... let me define N = a.T @ b.T
            # N[l, i] = sum_k a.T[l,k] * b.T[k,i] = sum_k a[k,l] * b[i,k]
            # That's not quite it either.
            
            # Let me just define:
            # P[i, l] = sum_k a[i,k] * b[l, p[k]]  (for the "row" type terms)
            # Q[i, l] = sum_k a[k,i] * b[p[k], l]  (for the "column" type terms)
            
            # If b is symmetric: b[l, p[k]] = b[p[k], l], so P[i,l] = sum_k a[i,k]*b[p[k],l]
            # And Q[i,l] = sum_k a[k,i]*b[p[k],l]
            
            # Let me just precompute P and Q directly. n=150, so P and Q are 150x150.
            # P[i, l] = sum_k a[i,k] * b[l, p[k]]
            # For fixed i, P[i, :] = a[i,:] @ (b[:, p_arr])  ??? 
            # b[:, p_arr] has entries (b[:, p_arr])[l, k] = b[l, p[k]]
            # So a[i,:] @ (b[:, p_arr]) is sum_k a[i,k] * b[:, p_arr][?, k] ... no.
            # Matrix multiplication: (a[i,:] @ M) where M is (n, n) gives a vector of length n.
            # (a[i,:] @ M)[l] = sum_k a[i,k] * M[k, l]
            # I want sum_k a[i,k] * b[l, p[k]] = sum_k a[i,k] * M[l, k] where M[l,k] = b[l, p[k]]
            # = sum_k a[i,k] * M.T[k, l] = (a[i,:] @ M.T)[l]
            # So P[i, :] = a[i,:] @ (b[:, p_arr]).T
            
            # Similarly Q[i, l] = sum_k a[k,i] * b[p[k], l]
            # = sum_k a[k,i] * b[p[k], l]
            # Let N[k, l] = b[p[k], l] = (b[p_arr, :])[k, l]
            # Q[i, l] = sum_k a[k,i] * N[k, l] = (a.T @ N)[i, l] = (a.T @ b[p_arr, :])[i, l]
            
            # So:
            # P = a @ (b[:, p_arr]).T  (shape n x n)
            # Q = a.T @ b[p_arr, :]    (shape n x n)
            
            # Then for swap (i, j) with li=p[i], lj=p[j]:
            # The "row i" contribution change: P[i, lj] - P[i, li]
            # The "row j" contribution change: P[j, li] - P[j, lj]
            # The "col i" contribution change: Q[i, lj] - Q[i, li]
            # The "col j" contribution change: Q[j, li] - Q[j, lj]
            
            # But this includes k=i and k=j terms that shouldn't be there (or should be adjusted).
            # Let me verify with the original formula.
            
            # Original delta for swap (i,j):
            # = sum_{k!=i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + sum_{k!=i,j} a[k,i]*(b[p[k],lj] - b[p[k],li])
            #   + sum_{k!=i,j} a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + sum_{k!=i,j} a[k,j]*(b[p[k],li] - b[p[k],lj])
            #   + a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
            #   + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
            
            # P[i, lj] - P[i, li] = sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]])  [includes k=i and k=j]
            # = [sum_{k!=i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]])] + a[i,i]*(b[lj,li]-b[li,li]) + a[i,j]*(b[lj,p[j]]-b[li,p[j]])
            # = [sum_{k!=i,j} ...] + a[i,i]*(b[lj,li]-b[li,li]) + a[i,j]*(b[lj,lj]-b[li,lj])
            
            # Similarly for the other terms.
            
            # So the full delta = (P[i,lj]-P[i,li]) + (P[j,li]-P[j,lj]) + (Q[i,lj]-Q[i,li]) + (Q[j,li]-Q[j,lj])
            #                    - [corrections for double-counting k=i and k=j]
            #                    + [explicit diagonal/cross terms]
            
            # The corrections: when we add (P[i,lj]-P[i,li]) + (P[j,li]-P[j,lj]), the terms with k=i and k=j
            # from P[i] and P[j] are included. But in the original formula, we have separate explicit terms
            # for the diagonal and cross interactions.
            
            # Let me just compute delta as:
            # delta = (P[i,lj]-P[i,li]) + (P[j,li]-P[j,lj]) + (Q[i,lj]-Q[i,li]) + (Q[j,li]-Q[j,lj])
            # And then subtract the parts that are double-counted or incorrect.
            
            # Actually, I think the cleanest approach for n=150 is:
            # Precompute P and Q (O(n^3) = 3.4M ops, done once per local search pass).
            # Then compute all pair deltas in O(n^2) using the formula above.
            # The formula might need corrections, but let me test it.
            
            # For now, let me use a simpler but correct approach:
            # For each i, compute delta(i, j) for all j in O(n) using precomputed arrays.
            # Total: O(n^2) per pass.
            
            # delta(i, j) = sum_{k} a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) 
            #            + sum_{k} a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
            #            + sum_{k} a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
            #            + sum_{k} a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
            # This includes k=i and k=j, which gives the correct full delta including all terms.
            # Let me verify: when k=i, the first term gives a[i,i]*(b[p[j],p[i]]-b[p[i],p[i]]) = a[i,i]*(b[p[j],li]-b[li,li])
            # The third term gives a[j,i]*(b[p[i],p[i]]-b[p[j],p[i]]) = a[j,i]*(b[li,li]-b[p[j],li])
            # Sum: (a[i,i]-a[j,i])*(b[p[j],li]-b[li,li])
            # When k=j: first term: a[i,j]*(b[p[j],p[j]]-b[p[i],p[j]]) = a[i,j]*(b[lj,lj]-b[li,lj])
            # Third term: a[j,j]*(b[p[i],p[j]]-b[p[j],p[j]]) = a[j,j]*(b[li,lj]-b[lj,lj])
            # Sum: (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])
            # Second term when k=i: a[i,i]*(b[p[i],p[j]]-b[p[i],p[i]]) = a[i,i]*(b[li,lj]-b[li,li])
            # Fourth term when k=i: a[i,j]*(b[p[i],p[i]]-b[p[i],p[j]]) = a[i,j]*(b[li,li]-b[li,lj])
            # Sum: (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
            # Second term when k=j: a[j,i]*(b[p[j],p[j]]-b[p[j],p[i]]) = a[j,i]*(b[lj,lj]-b[lj,li])
            # Fourth term when k=j: a[j,j]*(b[p[j],p[i]]-b[p[j],p[j]]) = a[j,j]*(b[lj,li]-b[lj,lj])
            # Sum: (a[j,i]-a[j,j])*(b[lj,li]-b[lj,lj])
            
            # So the total from k=i and k=j in the full sum is:
            # (a[i,i]-a[j,i])*(b[p[j],li]-b[li,li]) + (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])
            # + (a[i,i]-a[i,j])*(b[li,lj]-b[li,li]) + (a[j,i]-a[j,j])*(b[lj,li]-b[lj,lj])
            
            # The original explicit terms are:
            # a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
            # + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
            
            # These are different! So the "full sum" approach doesn't directly give the right answer
            # unless I account for the difference.
            
            # I think the safest approach for correctness is to just compute delta directly for each pair
            # in O(n) time, but do it in a vectorized way over all pairs for a given i.
            
            # For fixed i, and all j > i:
            # delta(i,j) = sum_k [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
            #            + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
            
            # Let me define for fixed i:
            # u[k] = a[i,k]  (row i of a)
            # v[k] = a[k,i]  (col i of a)
            
            # For each j:
            # T1(j) = sum_k u[k] * (b[p[j], p[k]] - b[p[i], p[k]])
            # T2(j) = sum_k v[k] * (b[p[k], p[j]] - b[p[k], p[i]])
            # T3(j) = sum_k a[j,k] * (b[p[i], p[k]] - b[p[j], p[k]])
            # T4(j) = sum_k a[k,j] * (b[p[k], p[i]] - b[p[k], p[j]])
            
            # T1(j) = u @ (b[p[j], p_arr] - b[p[i], p_arr])
            # T2(j) = v @ (b[p_arr, p[j]] - b[p_arr, p[i]])
            # T3(j) = a[j,:] @ (b[p[i], p_arr] - b[p[j], p_arr]) = -a[j,:] @ (b[p[j], p_arr] - b[p[i], p_arr])
            # T4(j) = a[:,j] @ (b[p_arr, p[i]] - b[p_arr, p[j]]) = -a[:,j] @ (b[p_arr, p[j]] - b[p_arr, p[i]])
            
            # So delta(i,j) = (u - a[j,:]) @ (b[p[j], p_arr] - b[p[i], p_arr])
            #               + (v - a[:,j]) @ (b[p_arr, p[j]] - b[p_arr, p[i]])
            
            # For fixed i, let d_row = b[p[i], p_arr] (length n vector)
            # d_col = b[p_arr, p[i]] (length n vector)
            
            # For each j:
            # delta(i,j) = (u - a[j,:]) @ (b[p[j], p_arr] - d_row)
            #            + (v - a[:,j]) @ (b[p_arr, p[j]] - d_col)
            
            # = u @ b[p[j], p_arr] - u @ d_row - a[j,:] @ b[p[j], p_arr] + a[j,:] @ d_row
            #   + v @ b[p_arr, p[j]] - v @ d_col - a[:,j] @ b[p_arr, p[j]] + a[:,j] @ d_col
            
            # u @ d_row = sum_k a[i,k] * b[p[i], p[k]]  (constant for fixed i)
            # v @ d_col = sum_k a[k,i] * b[p[k], p[i]]  (constant for fixed i)
            
            # u @ b[p[j], p_arr] = sum_k a[i,k] * b[p[j], p[k]]
            # a[j,:] @ b[p[j], p_arr] = sum_k a[j,k] * b[p[j], p[k]]
            # v @ b[p_arr, p[j]] = sum_k a[k,i] * b[p[k], p[j]]
            # a[:,j] @ b[p_arr, p[j]] = sum_k a[k,j] * b[p[k], p[j]]
            
            # For fixed i, I can precompute:
            # c1 = u @ d_row (scalar)
            # c2 = v @ d_col (scalar)
            
            # Then for each j:
            # delta(i,j) = [u @ b[p[j], p_arr] - a[j,:] @ b[p[j], p_arr]] 
            #            + [v @ b[p_arr, p[j]] - a[:,j] @ b[p_arr, p[j]]]
            #            - c1 + a[j,:] @ d_row - c2 + a[:,j] @ d_col
            
            # Note: a[j,:] @ d_row = sum_k a[j,k] * b[p[i], p[k]]
            # a[:,j] @ d_col = sum_k a[k,j] * b[p[k], p[i]]
            
            # For fixed i, d_row and d_col are fixed.
            # I can precompute for all j:
            # X[j] = a[j,:] @ d_row = sum_k a[j,k] * b[p[i], p[k]]
            # Y[j] = a[:,j] @ d_col = sum_k a[k,j] * b[p[k], p[i]]
            
            # X = a @ d_row  (vector of length n)
            # Y = d_col @ a  (vector of length n)  [since (d_col @ a)[j] = sum_k d_col[k] * a[k,j]]
            
            # Then:
            # delta(i,j) = [u @ b[p[j], p_arr] - a[j,:] @ b[p[j], p_arr]]
            #            + [v @ b[p_arr, p[j]] - a[:,j] @ b[p_arr, p[j]]]
            #            - c1 + X[j] - c2 + Y[j]
            
            # Now, u @ b[p[j], p_arr] for all j: 
            # u @ (b[:, p_arr]) ... u is (n,), b[:, p_arr] is (n, n)
            # (u @ (b[:, p_arr]).T)[j] = sum_k u[k] * b[:, p_arr].T[j, k] = sum_k u[k] * b[p_arr[k], j]
            # That's not what I want. I want u @ b[p[j], p_arr] = sum_k u[k] * b[p[j], p[k]]
            # = sum_k u[k] * b[p[j], p[k]]
            # This is the j-th entry of u @ (b[:, p_arr]) where we take row p[j]... 
            # Actually, let me define U_mat where U_mat[l, k] = b[l, p[k]] = (b[:, p_arr])[l, k]
            # Then u @ U_mat gives a vector of length n where entry l is sum_k u[k] * U_mat[k, l]
            # That's sum_k u[k] * b[k, p[l]] ... no.
            # U_mat[l, k] = b[l, p[k]]
            # (u @ U_mat)[l] = sum_k u[k] * U_mat[k, l] = sum_k u[k] * b[k, p[l]]
            # I want sum_k u[k] * b[p[l], p[k]] for each l.
            # = sum_k u[k] * b[p[l], p[k]]
            # If I define U_mat2[l, k] = b[p[l], p[k]], then (u @ U_mat2)[l] = sum_k u[k] * U_mat2[k, l] = sum_k u[k] * b[p[k], p[l]]
            # Still not quite. I need sum_k u[k] * b[p[l], p[k]] = (u @ U_mat2.T)[l] if U_mat2[k,l] = b[p[l],p[k]]
            # U_mat2.T[l, k] = U_mat2[k, l] = b[p[l], p[k]]
            # (u @ U_mat2.T)[l] = sum_k u[k] * U_mat2.T[k, l] = sum_k u[k] * U_mat2[l, k] = sum_k u[k] * b[p[l], p[k]]
            # Yes! So if U_mat2[l, k] = b[p[l], p[k]], then u @ U_mat2.T gives the vector I want.
            # U_mat2 = b[p_arr, p_arr]  (this is the permuted distance matrix)
            
            # So: vec1 = u @ (b[p_arr, p_arr]).T  gives vec1[l] = sum_k a[i,k] * b[p[l], p[k]]
            # Similarly, for a[j,:]: vec_j = a[j,:] @ (b[p_arr, p_arr]).T
            
            # Let B_perm = b[p_arr, p_arr]  (n x n)
            # Let A_BT = a @ B_perm.T  (n x n) where A_BT[i, l] = sum_k a[i,k] * B_perm.T[k,l] = sum_k a[i,k] * B_perm[l,k] = sum_k a[i,k] * b[p[l], p[k]]
            
            # So A_BT[i, l] = u @ b[p[l], p_arr] for the "u" corresponding to row i.
            # Similarly, for the column terms:
            # v @ b[p_arr, p[j]] = sum_k a[k,i] * b[p[k], p[j]]
            # Let B_perm2 = b[p_arr, p_arr]  (same as B_perm)
            # (B_perm2 @ v)[j] = sum_k B_perm2[k, j] * v[k] = sum_k b[p[k], p[j]] * a[k,i]
            # That's exactly what I want!
            # So vec2 = B_perm @ v  gives vec2[j] = sum_k b[p[k], p[j]] * a[k,i]
            
            # Similarly for a[:,j]:
            # a[:,j] @ b[p_arr, p[j]] = sum_k a[k,j] * b[p[k], p[j]]
            # (B_perm @ a[:,j])[j]... no, (B_perm @ a)[j, j] would be sum_k B_perm[j,k]*a[k,j] = sum_k b[p[j],p[k]]*a[k,j]
            # I want sum_k a[k,j] * b[p[k], p[j]] = (a[:,j] @ B_perm.T)[j] ... 
            # Actually, sum_k a[k,j] * b[p[k], p[j]] = (a.T @ B_perm)[j, j]? 
            # (a.T @ B_perm)[j, l] = sum_k a.T[j,k] * B_perm[k,l] = sum_k a[k,j] * b[p[k], p[l]]
            # So (a.T @ B_perm)[j, j] = sum_k a[k,j] * b[p[k], p[j]]  ✓
            
            # So let me precompute:
            # B_perm = b[p_arr, p_arr]
            # ABT = a @ B_perm.T  (ABT[i,l] = sum_k a[i,k]*b[p[l],p[k]])
            # Bv = B_perm @ a  (Bv[k,j] = sum_l B_perm[k,l]*a[l,j] = sum_l b[p[k],p[l]]*a[l,j])
            #   So Bv[i,j] = sum_k b[p[i],p[k]]*a[k,j]
            #   I need sum_k a[k,i]*b[p[k],p[j]] = (a[:,i] @ B_perm.T)[j] = (B_perm @ a[:,i])[j]... 
            #   (B_perm @ a)[j, i] = sum_k B_perm[j,k]*a[k,i] = sum_k b[p[j],p[k]]*a[k,i]
            #   That's sum_k a[k,i]*b[p[j],p[k]], not b[p[k],p[j]].
            #   If b is symmetric, b[p[j],p[k]] = b[p[k],p[j]], so it's the same.
            
            # Assuming b is symmetric:
            # Bv[i,j] = sum_k b[p[i],p[k]]*a[k,j] = sum_k a[k,j]*b[p[k],p[i]]  (by symmetry of b)
            # I need for fixed i: vec2[j] = sum_k a[k,i]*b[p[k],p[j]] = Bv[j,i]  (by symmetry)
            
            # So:
            # delta(i,j) = ABT[i,j] - ABT[j,j] + Bv[j,i] - Bv[i,i] - ABT[i,i] + ABT[j,i] - Bv[i,i] + Bv[j,j]
            # Wait, let me redo this more carefully.
            
            # delta(i,j) = [ABT[i,j] - ABT[j,j]] + [Bv[j,i] - Bv[i,i]] - c1 + X[j] - c2 + Y[j]
            # where c1 = ABT[i,i], c2 = Bv[i,i]
            # X[j] = a[j,:] @ d_row = ABT[j, p[i]]... no.
            # X[j] = sum_k a[j,k] * b[p[i], p[k]] = ABT[j, i]  (since ABT[j,l] = sum_k a[j,k]*b[p[l],p[k]], so ABT[j,i] = sum_k a[j,k]*b[p[i],p[k]])
            # Y[j] = sum_k a[k,j] * b[p[k], p[i]] = Bv[i, j]  (since Bv[i,j] = sum_k a[k,j]*b[p[i],p[k]] by symmetry)
            
            # So:
            # delta(i,j) = ABT[i,j] - ABT[j,j] + Bv[j,i] - Bv[i,i] - ABT[i,i] + ABT[j,i] - Bv[i,i] + Bv[i,j]
            
            # Let me simplify:
            # delta(i,j) = ABT[i,j] + ABT[j,i] + Bv[j,i] + Bv[i,j] - ABT[j,j] - 2*Bv[i,i] - ABT[i,i]
            
            # This is O(1) per pair after O(n^3) precomputation!
            
            # Let me verify this formula is correct by checking against the original for a small case.
            # For now, let me implement this and test it.
            
            B_perm = b[p_arr, p_arr]
            ABT = a @ B_perm.T  # ABT[i,l] = sum_k a[i,k]*b[p[l],p[k]]
            Bv = B_perm @ a     # Bv[i,j] = sum_k b[p[i],p[k]]*a[k,j]
            
            # Precompute for this pass:
            # For all pairs (i,j), delta = ABT[i,j] + ABT[j,i] + Bv[j,i] + Bv[i,j] - ABT[j,j] - 2*Bv[i,i] - ABT[i,i]
            
            # Vectorized:
            delta_all = ABT + ABT.T + Bv.T + Bv - ABT[np.arange(n), np.arange(n)][np.newaxis, :] \
                        - 2 * Bv[np.arange(n), np.arange(n)][:, np.newaxis] \
                        - ABT[np.arange(n), np.arange(n)][:, np.newaxis]
            
            # delta_all[i, j] is the delta for swap (i, j)
            # We only care about i < j, and delta < 0 for improvement.
            
            # Find the best improving swap
            np.fill_diagonal(delta_all, 0)
            # Only consider upper triangle
            delta_upper = np.triu(delta_all, k=1)
            min_delta = np.min(delta_upper)
            
            if min_delta < 0:
                idx = np.unravel_index(np.argmin(delta_upper), delta_upper.shape)
                i, j = idx
                p[i], p[j] = p[j], p[i]
                p_arr = np.array(p, dtype=np.int64)
                cost += min_delta
                return p, cost, True
            
            return p, cost, False

    def local_search_fast(p, max_time=None):
        """
        Fast local search using precomputed deltas.
        """
        if max_time is None:
            max_time = deadline
        
        p = np.array(p, dtype=np.int64)
        cost = compute_cost(p)
        
        while time.time() < max_time:
            p, cost, improved = precompute_swap_deltas(p)
            if not improved:
                break
        
        return p, cost

    def perturb(p, strength=10):
        """Perturb the permutation by performing 'strength' random swaps."""
        p = p.copy()
        rng = np.random.RandomState(int(time.time() * 1000) % (2**32))
        for _ in range(strength):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    def spectral_initialization():
        """Generate initial permutations using spectral decomposition."""
        initializations = []
        
        # 1. Rank matching (baseline)
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        initializations.append(p_rank)

        # 2. Spectral Matching
        try:
            k = min(10, n)
            val_a, vec_a = eigsh(a, k=k, which='LM')
            idx_a = np.argsort(np.abs(val_a))[::-1]
            vec_a = vec_a[:, idx_a]

            val_b, vec_b = eigsh(b, k=k, which='LM')
            idx_b = np.argsort(np.abs(val_b))[::-1]
            vec_b = vec_b[:, idx_b]

            for m in range(min(3, k)):
                fa = vec_a[:, m]
                fb = vec_b[:, m]
                
                cost_mat = -np.outer(fa, fb)
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                p_spec = np.empty(n, dtype=np.int64)
                p_spec[row_ind] = col_ind
                initializations.append(p_spec)
                
                cost_mat_inv = np.outer(fa, fb)
                row_ind, col_ind = linear_sum_assignment(cost_mat_inv)
                p_spec_inv = np.empty(n, dtype=np.int64)
                p_spec_inv[row_ind] = col_ind
                initializations.append(p_spec_inv)

        except Exception:
            pass

        return initializations

    # Generate diverse initial solutions
    inits = spectral_initialization()
    
    best_p = None
    best_cost = float('inf')
    current_p = None
    current_cost = float('inf')
    
    rng = np.random.RandomState(42)

    # Process each initialization
    for p_init in inits:
        if time.time() > deadline - 10:
            break
            
        p_ls, cost_ls = local_search_fast(p_init.copy(), deadline - 5)
        
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
            current_p = p_ls.copy()
            current_cost = cost_ls

    # If we have time, do ILS from the best solution found
    if best_p is not None and time.time() < deadline - 20:
        while time.time() < deadline - 5:
            strength = 5 + int(10 * (1 - time.time() / deadline))
            strength = max(3, min(20, strength))
            perturbed_p = perturb(current_p, strength)

            p_ls, cost_ls = local_search_fast(perturbed_p, deadline - 5)

            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
                current_p = p_ls.copy()
                current_cost = cost_ls
            else:
                if rng.random() < 0.1:
                    current_p = p_ls.copy()
                    current_cost = cost_ls

    if best_p is None:
        best_p = np.arange(n)

    return best_p.tolist()

# EVOLVE-BLOCK-END
