"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: sum over i of a[i,:] dot b[p[i], p[i]]? No.
    # cost = trace(p^T b p^T a p)? 
    # Let's use: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # This is equal to the sum of element-wise products of a and (b indexed by p,p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """Compute the delta in cost when swapping facilities at positions i and j."""
    # We swap p[i] and p[j].
    # Let x = p[i], y = p[j].
    # Old contribution involving i or j:
    # New contribution involving i or j:
    # Delta = New - Old
    
    # This can be computed in O(n) per swap.
    # For n=256, O(n) is fine.
    
    x = p[i]
    y = p[j]
    
    # Sum over all k != i, j
    # a[i,k]*b[x,p[k]] + a[k,i]*b[p[k],x] + a[j,k]*b[y,p[k]] + a[k,j]*b[p[k],y]
    # becomes
    # a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y] + a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
    
    # Plus the i-j term: a[i,j]*b[x,y] + a[j,i]*b[y,x] becomes a[i,j]*b[y,x] + a[j,i]*b[x,y]
    
    # Let's vectorize the k-loop.
    # mask = np.ones(n, dtype=bool); mask[i]=mask[j]=False
    
    # Actually, simpler:
    # old = a[i,j]*b[x,y] + a[j,i]*b[y,x]
    # new = a[i,j]*b[y,x] + a[j,i]*b[x,y]
    # diff_ij = new - old
    
    # For k != i, j:
    # old_k = a[i,k]*b[x,p[k]] + a[k,i]*b[p[k],x] + a[j,k]*b[y,p[k]] + a[k,j]*b[p[k],y]
    # new_k = a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y] + a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
    # diff_k = (a[i,k]-a[j,k])*(b[x,p[k]]-b[p[k],y]) + (a[k,i]-a[k,j])*(b[p[k],x]-b[y,p[k]])
    # Wait, let's re-derive carefully.
    
    # Term 1: a[i,k] * b[x, p[k]] -> a[i,k] * b[y, p[k]]  => diff: a[i,k]*(b[x,p[k]] - b[y,p[k]])
    # Term 2: a[k,i] * b[p[k], x] -> a[k,i] * b[p[k], y]  => diff: a[k,i]*(b[p[k],x] - b[p[k],y])
    # Term 3: a[j,k] * b[y, p[k]] -> a[j,k] * b[x, p[k]]  => diff: a[j,k]*(b[y,p[k]] - b[x,p[k]])
    # Term 4: a[k,j] * b[p[k], y] -> a[k,j] * b[p[k], x]  => diff: a[k,j]*(b[p[k],y] - b[p[k],x])
    
    # Grouping terms:
    # b[x,p[k]] - b[y,p[k]] appears in Term 1 with +a[i,k] and Term 3 with -a[j,k].
    # So: (a[i,k] - a[j,k]) * (b[x,p[k]] - b[y,p[k]])
    # b[p[k],x] - b[p[k],y] appears in Term 2 with +a[k,i] and Term 4 with -a[k,j].
    # So: (a[k,i] - a[k,j]) * (b[p[k],x] - b[p[k],y])
    
    # Sum over k != i, j.
    
    # Let's compute this vectorized.
    # p is an array.
    # We need to exclude k=i and k=j.
    
    # b_x_pk = b[x, p]  # vector of b[x, p[k]]
    # b_y_pk = b[y, p]  # vector of b[y, p[k]]
    # b_pk_x = b[p, x]  # vector of b[p[k], x]
    # b_pk_y = b[p, y]  # vector of b[p[k], y]
    
    # diff1 = (a[i,:] - a[j,:]) * (b[x,p] - b[y,p])
    # diff2 = (a[:,i] - a[:,j]) * (b[p,x] - b[p,y])
    
    # total_diff = sum(diff1[k] + diff2[k] for k != i, j) + (new_ij - old_ij)
    
    # Note: a is symmetric? Not necessarily. b is symmetric? QAP matrices are usually symmetric.
    # Assuming symmetry: a[i,k]=a[k,i], b[x,p[k]]=b[p[k],x].
    # If symmetric, diff1 and diff2 are identical (element-wise).
    # Let's assume symmetry for speed, but keep general code if possible.
    # For tai256c, matrices are symmetric.
    
    # Let's use the symmetric assumption to speed up, but verify with general code if needed.
    # Actually, the formula (a[i,k] - a[j,k]) * (b[x,p[k]] - b[y,p[k]]) works even if not symmetric?
    # Let's check Term 2 again: a[k,i]*(b[p[k],x] - b[p[k],y]).
    # If a is symmetric, a[k,i] = a[i,k].
    # If b is symmetric, b[p[k],x] = b[x,p[k]] and b[p[k],y] = b[y,p[k]].
    # Then Term 2 diff = a[i,k]*(b[x,p[k]] - b[y,p[k]]).
    # So diff1 and diff2 are the same vector.
    # Sum = 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[x,p[k]] - b[y,p[k]])
    
    # Let's implement the general case first to be safe, or just assume symmetry.
    # QAPLIB instances are symmetric.
    
    # Vectorized calculation:
    # row_diff_a = a[i, :] - a[j, :]  # shape (n,)
    # col_diff_a = a[:, i] - a[:, j]  # shape (n,)
    
    # b_x_p = b[x, p]
    # b_y_p = b[y, p]
    # b_p_x = b[p, x]
    # b_p_y = b[p, y]
    
    # term1 = row_diff_a * (b_x_p - b_y_p)
    # term2 = col_diff_a * (b_p_x - b_p_y)
    
    # diff_vec = term1 + term2
    
    # Set diff_vec[i] = 0 and diff_vec[j] = 0
    # sum_k = np.sum(diff_vec)
    
    # old_ij = a[i,j]*b[x,y] + a[j,i]*b[y,x]
    # new_ij = a[i,j]*b[y,x] + a[j,i]*b[x,y]
    
    # delta = sum_k + new_ij - old_ij
    
    # This is O(n) per swap.
    
    # Let's implement this.
    
    row_diff_a = a[i, :] - a[j, :]
    col_diff_a = a[:, i] - a[:, j]
    
    b_x_p = b[x, p]
    b_y_p = b[y, p]
    b_p_x = b[p, x]
    b_p_y = b[p, y]
    
    term1 = row_diff_a * (b_x_p - b_y_p)
    term2 = col_diff_a * (b_p_x - b_p_y)
    
    diff_vec = term1 + term2
    diff_vec[i] = 0
    diff_vec[j] = 0
    
    sum_k = np.sum(diff_vec)
    
    old_ij = a[i, j] * b[x, y] + a[j, i] * b[y, x]
    new_ij = a[i, j] * b[y, x] + a[j, i] * b[x, y]
    
    return sum_k + new_ij - old_ij


def _local_search_2opt(a, b, p, max_time=100):
    """
    Perform 2-opt local search.
    p is a numpy array of int64.
    Returns the best p found.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute row sums of a? Not directly useful for 2-opt delta.
    
    # We will iterate until no improvement or time limit.
    # To speed up, we can use a list of candidates.
    
    # For n=256, there are ~32k pairs.
    # Each delta is O(n) ~ 256 ops.
    # Total ~ 8M ops per full pass.
    # In Python, this might be slow.
    # Vectorizing the delta calculation for ALL pairs at once is hard due to dependencies.
    # But we can vectorize the inner loop of delta calculation for a fixed i, varying j.
    
    # Let's try a simple loop first and see if it's fast enough.
    # If not, we need a more vectorized approach.
    
    # Alternative: Greedy hill climbing.
    # At each step, find the best swap.
    # This is O(n^2) swaps to check, each O(n). Total O(n^3) per step.
    # 256^3 = 16M ops. In Python, this is too slow (seconds per step).
    
    # We need a faster local search.
    # Maybe use a random subset of swaps?
    # Or use a more efficient delta calculation.
    
    # Let's try to vectorize the calculation of deltas for a fixed i against all j.
    
    # For fixed i, and all j != i:
    # x = p[i]
    # y = p[j]
    # delta(i, j) = ...
    
    # This is complex to vectorize fully because b[x, p[j]] depends on j.
    
    # Let's stick to a simple 2-opt but with early stopping or limited passes.
    # Or use a different heuristic.
    
    # Given the time limit (150s), and the complexity, a simple 2-opt might not converge enough.
    # Let's try a "First Improvement" or "Best Improvement" 2-opt.
    
    # To make it faster, we can use NumPy to compute deltas for a batch of j's for a given i.
    
    # Let's try this:
    # For i in range(n):
    #   Compute deltas for all j > i (or all j != i)
    #   Find best j
    #   Swap if improved
    
    # This is O(n^2) delta calculations per pass.
    # If we can vectorize the delta calculation for a fixed i, it might be fast.
    
    # Let's implement a vectorized delta for fixed i.
    
    def _deltas_for_i(a, b, p, i):
        """Compute deltas for swapping p[i] with p[j] for all j != i."""
        n = len(p)
        x = p[i]
        
        # y = p[j] for all j
        y = p  # shape (n,)
        
        # row_diff_a = a[i, :] - a[j, :]  -> a[i, :] - a[y, :]  (shape (n, n))
        # This is a[i, :] (1, n) - a[y, :] (n, n)
        # a[i, :] is shape (n,), a[y, :] is shape (n, n)
        # row_diff_a = a[i, np.newaxis, :] - a[y, :]  # shape (n, n)
        
        # col_diff_a = a[:, i] - a[:, j] -> a[:, i] - a[:, y]
        # a[:, i] is shape (n,), a[:, y] is shape (n, n)
        # col_diff_a = a[:, i][:, np.newaxis] - a[:, y]  # shape (n, n)
        
        # b_x_p = b[x, p]  # shape (n,)
        # b_y_p = b[y, p]  # shape (n, n) -> b[y, p]
        # b_p_x = b[p, x]  # shape (n,)
        # b_p_y = b[p, y]  # shape (n, n) -> b[p, y]
        
        # term1 = row_diff_a * (b_x_p[np.newaxis, :] - b_y_p)  # shape (n, n)
        # term2 = col_diff_a * (b_p_x[np.newaxis, :] - b_p_y)  # shape (n, n)
        
        # diff_vec = term1 + term2  # shape (n, n)
        # diff_vec[i, :] = 0  # exclude j=i
        # diff_vec[:, i] = 0  # exclude j=i (redundant)
        
        # sum_k = np.sum(diff_vec, axis=1)  # shape (n,)
        
        # old_ij = a[i, y] * b[x, y] + a[y, i] * b[y, x]  # shape (n,)
        # new_ij = a[i, y] * b[y, x] + a[y, i] * b[x, y]  # shape (n,)
        
        # deltas = sum_k + new_ij - old_ij  # shape (n,)
        
        # This is O(n^2) memory and computation per i.
        # Total O(n^3) per pass.
        # For n=256, 256^3 = 16M ops.
        # In NumPy, this should be fast (milliseconds per pass).
        
        row_diff_a = a[i, np.newaxis, :] - a[y, :]
        col_diff_a = a[:, i][:, np.newaxis] - a[:, y]
        
        b_x_p = b[x, p]
        b_y_p = b[y, p]
        b_p_x = b[p, x]
        b_p_y = b[p, y]
        
        term1 = row_diff_a * (b_x_p[np.newaxis, :] - b_y_p)
        term2 = col_diff_a * (b_p_x[np.newaxis, :] - b_p_y)
        
        diff_vec = term1 + term2
        diff_vec[i, :] = 0
        diff_vec[:, i] = 0
        
        sum_k = np.sum(diff_vec, axis=1)
        
        old_ij = a[i, y] * b[x, y] + a[y, i] * b[y, x]
        new_ij = a[i, y] * b[y, x] + a[y, i] * b[x, y]
        
        deltas = sum_k + new_ij - old_ij
        
        return deltas
    
    # Now, the main loop:
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            deltas = _deltas_for_i(a, b, p, i)
            # Find best j != i
            # Exclude i
            deltas[i] = np.inf
            best_j = np.argmin(deltas)
            if deltas[best_j] < 0:
                # Swap
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += deltas[best_j]
                improved = True
        if time.time() - start_time > max_time:
            break
    
    return p, current_cost


def _greedy_initial(a, b):
    """
    Greedy initialization.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Sort facilities by row sum of a (descending)
    a_sums = a.sum(axis=1)
    order = np.argsort(-a_sums, kind="stable")
    
    # Sort locations by row sum of b (ascending)
    b_sums = b.sum(axis=1)
    loc_order = np.argsort(b_sums, kind="stable")
    
    # Assign in order
    for i, fac in enumerate(order):
        # Choose the best available location for fac
        # Cost contribution of placing fac at loc l:
        # sum_{k assigned} a[fac, k] * b[l, p[k]] + sum_{k assigned} a[k, fac] * b[p[k], l]
        # + self term? No, self term is constant for all l? No, b[l,l] is 0 usually.
        
        # We need to compute the incremental cost for each available location.
        # This is O(n^2) per assignment, total O(n^3).
        # For n=256, this is 16M ops. Acceptable.
        
        best_loc = -1
        best_cost = np.inf
        
        for loc in loc_order:
            if used[loc]:
                continue
            
            # Compute incremental cost
            # k are the already assigned facilities
            # assigned_facs = order[:i]
            # assigned_locs = p[order[:i]]
            
            # cost = sum_{k in assigned} a[fac, k] * b[loc, p[k]] + a[k, fac] * b[p[k], loc]
            
            # Vectorized:
            # a_fac_assigned = a[fac, order[:i]]
            # b_loc_assigned = b[loc, p[order[:i]]]
            # term1 = np.sum(a_fac_assigned * b_loc_assigned)
            
            # a_assigned_fac = a[order[:i], fac]
            # b_assigned_loc = b[p[order[:i]], loc]
            # term2 = np.sum(a_assigned_fac * b_assigned_loc)
            
            # total = term1 + term2
            
            # This is O(i) per loc. Total O(n^2) per fac.
            
            if i == 0:
                cost = 0
            else:
                a_fac_assigned = a[fac, order[:i]]
                b_loc_assigned = b[loc, p[order[:i]]]
                term1 = np.dot(a_fac_assigned, b_loc_assigned)
                
                a_assigned_fac = a[order[:i], fac]
                b_assigned_loc = b[p[order[:i]], loc]
                term2 = np.dot(a_assigned_fac, b_assigned_loc)
                
                cost = term1 + term2
            
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        
        p[fac] = best_loc
        used[best_loc] = True
    
    return p


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = len(a)
    
    # 1. Greedy Initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # 2. Local Search 2-opt
    # We have ~130s left.
    # Let's run 2-opt until time limit.
    p, current_cost = _local_search_2opt(a, b, p, max_time=130)
    
    # 3. Return
    if time.time() > deadline:
        # Should not happen, but safe
        pass
    
    return p.tolist()

# EVOLVE-BLOCK-END
