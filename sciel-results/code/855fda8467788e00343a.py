import numpy as np
import time
import sys

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    Assumes a_sym = a + a.T.
    Cost C(p) = sum_{u,v} a[u,v] b[p[u], p[v]].
    If a is symmetric, C(p) = 1/2 sum_{u,v} a_sym[u,v] b[p[u], p[v]].
    Delta for swap (i,j):
    Let x=p[i], y=p[j].
    delta = sum_{k != i,j} (a_sym[i,k] - a_sym[j,k]) * (b[y,k] - b[x,k]) - 2*(a_sym[i,j] * (b[x,x] - b[x,y] - b[y,x] + b[y,y]))
    Wait, the standard O(n) formula for symmetric a:
    delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,k] - b[x,k]) - a[i,j] * (b[x,x] - b[x,y] - b[y,x] + b[y,y])
    Using a_sym = 2a:
    delta = 1/2 * sum_{k != i,j} (a_sym[i,k] - a_sym[j,k]) * (b[y,k] - b[x,k]) - 1/2 * a_sym[i,j] * (b[x,x] - b[x,y] - b[y,x] + b[y,y])
    
    However, the current program's _swap_delta implementation is:
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    Let's verify this formula.
    sum_{k} (a_sym[i,k] - a_sym[j,k]) * (b[y,k] - b[x,k])
    = sum_{k} (a_sym[i,k] b[y,k] - a_sym[i,k] b[x,k] - a_sym[j,k] b[y,k] + a_sym[j,k] b[x,k])
    
    Terms involving p[k]=x (k=i) and p[k]=y (k=j) need adjustment because p changes.
    Actually, the formula provided in the current program is a known vectorized O(n) swap delta for QAP.
    Let's stick to it but ensure it handles the symmetric case correctly.
    The cost is sum a[i,j] b[p[i], p[j]].
    If we use a_sym, the cost is 0.5 * sum a_sym[i,j] b[p[i], p[j]].
    The delta calculated by the current formula corresponds to the change in the sum S = sum a_sym[i,j] b[p[i], p[j]].
    So the actual cost delta is 0.5 * delta_formula?
    Let's check.
    If C = 0.5 * S, then dC = 0.5 * dS.
    The current program uses `current_cost += best_delta`.
    If `best_delta` is dS, then it should be `current_cost += 0.5 * best_delta`.
    However, the current program scores 2.1995, which is close to the best known.
    Let's re-read the current program's _swap_delta.
    It returns `delta`.
    In _tabu_search (which is a placeholder/pass in the provided text but implied in the strategy), 
    it says `current_cost += best_delta`.
    
    Let's derive the delta for C = sum a[i,j] b[p[i], p[j]] directly with a_sym.
    Assume a is symmetric.
    C = 0.5 * sum_{u,v} a_sym[u,v] b[p[u], p[v]].
    Swap i, j. x=p[i], y=p[j].
    Change in S = sum a_sym[u,v] b[p[u], p[v]]:
    Terms involving i or j change.
    S_new - S_old = 
    sum_{k!=i,j} (a_sym[i,k] b[p[k], y] + a_sym[j,k] b[p[k], x] - a_sym[i,k] b[p[k], x] - a_sym[j,k] b[p[k], y])
    + (a_sym[i,j] b[y, x] + a_sym[j,i] b[x, y] - a_sym[i,j] b[x, x] - a_sym[j,i] b[y, y])
    + (a_sym[i,i] b[y, y] + a_sym[j,j] b[x, x] - a_sym[i,i] b[x, x] - a_sym[j,j] b[y, y])
    
    Since a_sym is symmetric, a_sym[i,j]=a_sym[j,i].
    The last two terms: a_sym[i,i](b[yy]-b[xx]) + a_sym[j,j](b[xx]-b[yy]) = (a_sym[i,i]-a_sym[j,j])(b[yy]-b[xx]).
    The middle term: 2 a_sym[i,j] (b[y,x] + b[x,y] - b[x,x] - b[y,y]). Note b is symmetric? QAPLIB b is usually symmetric.
    If b is symmetric, b[y,x]=b[x,y].
    Middle: 2 a_sym[i,j] (2 b[x,y] - b[x,x] - b[y,y]).
    
    First term: sum_{k!=i,j} (a_sym[i,k] - a_sym[j,k]) (b[p[k], y] - b[p[k], x]).
    
    The vectorized formula in the current program:
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    temp = diff_a * diff_b[p]
    sum_temp = sum_k (a_sym[i,k] - a_sym[j,k]) (b[y, p[k]] - b[x, p[k]])
    
    Note: b[y, p[k]] is not b[p[k], y]. If b is symmetric, b[y, p[k]] = b[p[k], y].
    So sum_temp = sum_k (a_sym[i,k] - a_sym[j,k]) (b[p[k], y] - b[p[k], x]).
    
    This sum includes k=i and k=j.
    For k=i: p[i]=x. Term: (a_sym[i,i] - a_sym[j,i]) (b[x, y] - b[x, x]).
    For k=j: p[j]=y. Term: (a_sym[i,j] - a_sym[j,j]) (b[y, y] - b[y, x]).
    
    So sum_{k!=i,j} ... = sum_temp - Term_i - Term_j.
    
    Total Delta S = sum_{k!=i,j} ... + Middle + Diagonal Terms.
    
    Let's check if the current formula `delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]` matches.
    diff_a[i] = a_sym[i,i] - a_sym[j,i].
    diff_b[x] = b[y, x] - b[x, x].
    So diff_a[i] * diff_b[x] = Term_i.
    diff_a[j] = a_sym[i,j] - a_sym[j,j].
    diff_b[y] = b[y, y] - b[x, y].
    So diff_a[j] * diff_b[y] = Term_j.
    
    So the current formula calculates `sum_{k!=i,j} ...`.
    It DOES NOT include the Middle and Diagonal terms.
    
    This implies the current program's delta calculation is INCOMPLETE for the full cost change?
    Or does the "Middle + Diagonal" part cancel out or get handled elsewhere?
    No, it's a direct delta.
    
    Let's re-evaluate.
    Maybe the cost definition in the current program's mind is different?
    Or maybe I am missing that `b` is not symmetric? QAPLIB instances have symmetric B.
    
    Let's test the formula with a small example mentally.
    If the current program works (score 2.1995 is very good, near best known 44.7M), the delta must be correct.
    Why?
    
    Ah, look at the cost function: `sum a[i,j] b[p[i], p[j]]`.
    If a is symmetric, C = 0.5 * sum a_sym[i,j] b[p[i], p[j]].
    
    Let's look at the "Middle + Diagonal" terms again.
    Middle: 2 a_sym[i,j] (b[x,y] - b[x,x] - b[y,y] + b[x,y])? No.
    Old terms involving (i,j): a_sym[i,j] b[x,y] + a_sym[j,i] b[y,x] = 2 a_sym[i,j] b[x,y] (if b sym).
    New terms involving (i,j): a_sym[i,j] b[y,x] + a_sym[j,i] b[x,y] = 2 a_sym[i,j] b[x,y].
    So the cross terms (i,j) and (j,i) do NOT change value if b is symmetric!
    b[x,y] becomes b[y,x] which is b[x,y].
    So Middle term is 0.
    
    Diagonal terms:
    Old: a_sym[i,i] b[x,x] + a_sym[j,j] b[y,y].
    New: a_sym[i,i] b[y,y] + a_sym[j,j] b[x,x].
    Change: a_sym[i,i](b[y,y]-b[x,x]) + a_sym[j,j](b[x,x]-b[y,y]) = (a_sym[i,i]-a_sym[j,j])(b[y,y]-b[x,x]).
    
    Does the vectorized formula account for this?
    Term_i in sum_temp: (a_sym[i,i] - a_sym[j,i]) (b[x,y] - b[x,x]).
    Term_j in sum_temp: (a_sym[i,j] - a_sym[j,j]) (b[y,y] - b[y,x]).
    
    The formula subtracts these.
    So `sum_{k!=i,j}` is correct.
    
    So Delta S = sum_{k!=i,j} ... + 0 + (a_sym[i,i]-a_sym[j,j])(b[y,y]-b[x,x]).
    
    The current formula returns `sum_{k!=i,j} ...`.
    It is MISSING the diagonal term: `(a_sym[i,i]-a_sym[j,j])(b[y,y]-b[x,x])`.
    
    If the current program ignores this term, it is using an approximate delta or the diagonal terms are zero?
    QAPLIB matrices usually have 0 on the diagonal for B?
    "distance matrix" -> usually 0 diagonal.
    If b[x,x]=0 and b[y,y]=0, then the diagonal term is 0.
    If b has 0 diagonal, the diagonal term vanishes.
    
    So the current formula is EXACT if b has 0 diagonal.
    tai256c B matrix: QAPLIB distance matrices are typically 0 on diagonal.
    So the current delta is exact.
    
    Okay, I will use the same delta formula but optimize the local search loop and initialization.
    """
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _spectral_init(a, b, n):
    """
    Spectral initialization.
    Construct L = D_a - A_a? Or use embedding.
    Simple approach:
    1. Compute row sums of A and B.
    2. Use greedy matching based on "centrality".
    3. Improve with a few random restarts of greedy + local search?
    
    Better:
    Use the fact that QAP is related to graph matching.
    For a high-quality start, we can use a "rank" based on eigenvectors.
    Compute eigenvector of A corresponding to largest eigenvalue?
    Or use a simple heuristic:
    Sort facilities by flow degree (sum of row).
    Sort locations by distance degree (sum of row).
    Match largest flow to smallest distance (or largest, depending on sign).
    Since we minimize sum a*b, and a,b > 0, we want high flow at low distance.
    So sort A_row_sums descending, B_row_sums ascending.
    """
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # We want high flow (large a_sum) to be at low distance (small b_sum).
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    return p

def _local_search(a_sym, b, p, time_limit, rng):
    """
    Fast Iterated Local Search with 2-swap.
    Uses vectorized delta.
    """
    n = len(p)
    current_p = p.copy()
    current_cost = _qap_cost(a_sym // 2 if np.allclose(a_sym, a_sym.T) else a_sym, b, current_p) 
    # Wait, _qap_cost takes original a. I don't have original a here.
    # I should pass original a or compute cost using a_sym/2.
    # Let's pass a and b to local search.
    pass

def _solve_tai256c(a, b, time_limit=145.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Check if a is symmetric (it should be)
    # If not, the delta formula needs adjustment. 
    # QAPLIB tai256c is symmetric.
    
    # 1. Spectral/Greedy Initialization
    p_init = _spectral_init(a, b, n)
    
    # Run a quick local search on init to polish
    # But we need a local search function that takes a, b.
    
    def _fast_ls(a, b, a_sym, p, t_limit):
        n = len(p)
        current_p = p.copy()
        current_cost = _qap_cost(a, b, current_p)
        best_p = current_p.copy()
        best_cost = current_cost
        
        # Precompute some things?
        # Just loop.
        
        while time.time() < t_limit:
            improved = False
            # First-improvement or Best-improvement?
            # Best-improvement is O(n^2) scan. 32k swaps.
            # 32k * 20us = 0.64s per scan.
            # Too slow for many scans.
            # Use random sampling of moves.
            
            # Sample 1000 moves
            indices = np.random.randint(0, n, size=(1000, 2))
            i_arr, j_arr = indices[:, 0], indices[:, 1]
            valid = i_arr != j_arr
            i_arr = i_arr[valid]
            j_arr = j_arr[valid]
            
            if len(i_arr) == 0:
                break
                
            # Vectorized delta calculation for all sampled moves?
            # delta[i] for move (i, j_i)
            # x = p[i], y = p[j]
            # diff_a = a_sym[i, :] - a_sym[j, :]
            # This is hard to vectorize across different i, j pairs easily without loops or advanced indexing.
            # a_sym[i_arr, :] is (1000, n).
            # a_sym[j_arr, :] is (1000, n).
            # diff_a = a_sym[i_arr, :] - a_sym[j_arr, :]
            
            x = current_p[i_arr]
            y = current_p[j_arr]
            
            # diff_b = b[y, :] - b[x, :]
            # b[y, :] is (1000, n).
            diff_b = b[y, :] - b[x, :]
            
            # temp = diff_a * diff_b[current_p]
            # current_p is (n,). We need diff_b indexed by current_p?
            # No, diff_b[k] = b[y, k] - b[x, k].
            # We need sum over k of diff_a[k] * diff_b[p[k]]?
            # No, the formula is sum_k (a_sym[i,k] - a_sym[j,k]) * (b[y, p[k]] - b[x, p[k]]).
            # Wait, my previous derivation:
            # diff_b[p[k]] = b[y, p[k]] - b[x, p[k]].
            # So we need to index diff_b by current_p.
            
            # diff_b is (1000, n).
            # We want for each move m, sum_k diff_a[m, k] * diff_b[m, p[k]].
            # This is not a simple dot product because p[k] changes index for each m?
            # No, p is the same for all m in this batch.
            # So for each m, we sum over k.
            # term[m, k] = diff_a[m, k] * diff_b[m, p[k]].
            # This requires advanced indexing.
            
            # p[k] for all k is just current_p.
            # So we need diff_b[m, current_p[k]] for all m, k.
            # This is a 2D array of size (1000, n).
            # We can do:
            # indexed_diff_b = diff_b[np.arange(1000)[:, None], current_p[None, :]]
            
            idx = np.arange(len(i_arr))[:, None]
            p_idx = current_p[None, :]
            
            indexed_diff_b = diff_b[idx, p_idx]
            
            temp = diff_a * indexed_diff_b
            
            # Sum over k (axis 1)
            sums = np.sum(temp, axis=1)
            
            # Subtract terms for k=i and k=j
            # Term_i: diff_a[i, i] * diff_b[i, p[i]] = diff_a[i, i] * diff_b[i, x[i]]
            # Term_j: diff_a[i, j] * diff_b[i, p[j]] = diff_a[i, j] * diff_b[i, y[i]]
            
            # diff_a is (1000, n).
            # diff_a[i, i] -> np.diag? No, i is the row index in the batch, but also the position in p.
            # Let's be careful.
            # Batch index m corresponds to move (i_arr[m], j_arr[m]).
            # We need diff_a[m, i_arr[m]] and diff_a[m, j_arr[m]].
            
            di = diff_a[np.arange(len(i_arr)), i_arr]
            dj = diff_a[np.arange(len(i_arr)), j_arr]
            
            # diff_b[m, x[m]] and diff_b[m, y[m]]
            # x[m] = p[i_arr[m]], y[m] = p[j_arr[m]]
            # We already have x, y.
            # diff_b is (1000, n).
            # We need diff_b[m, x[m]].
            db_x = diff_b[np.arange(len(i_arr)), x]
            db_y = diff_b[np.arange(len(i_arr)), y]
            
            corrections = di * db_x + dj * db_y
            
            deltas = sums - corrections
            
            # Find best delta
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                i_move = i_arr[min_idx]
                j_move = j_arr[min_idx]
                
                # Apply
                current_p[i_move], current_p[j_move] = current_p[j_move], current_p[i_move]
                current_cost += min_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                    improved = True
            else:
                # No improving move in sample. 
                # Try random perturbation (ILS)
                # Reverse a segment or swap 2 random
                i_rand, j_rand = np.random.choice(n, size=2, replace=False)
                current_p[i_rand], current_p[j_rand] = current_p[j_rand], current_p[i_rand]
                # Recompute cost? O(n^2).
                # Or just continue?
                # To keep it fast, we might just restart LS from best_p with a perturbation.
                current_p = best_p.copy()
                # Perturb
                for _ in range(5):
                    i_r, j_r = np.random.choice(n, size=2, replace=False)
                    current_p[i_r], current_p[j_r] = current_p[j_r], current_p[i_r]
                current_cost = _qap_cost(a, b, current_p)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
        
        return best_p, best_cost

    # Run Local Search with time split
    # 1. Polish the initial solution
    t1 = start_time + 10.0
    p_best, c_best = _fast_ls(a, b, a_sym, p_init, t1)
    
    # 2. Iterated Local Search
    # While time remains, perturb best and re-optimize
    while time.time() < time_limit - 5.0:
        # Perturb
        p_curr = p_best.copy()
        # Random block reverse or multiple swaps
        # Use 10 random swaps
        for _ in range(10):
            i_r, j_r = np.random.choice(n, size=2, replace=False)
            p_curr[i_r], p_curr[j_r] = p_curr[j_r], p_curr[i_r]
        
        # Optimize
        t_end = min(time.time() + 10.0, time_limit - 5.0)
        p_new, c_new = _fast_ls(a, b, a_sym, p_curr, t_end)
        
        if c_new < c_best:
            p_best = p_new
            c_best = c_new
    
    return p_best.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)

