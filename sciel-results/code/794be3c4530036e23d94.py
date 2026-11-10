import time
import sys
import random

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115  # Safety margin before 120s kill

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Generate Baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    baseline_scope = max(max(row) for row in baseline_rows)
    
    # 2. Determine Search Range
    # Counting bound: n_rows * k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    lower_bound = n_rows * diffs_per_row
    
    # For the specific case (7, 5), we know 112 is optimal. 
    # We will search from lower_bound up to baseline_scope.
    # If baseline is already good (e.g., 112), we verify if we can go lower.
    
    # Optimization: If baseline is close to lower bound, start there.
    # If baseline is much higher, we might start a bit higher to find a valid set faster, 
    # but the task wants the MINIMIZED scope. So we MUST search from lower_bound upwards.
    
    # However, generating the graph for small M is fast.
    # For large M, graph generation is the bottleneck.
    
    # Strategy:
    # Iterate M from lower_bound to baseline_scope.
    # For each M:
    #   a. Generate all valid rows with max element <= M.
    #   b. Build conflict graph (adjacency list/bitmask).
    #   c. Check if Max Independent Set size >= n_rows.
    #   d. If yes, extract the set and return.
    
    # To handle time limits:
    # If M is small, graph is small -> fast.
    # If M is large, graph is huge -> slow.
    # We will cap the size of the graph or the time spent per M.
    
    # Special case handling for speed:
    # If n_rows is small (<= 10) and k is small (<= 7), graph approach is very effective.
    # For larger instances, DFS might be better, but let's stick to the MIS family as per directive.
    
    # Pre-compute bit masks for differences up to a reasonable limit?
    # Max difference is M.
    
    best_found = None
    
    # We search M from lower_bound to baseline_scope
    # But if baseline_scope is very large (e.g. 200+), iterating one by one might be too slow for graph gen.
    # We can jump M by 1.
    
    # To ensure we don't time out on graph generation for large M:
    # We estimate the number of rows. If it's too high, we might skip or use a heuristic.
    # But for (7,5), M=112, number of rows is manageable.
    
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > DEADLINE - 10:
            break
        
        # Generate candidates
        candidates = generate_candidates(M, k)
        
        # If generation took too long or returned None, break
        if candidates is None:
            break
            
        n_cands = len(candidates)
        
        # If we have fewer candidates than n_rows, impossible
        if n_cands < n_rows:
            continue
            
        # Build Conflict Graph
        # nodes: 0..n_cands-1
        # edge if (mask_i & mask_j) != 0
        
        # For efficiency, we use a recursive MIS solver with coloring bound.
        # We need to find an independent set of size exactly n_rows.
        
        sol = find_independent_set(candidates, n_rows, M)
        
        if sol is not None:
            # sol is a list of indices into candidates
            rows = [list(candidates[i][2]) for i in sol]
            # Verify validity just in case
            if is_valid(rows, n_rows, k):
                return rows
                
    return baseline_rows

def is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, k+1):
            if row[i] <= row[i-1]:
                return False
        # Check diffs
        for i in range(k+1):
            for j in range(i+1, k+1):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    return True

def generate_candidates(M, k):
    """
    Generates all valid rows of length k+1 starting at 0, with max element <= M.
    Returns list of (max_val, mask, row_tuple) or None if timeout.
    """
    start = time.time()
    timeout = 5.0 # 5 seconds per M step
    
    rows = []
    # bit_of[d] = 1 << d
    # We need bits up to M
    # Python ints are arbitrary precision, so 1 << M is fine.
    
    def backtrack(current_row, current_mask, last_val):
        if time.time() - start > timeout:
            return False
        
        if len(current_row) == k + 1:
            max_val = current_row[-1]
            rows.append((max_val, current_mask, tuple(current_row)))
            return True
        
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            # Check conflicts with existing elements in current_row
            # Since current_row is sorted, and x > last_val, all diffs are positive
            for a in current_row:
                d = x - a
                bit = 1 << d
                if current_mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                current_row.append(x)
                if not backtrack(current_row, current_mask | new_mask, x):
                    return False
                current_row.pop()
        
        return True

    if backtrack([0], 0, 0):
        return rows
    else:
        return None

def find_independent_set(candidates, target_size, M):
    """
    Finds an independent set of size target_size in the conflict graph defined by candidates.
    candidates: list of (max_val, mask, row_tuple)
    Returns list of indices or None.
    """
    n = len(candidates)
    if n < target_size:
        return None
    
    # Build adjacency bitmask
    # adj[i] is a bitmask of nodes adjacent to i
    adj = [0] * n
    masks = [c[1] for c in candidates]
    
    # Precompute conflicts
    # O(n^2) might be slow if n is large.
    # For M=112, k=5, n is likely in the hundreds or low thousands.
    # n^2 = 10^6 operations, which is fast in Python if optimized.
    
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            if mi & mj:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # MIS Search
    # We want to find a set of size target_size.
    # We can use a recursive branch and bound.
    # State: available nodes (bitmask), current set (list), current size.
    # Bound: Greedy coloring of the available subgraph to estimate max independent set size.
    
    # To speed up coloring, we can use a simple greedy coloring on the subgraph induced by available nodes.
    # However, doing full coloring at every node of the search tree is expensive.
    # A simpler bound: The number of available nodes is an upper bound.
    # A tighter bound: The size of a maximal independent set found greedily? No, that's a lower bound.
    # We need an UPPER bound to prune.
    # Upper bound 1: len(available)
    # Upper bound 2: If we can't find a clique of size... no, we need independent set.
    
    # Let's use the "Tomita" style coloring bound.
    # Colors are assigned such that nodes in the same color are NOT adjacent?
    # No, in coloring for MIS, we color the GRAPH. Nodes with same color form an independent set.
    # So the number of colors is an UPPER bound on the size of the independent set?
    # Wait. If we color the graph with C colors, any independent set can have at most 1 node of each color?
    # No. An independent set can have multiple nodes of the same color ONLY IF they are not adjacent.
    # But in a proper graph coloring, nodes of the same color are NOT adjacent.
    # So an independent set CAN contain all nodes of a specific color.
    # Therefore, the number of colors is a LOWER bound on the clique number, and an UPPER bound on the independent set?
    # Actually, the size of the maximum independent set is at most the number of colors used in a proper coloring?
    # No. Consider a path of 3 nodes 1-2-3. Coloring: 1(R), 2(G), 3(R). Colors=2. Max Independent Set = {1,3} size 2.
    # Consider a star K1,3. Center R, leaves G,B,Y. Colors=4. Max Independent Set = 3 leaves.
    # So Number of Colors is NOT an upper bound on MIS size.
    
    # Correct Bound for MIS:
    # We want to prune if current_size + upper_bound(available) < target_size.
    # A standard upper bound for MIS is the number of vertices in the available set.
    # A tighter bound is based on the "degree" or "clique cover".
    # Actually, for MIS, we can use the size of a maximum clique in the COMPLEMENT graph? That's hard.
    
    # Let's stick to a simpler, effective heuristic search:
    # Branch on the node with the highest degree (or lowest degree?) in the available set.
    # Actually, for finding a set of FIXED size K, we can use a simple backtracking with pruning:
    # 1. If current_size == K, return success.
    # 2. If current_size + count(available) < K, prune.
    # 3. Pick a node v from available.
    # 4. Branch 1: Include v. Remove v and its neighbors from available. Recurse.
    # 5. Branch 2: Exclude v. Remove v from available. Recurse.
    
    # To make this efficient, we should order nodes.
    # Nodes with higher "conflict" (degree) are more constrained. Including them removes more options.
    # So we should try including high-degree nodes first? Or excluding them?
    # Usually, for MIS, we pick a vertex, branch on include/exclude.
    
    # Optimization: If the available set size is small, we can brute force or use a faster method.
    
    # Let's implement a recursive search with bitmask operations.
    
    # To handle the "Upper Bound" better:
    # We can calculate the size of a greedy independent set on the available subgraph.
    # This gives a LOWER bound on what we can still add? No, it's a specific set.
    # If greedy finds size G, we can definitely add G more.
    # This doesn't help prune "impossible" branches unless G is very small?
    # Actually, if we just need to know if K is possible, and we have a greedy set of size G,
    # and current_size + G < K, it doesn't mean it's impossible, just that greedy failed.
    
    # Let's rely on the simple count bound and good variable ordering.
    # Variable ordering: Pick the node in `available` with the maximum number of neighbors in `available`.
    # This is the "Maximum Degree" heuristic.
    
    # Precompute degrees? Degrees change as we remove nodes.
    # Calculating degree in bitmask: (adj[v] & available).bit_count()
    
    def search(available_mask, current_set, current_size):
        if time.time() > DEADLINE - 5:
            return None
        
        if current_size == target_size:
            return current_set
            
        # Prune 1: Not enough nodes left
        if current_size + available_mask.bit_count() < target_size:
            return None
            
        # Prune 2: If we can't find any more nodes? (Covered by Prune 1)
        
        # Heuristic: Pick node v from available with max degree in available
        # To avoid iterating all bits every time, we can just pick the first one?
        # No, max degree is important for pruning.
        
        # Get list of available nodes
        # available_mask is an int.
        # We need to find v in available_mask that maximizes (adj[v] & available_mask).bit_count()
        
        # This loop is O(N) per node.
        # N is up to ~1000. Search depth is up to 7.
        # 1000 * 7 * branching factor. Should be okay.
        
        best_v = -1
        best_deg = -1
        
        # Iterate through set bits in available_mask
        # tmp = available_mask
        # while tmp:
        #   lsb = tmp & -tmp
        #   v = lsb.bit_length() - 1
        #   deg = (adj[v] & available_mask).bit_count()
        #   ...
        
        # Optimization: If available_mask is small, just iterate.
        
        v = (available_mask & -available_mask).bit_length() - 1
        # Actually, picking the first available node is often good enough if we have the count bound.
        # Let's try picking the node with max degree to prune faster.
        
        # To speed up, we can sample or just pick the first if N is large.
        # For N < 500, max degree is fast.
        
        if available_mask.bit_count() < 200:
             tmp = available_mask
             while tmp:
                 lsb = tmp & -tmp
                 v_idx = lsb.bit_length() - 1
                 deg = (adj[v_idx] & available_mask).bit_count()
                 if deg > best_deg:
                     best_deg = deg
                     best_v = v_idx
                 tmp ^= lsb
        else:
             # Just pick the first one to save time
             v = (available_mask & -available_mask).bit_length() - 1
             best_v = v

        if best_v == -1:
            return None
            
        # Branch 1: Include best_v
        # New available = available_mask & ~(1 << best_v) & ~adj[best_v]
        new_avail_1 = available_mask & ~(1 << best_v) & ~adj[best_v]
        res = search(new_avail_1, current_set + [best_v], current_size + 1)
        if res:
            return res
            
        # Branch 2: Exclude best_v
        new_avail_2 = available_mask & ~(1 << best_v)
        res = search(new_avail_2, current_set, current_size)
        if res:
            return res
            
        return None

    # Initial call
    # Available is all nodes: (1 << n) - 1
    # But if n > 60, bitmask shifts are fine in Python.
    initial_avail = (1 << n) - 1
    return search(initial_avail, [], 0)

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows
