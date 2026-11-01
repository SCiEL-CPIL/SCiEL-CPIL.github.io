"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def _classic_superpermutation(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _solve_ilp_superpermutation(n, deadline):
    """
    Solve the shortest superpermutation problem via ILP formulation.
    
    Formulation: We have n! permutations as nodes. We want to find a path through
    all nodes minimizing total length. The length contribution of moving from
    permutation A to permutation B is (n - k) where k is the maximum overlap
    (suffix of A matching prefix of B).
    
    For n=7, this is 5040 nodes which is too large for direct ILP.
    We use a heuristic approach with local search instead.
    """
    start_time = time.time()
    timeout = deadline - (time.time() - start_time)
    
    # For small n, we can try exact methods
    if n <= 4:
        # Try to find optimal via exhaustive search or ILP
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        import numpy as np
        
        perms = list(itertools.permutations(range(1, n+1)))
        N = len(perms)
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        
        # Compute overlap matrix: overlap[i][j] = max k such that
        # perms[i][-k:] == perms[j][:k]
        overlaps = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    overlaps[i][j] = 0
                    continue
                max_k = 0
                for k in range(n-1, 0, -1):
                    if perms[i][-k:] == perms[j][:k]:
                        max_k = k
                        break
                overlaps[i][j] = max_k
        
        # Cost of edge i->j is (n - overlaps[i][j]) - 1 for the additional symbols
        # Actually, if we have permutation A and then B, and they overlap by k,
        # we add (n-k) new symbols. The total word length is n + sum of (n - overlap) for each transition.
        
        # For n<=4, N is at most 24, so we can try TSP-like approach
        # Use Held-Karp or just brute force for small N
        if N <= 12:
            # Brute force with memoization
            from functools import lru_cache
            import sys
            sys.setrecursionlimit(10000)
            
            @lru_cache(maxsize=None)
            def dp(mask, last):
                if mask == (1 << N) - 1:
                    return 0
                best = float('inf')
                for nxt in range(N):
                    if not (mask & (1 << nxt)):
                        cost = n - overlaps[last][nxt]
                        val = cost + dp(mask | (1 << nxt), nxt)
                        if val < best:
                            best = val
                return best
            
            # Try all starting points
            best_total = float('inf')
            best_start = 0
            for s in range(N):
                val = dp(1 << s, s)
                total_len = n + val  # n for initial permutation + additional symbols
                if total_len < best_total:
                    best_total = total_len
                    best_start = s
            
            # Reconstruct path
            mask = 1 << best_start
            last = best_start
            path = [last]
            while mask != (1 << N) - 1:
                for nxt in range(N):
                    if not (mask & (1 << nxt)):
                        cost = n - overlaps[last][nxt]
                        if cost + dp(mask | (1 << nxt), nxt) == dp(mask, last):
                            path.append(nxt)
                            mask |= (1 << nxt)
                            last = nxt
                            break
            
            # Build the word
            word = list(perms[path[0]])
            for i in range(1, len(path)):
                ov = overlaps[path[i-1]][path[i]]
                word.extend(perms[path[i]][ov:])
            
            return "".join(map(str, word))
        
        # For n=4, N=24, use Held-Karp
        import numpy as np
        INF = float('inf')
        dp = [[INF] * N for _ in range(1 << N)]
        parent = [[-1] * N for _ in range(1 << N)]
        
        for s in range(N):
            dp[1 << s][s] = 0
        
        for mask in range(1 << N):
            for last in range(N):
                if dp[mask][last] == INF:
                    continue
                for nxt in range(N):
                    if not (mask & (1 << nxt)):
                        cost = n - overlaps[last][nxt]
                        new_mask = mask | (1 << nxt)
                        new_val = dp[mask][last] + cost
                        if new_val < dp[new_mask][nxt]:
                            dp[new_mask][nxt] = new_val
                            parent[new_mask][nxt] = last
        
        full_mask = (1 << N) - 1
        best_end = -1
        best_val = INF
        for end in range(N):
            if dp[full_mask][end] < best_val:
                best_val = dp[full_mask][end]
                best_end = end
        
        # Reconstruct
        path = [best_end]
        mask = full_mask
        last = best_end
        while parent[mask][last] != -1:
            last = parent[mask][last]
            mask ^= (1 << last)
            path.append(last)
        path.reverse()
        
        word = list(perms[path[0]])
        for i in range(1, len(path)):
            ov = overlaps[path[i-1]][path[i]]
            word.extend(perms[path[i]][ov:])
        
        return "".join(map(str, word))
    
    # For n >= 5, use heuristic local search
    # Start with classic construction and improve via 2-opt / 3-opt
    classic = _classic_superpermutation(n)
    classic_list = [int(c) for c in classic]
    
    # Extract permutations in order of appearance
    perms_in_order = []
    seen = set()
    for i in range(len(classic_list) - n + 1):
        window = tuple(classic_list[i:i+n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms_in_order.append(window)
    
    # Build overlap matrix for these permutations
    N = len(perms_in_order)
    if N > 5000:
        # Too large for efficient local search, return classic
        return classic
    
    overlaps = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            max_k = 0
            for k in range(n-1, 0, -1):
                if perms_in_order[i][-k:] == perms_in_order[j][:k]:
                    max_k = k
                    break
            overlaps[i][j] = max_k
    
    # Current order: try to get a good initial ordering via greedy nearest-neighbor
    # Start from each of a few seeds and pick the best
    import random
    random.seed(42)
    
    def greedy_order(start_idx):
        """Greedy nearest-neighbor ordering"""
        remaining = set(range(N))
        remaining.discard(start_idx)
        ord_ = [start_idx]
        cur = start_idx
        while remaining:
            best_next = -1
            best_cost = float('inf')
            for nxt in remaining:
                c = n - overlaps[cur][nxt]
                if c < best_cost:
                    best_cost = c
                    best_next = nxt
            ord_.append(best_next)
            remaining.discard(best_next)
            cur = best_next
        return ord_
    
    # Try a few greedy starts
    best_init_order = None
    best_init_len = float('inf')
    n_seeds = min(10, N)
    seed_indices = list(range(n_seeds))
    for s in seed_indices:
        ord_ = greedy_order(s)
        total = n
        for i in range(len(ord_) - 1):
            total += n - overlaps[ord_[i]][ord_[i+1]]
        if total < best_init_len:
            best_init_len = total
            best_init_order = ord_
    
    if best_init_order is None:
        best_init_order = list(range(N))
    
    order = best_init_order[:]
    
    def compute_length(ord_):
        """Compute total word length given permutation order"""
        total = n
        for i in range(len(ord_) - 1):
            total += n - overlaps[ord_[i]][ord_[i+1]]
        return total
    
    def build_word(ord_):
        """Build the superpermutation word from permutation order"""
        word = list(perms_in_order[ord_[0]])
        for i in range(1, len(ord_)):
            ov = overlaps[ord_[i-1]][ord_[i]]
            word.extend(perms_in_order[ord_[i]][ov:])
        return "".join(map(str, word))
    
    current_len = compute_length(order)
    best_order = order[:]
    best_len = current_len
    
    # Precompute cost matrix for O(1) incremental updates
    # cost[i][j] = n - overlaps[i][j]
    cost = [[n - overlaps[i][j] for j in range(N)] for i in range(N)]
    
    # 2-opt local search with incremental length computation
    deadline_time = start_time + timeout * 0.75
    improved = True
    while improved and (time.time() - start_time) < deadline_time:
        improved = False
        for i in range(N - 1):
            if (time.time() - start_time) >= deadline_time:
                break
            for j in range(i + 1, N):
                # Try reversing segment i..j
                # Old edges affected: (i-1, i) and (j, j+1) if they exist
                old_cost = 0
                if i > 0:
                    old_cost += cost[order[i-1]][order[i]]
                if j < N - 1:
                    old_cost += cost[order[j]][order[j+1]]
                
                # New edges after reversal:
                # (i-1, j) and (i, j+1) if they exist
                new_cost = 0
                if i > 0:
                    new_cost += cost[order[i-1]][order[j]]
                if j < N - 1:
                    new_cost += cost[order[i]][order[j+1]]
                
                # Internal edges of the reversed segment:
                # Old internal: cost[order[i]][order[i+1]] + ... + cost[order[j-1]][order[j]]
                # New internal: cost[order[j]][order[j-1]] + ... + cost[order[i+1]][order[i]]
                old_internal = 0
                for k in range(i, j):
                    old_internal += cost[order[k]][order[k+1]]
                new_internal = 0
                for k in range(i, j):
                    new_internal += cost[order[j - (k - i)]][order[j - (k - i) + 1]]
                
                delta = (new_cost + new_internal) - (old_cost + old_internal)
                if delta < 0:
                    # Perform the reversal
                    order[i:j+1] = order[i:j+1][::-1]
                    current_len += delta
                    improved = True
                    if current_len < best_len:
                        best_len = current_len
                        best_order = order[:]
    
    # 3-opt local search: try moving a single element
    deadline_time2 = start_time + timeout * 0.9
    improved = True
    while improved and (time.time() - start_time) < deadline_time2:
        improved = False
        for i in range(N):
            if (time.time() - start_time) >= deadline_time2:
                break
            for j in range(N):
                if i == j or i == j + 1:
                    continue
                # Move element at position i to position j
                # Remove from i, insert at j (adjust j if j > i)
                elem = order[i]
                # Old edges: (i-1, i), (i, i+1)
                old_cost = 0
                if i > 0:
                    old_cost += cost[order[i-1]][order[i]]
                if i < N - 1:
                    old_cost += cost[order[i]][order[i+1]]
                
                # Determine insertion point
                if j > i:
                    # After removal, j shifts left by 1
                    insert_pos = j
                else:
                    insert_pos = j
                
                # New edges: (insert_pos-1, elem), (elem, insert_pos)
                new_cost = 0
                if insert_pos > 0:
                    new_cost += cost[order[insert_pos - 1]][elem]
                if insert_pos < N - 1:
                    new_cost += cost[elem][order[insert_pos]]
                
                delta = new_cost - old_cost
                if delta < 0:
                    # Perform the move
                    order.pop(i)
                    order.insert(insert_pos, elem)
                    current_len += delta
                    improved = True
                    if current_len < best_len:
                        best_len = current_len
                        best_order = order[:]
    
    # Return the best word found
    return build_word(best_order)

def construct(n):
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # For n <= 4, try exact solution
    if n <= 4:
        try:
            result = _solve_ilp_superpermutation(n, deadline)
            return result
        except Exception:
            pass
    
    # For n >= 5, use heuristic approach with time limit
    try:
        result = _solve_ilp_superpermutation(n, deadline)
        return result
    except Exception:
        # Fallback to classic construction
        return _classic_superpermutation(n)
# EVOLVE-BLOCK-END
