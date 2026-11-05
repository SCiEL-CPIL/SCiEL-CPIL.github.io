import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors for each word (distance <= 1)
    neighbor_indices = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        # Distance 0
        neighbor_indices[i].append(i)
        # Distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = word_to_idx[tuple(new_w)]
                    neighbor_indices[i].append(idx)

    # 1. Get a good initial solution using Greedy Set Cover
    uncovered = set(range(total_words))
    greedy_code = []
    
    # Optimization: Precompute the "coverage" of each word as a bitmask or set?
    # 2187 bits is ~273 bytes. A Python set of ints is slower.
    # Using a boolean array for uncovered is fast.
    is_covered = [False] * total_words
    
    # We need to pick words that cover the most uncovered nodes.
    # To speed up greedy, we can maintain a count of how many uncovered neighbors each node has.
    # But updating counts for all nodes when we pick one is expensive (O(N * deg)).
    # N=2187, deg=15. 2187*15 = 32k ops per pick. 186 picks -> 6M ops. This is fine.
    
    # Let's use a list of counts.
    uncovered_counts = [len(neighbor_indices[i]) for i in range(total_words)]
    # Wait, this is the count of ALL neighbors. We need count of UNCOVERED neighbors.
    # Initially, all are uncovered.
    # When we pick a node `u`, for every neighbor `v` of `u`, we decrement `uncovered_counts[v]`?
    # No, `uncovered_counts[v]` should reflect how many of v's neighbors are still uncovered?
    # Actually, the gain of picking `u` is the number of uncovered nodes in `neighbor_indices[u]`.
    
    # Let's stick to the simpler O(N * deg) scan for greedy, it's fast enough.
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(total_words):
            if i in uncovered: # Only consider picking from uncovered? No, we can pick any.
                # Actually, if i is covered, picking it might still cover new nodes.
                # But usually we pick from the uncovered set or just scan all.
                pass
        
        # Scan all nodes to find the one covering the most uncovered nodes
        for i in range(total_words):
            gain = 0
            for nb in neighbor_indices[i]:
                if nb in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
            
        greedy_code.append(best_idx)
        for nb in neighbor_indices[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    initial_size = len(greedy_code)
    
    # 2. Try to improve using Local Search / Branch and Bound
    # We will try to find a solution of size < initial_size.
    # We use a time limit.
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    best_code = greedy_code[:]
    
    # Branch and Bound for Dominating Set
    # State: current_set, remaining_uncovered
    # We try to add nodes to current_set until remaining_uncovered is empty.
    # We prune if len(current_set) + lower_bound(remaining_uncovered) >= best_size.
    
    # Lower bound for remaining uncovered:
    # Each node covers at most `max_degree` uncovered nodes.
    # So LB = ceil(len(remaining_uncovered) / max_degree)
    # max_degree is 15.
    
    def lower_bound(uncovered_set):
        if not uncovered_set:
            return 0
        # Max coverage per node is 15 (1 + 14)
        # But as we pick nodes, the effective coverage drops.
        # Simple LB: len(uncovered_set) // 15 + (1 if len(uncovered_set) % 15 else 0)
        return (len(uncovered_set) + 14) // 15

    # We will perform a DFS search for a solution of size K.
    # We start with K = initial_size - 1 and decrease? No, increase from lower bound?
    # Let's try to find a solution of size `target_size`.
    # We iterate target_size from max(156, initial_size - 5) up to initial_size - 1.
    
    found = False
    best_size_found = initial_size
    
    # To make DFS efficient, we order candidates by how many uncovered nodes they cover.
    
    def solve_dominating_set(target_size):
        nonlocal found, best_size_found
        if found:
            return True
            
        current_set = []
        remaining_uncovered = set(range(total_words))
        
        # Recursive helper
        def dfs(depth, remaining_unc):
            nonlocal found
            if found:
                return True
            if time.time() - start_time > time_limit:
                return False
                
            if not remaining_unc:
                if len(current_set) <= target_size:
                    found = True
                    best_code[:] = current_set[:]
                    return True
                return False
                
            if len(current_set) >= target_size:
                return False
                
            # Pruning: If current size + LB > target_size, prune
            lb = lower_bound(remaining_unc)
            if len(current_set) + lb > target_size:
                return False
                
            # Heuristic: Pick the node that covers the most remaining_unc nodes
            # We need to generate candidates.
            # To avoid O(N) scan at every node, we can maintain a list of "active" nodes?
            # For simplicity, we scan all nodes but sort them by gain.
            
            candidates = []
            for i in range(total_words):
                # Only consider nodes that cover at least 1 remaining node
                gain = 0
                for nb in neighbor_indices[i]:
                    if nb in remaining_unc:
                        gain += 1
                if gain > 0:
                    candidates.append((gain, i))
                    
            # Sort by gain descending
            candidates.sort(reverse=True)
            
            for gain, i in candidates:
                # If gain is 0, no point
                if gain == 0:
                    break
                    
                # Try adding i
                current_set.append(i)
                new_remaining = set(remaining_unc)
                for nb in neighbor_indices[i]:
                    if nb in new_remaining:
                        new_remaining.remove(nb)
                        
                if dfs(depth + 1, new_remaining):
                    return True
                    
                current_set.pop()
                
            return False
            
        return dfs(0, remaining_uncovered)

    # Try to find a smaller code
    # We know the best known is 186.
    # If initial_size is close to 186, we try 186.
    # If initial_size is 190, we try 189, 188, 187, 186.
    
    min_target = 186 # Known best
    if initial_size > min_target:
        for t in range(initial_size - 1, min_target - 1, -1):
            if time.time() - start_time > time_limit:
                break
            found = False
            if solve_dominating_set(t):
                best_size_found = t
                break # Found a solution of size t, can we find smaller?
                # The loop goes downwards, so if we find t, we try t-1 next.
                # But solve_dominating_set finds *a* solution of size <= t.
                # If it finds one, we should try to find one of size t-1.
                # So we don't break, we continue to lower t.
                # Wait, if we find a solution of size 186, we can't go lower (known min is 186? No, lower bound is 156).
                # But 186 is the *best known*. It is possible 185 exists? Unlikely.
                # Let's just stop if we hit 186 to save time, as 186 is a very strong result.
                if t == 186:
                    break

    # If we didn't find anything better, return the greedy one
    # But wait, if we found a solution in DFS, best_code is updated.
    
    result = [words[i] for i in best_code]
    
    # Final validity check (just in case)
    # This is O(N * |code|)
    covered = [False] * total_words
    for idx in best_code:
        for nb in neighbor_indices[idx]:
            covered[nb] = True
    if not all(covered):
        # Fallback to greedy if DFS returned invalid (should not happen)
        result = [words[i] for i in greedy_code]
        
    return result
