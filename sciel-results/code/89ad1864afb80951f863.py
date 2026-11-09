import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Build conflict graph: edge between a,b if a+b is a perfect square
    # Include self-loops: a+a is a perfect square means a cannot be in set
    # Actually, if a+a is a perfect square, a is forbidden entirely
    
    # Perfect squares up to 512
    squares = set()
    for k in range(1, 23):  # 1^2=1, 22^2=484, 23^2=529>512
        squares.add(k * k)
    
    # Check which numbers are forbidden (a+a is perfect square)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    # Build adjacency: for a != b, edge if a+b is square
    # We want max independent set on graph with nodes 1..256 \ forbidden
    # where edge exists if a+b is perfect square
    
    # Precompute for each node which other nodes conflict
    # conflicts[i] = set of j != i such that i+j is a perfect square
    # Only consider i,j not in forbidden
    
    candidates = [i for i in range(1, N + 1) if i not in forbidden]
    # Map candidate to index
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    n_cands = len(candidates)
    
    # Build conflict sets (only among candidates)
    conflicts = [set() for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            a, b = candidates[i], candidates[j]
            if a + b in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_set(s):
        """Verify a set is valid (no pair sums to square, including self)"""
        s_list = sorted(s)
        for i in range(len(s_list)):
            a = s_list[i]
            if 2 * a in squares:
                return False
            for j in range(i + 1, len(s_list)):
                if a + s_list[j] in squares:
                    return False
        return True
    
    # Greedy: pick nodes in order of lowest degree, add if no conflict with already picked
    # Try multiple orderings
    
    def greedy_with_order(order):
        """order: list of candidate indices in priority order"""
        picked = set()
        for idx in order:
            if all(c not in picked for c in conflicts[idx]):
                picked.add(idx)
        return {candidates[i] for i in picked}
    
    # Initial greedy: sort by degree ascending
    degrees = [len(conflicts[i]) for i in range(n_cands)]
    order = sorted(range(n_cands), key=lambda i: (degrees[i], i))
    best_set = greedy_with_order(order)
    best_size = len(best_set)
    
    # Local search: try to add excluded nodes
    # For each excluded node, check if it conflicts with picked nodes
    # If it conflicts with only 1 picked node, try swapping
    
    def local_search(s, max_iter=5000):
        """Improve set s via add/swap moves"""
        s = set(s)
        s_set = s  # work with value set
        # Convert to idx set for efficiency
        idx_set = {cand_to_idx[v] for v in s_set if v in cand_to_idx}
        
        improved = True
        iters = 0
        while improved and iters < max_iter and elapsed() < DEADLINE:
            improved = False
            iters += 1
            
            # Try adding each excluded node
            for idx in range(n_cands):
                if idx in idx_set:
                    continue
                # Check conflicts with current set
                conflicting = conflicts[idx] & idx_set
                if not conflicting:
                    idx_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one picked, add one excluded
            # For each excluded node with exactly 1 conflict, try removing that conflict
            for idx in range(n_cands):
                if idx in idx_set:
                    continue
                conflicting = conflicts[idx] & idx_set
                if len(conflicting) == 1:
                    victim = next(iter(conflicting))
                    # Check: after removing victim, can we add idx?
                    # Also check victim isn't needed for others... just do simple swap
                    idx_set.discard(victim)
                    idx_set.add(idx)
                    improved = True
                    break
        
        return {candidates[i] for i in idx_set}
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # Try multiple greedy seeds with different orderings
    import random
    rng = random.Random(42)
    
    for seed_iter in range(20):
        if elapsed() > DEADLINE - 10:
            break
        
        # Different ordering strategies
        if seed_iter % 3 == 0:
            # Random perturbation of degree order
            order = sorted(range(n_cands), key=lambda i: (degrees[i] + rng.random() * 0.5, i))
        elif seed_iter % 3 == 1:
            # Reverse degree order (high degree first)
            order = sorted(range(n_cands), key=lambda i: (-degrees[i], i))
        else:
            # Random order
            order = list(range(n_cands))
            rng.shuffle(order)
        
        s = greedy_with_order(order)
        s = local_search(s, max_iter=2000)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Final: try to improve best_set with more aggressive local search
    if elapsed() < DEADLINE - 5:
        best_set = local_search(best_set, max_iter=10000)
        best_size = len(best_set)
    
    # Verify
    if not is_valid_set(best_set):
        # Fallback: simple valid set
        fallback = set()
        for a in range(1, N + 1):
            if 2 * a in squares:
                continue
            if all(a + b not in squares for b in fallback):
                fallback.add(a)
        best_set = fallback
        best_size = len(fallback)
    
    return sorted(best_set)
