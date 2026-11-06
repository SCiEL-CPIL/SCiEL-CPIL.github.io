import random
import time
from itertools import combinations
import sys

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    triple_to_kconfigs = {t: [] for t in all_triples}
    kconfigs = []
    
    def partitions_into_pairs(points):
        a = points[0]
        for i in range(1, 6):
            b = points[i]
            remaining = [p for j, p in enumerate(points) if j not in (0, i)]
            c, d, e, f = remaining
            yield (a, b), (c, d), (e, f)
            yield (a, b), (c, e), (d, f)
            yield (a, b), (c, f), (d, e)
    
    for six_points in combinations(range(n), 6):
        for pairs in partitions_into_pairs(six_points):
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            transversals_canon = set()
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        transversals_canon.add(tuple(sorted((x, y, z))))
            kconfigs.append(transversals_canon)
            for t in transversals_canon:
                triple_to_kconfigs[t].append(len(kconfigs) - 1)
    
    num_kconfigs = len(kconfigs)

    # 1. Greedy Initialization to get a strong lower bound
    def greedy_solve(order):
        present = set()
        kconfig_counts = [0] * num_kconfigs
        for t in order:
            can_add = True
            for kidx in triple_to_kconfigs[t]:
                if kconfig_counts[kidx] == 7:
                    can_add = False
                    break
            if not can_add:
                continue
            present.add(t)
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] += 1
        return present, kconfig_counts

    # Run a few greedy passes with different shuffles to get a good initial bound
    best_count = 0
    best_triples = None
    random.seed(42)
    
    for _ in range(10):
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        present, _ = greedy_solve(shuffled)
        if len(present) > best_count:
            best_count = len(present)
            best_triples = [list(t) for t in present]
            
    # 2. Branch and Bound
    # Order triples by "constraint degree" (number of kconfigs they touch) descending
    # This prunes the search tree earlier
    triple_degrees = [(len(triple_to_kconfigs[t]), t) for t in all_triples]
    triple_degrees.sort(reverse=True)
    ordered_triples = [t for _, t in triple_degrees]
    
    # Map triple to its index in ordered_triples for quick access? 
    # We just iterate through ordered_triples.
    
    start_time = time.time()
    time_limit = 110.0
    
    # State for B&B
    # We need to track which triples are included to check constraints quickly.
    # Since we go in order, we can maintain a list of included triples.
    # But checking constraints requires knowing counts of kconfigs.
    # We can maintain `kconfig_counts` incrementally.
    
    # To speed up B&B, we can use a bitmask or boolean array for included triples?
    # With 560 triples, a boolean array is fine.
    
    included = [False] * num_triples
    # Map original triple to index in all_triples?
    # all_triples is a list. We can create a dict for O(1) lookup of index.
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # We will use the ordered_triples list.
    # But we need to update kconfig_counts.
    
    def branch_and_bound(idx, current_count, kconfig_counts):
        nonlocal best_count, best_triples
        
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Upper bound check
        # Max possible additional triples is num_triples - idx
        if current_count + (num_triples - idx) <= best_count:
            return
            
        if idx == num_triples:
            if current_count > best_count:
                best_count = current_count
                # Reconstruct solution
                sol = []
                for i in range(num_triples):
                    if included[i]:
                        sol.append(list(all_triples[i]))
                best_triples = sol
            return
        
        t = ordered_triples[idx]
        t_idx = triple_index[t]
        
        # Option 1: Include t
        # Check if valid
        can_include = True
        for kidx in triple_to_kconfigs[t]:
            if kconfig_counts[kidx] == 7:
                can_include = False
                break
        
        if can_include:
            # Add t
            included[t_idx] = True
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] += 1
                
            branch_and_bound(idx + 1, current_count + 1, kconfig_counts)
            
            # Undo
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] -= 1
            included[t_idx] = False
            
        # Option 2: Exclude t
        # Optimization: If including t was possible, we tried it.
        # We should also try excluding it to see if a better solution exists
        # where this slot is empty but later slots are filled.
        branch_and_bound(idx + 1, current_count, kconfig_counts)

    # Initialize state
    initial_kconfig_counts = [0] * num_kconfigs
    
    # To help B&B, we can start with the greedy solution's "included" status?
    # No, B&B explores the space. But we can use the greedy solution to set
    # a lower bound. The B&B will try to beat it.
    
    # However, B&B might be too slow if it doesn't prune well.
    # We can add a heuristic: if we are deep in the tree and current_count is low,
    # we might prune.
    
    # Let's run B&B.
    # Note: Recursion depth is 560. Python default is 1000. We are safe.
    
    # One issue: The order of "Include" vs "Exclude".
    # If we include first, we find a solution quickly (likely the greedy one or similar).
    # Then we backtrack to try to improve.
    
    # To ensure we find a good solution fast, we can bias the order.
    # But the standard B&B with "Include" first should find a valid maximal set quickly.
    
    # Let's add a small local search at the end if time permits?
    # No, the directive says "new program... from scratch". 
    # The B&B is the core.
    
    # To make B&B more effective, we can use "Forward Checking":
    # Before recursing, check if any future triple is *blocked* by the current choices?
    # That's expensive.
    
    # Let's just run the B&B.
    # We need to be careful with the time limit.
    
    # To prevent stack overflow or excessive time in a single deep path,
    # we rely on the time check.
    
    # Let's run it.
    branch_and_bound(0, 0, initial_kconfig_counts)
    
    return best_triples
