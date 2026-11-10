import time
import random
import numpy as np
from collections import defaultdict

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    def can_add_fast(chosen_list, counts, x):
        """Check if x can be added. chosen_list is sorted for efficiency."""
        if counts[2 * x] >= 2:
            return False
        for y in chosen_list:
            if y != x and counts[x + y] >= 2:
                return False
        return True
    
    def add_to_set(chosen_list, chosen_set, counts, x):
        """Add x to set, updating counts."""
        for y in chosen_list:
            if y != x:
                counts[x + y] += 1
        counts[2 * x] += 1
        chosen_list.append(x)
        chosen_set.add(x)
    
    def remove_from_set(chosen_list, chosen_set, counts, x):
        """Remove x from set, updating counts."""
        for y in chosen_list:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen_list.remove(x)
        chosen_set.discard(x)
    
    def greedy_build(order, counts_init=None):
        """Greedy construction with given order."""
        chosen_list = []
        chosen_set = set()
        counts = [0] * (max_sum + 1) if counts_init is None else counts_init[:]
        
        for x in order:
            if x in chosen_set:
                continue
            if can_add_fast(chosen_list, counts, x):
                add_to_set(chosen_list, chosen_set, counts, x)
        
        return chosen_list, chosen_set, counts
    
    def local_search(chosen_list, chosen_set, counts, time_limit=5.0):
        """Efficient local search with targeted swaps."""
        ts = time.time()
        improved = True
        
        while improved and (time.time() - ts) < time_limit:
            improved = False
            
            # Phase 1: Try insertions (add without removing)
            # Focus on candidates that are "promising" - not in set, and their sums aren't blocked
            candidate_pool = []
            for x in range(1, N + 1):
                if x not in chosen_set:
                    # Quick check: is 2x already blocked?
                    if counts[2 * x] < 2:
                        candidate_pool.append(x)
            
            random.shuffle(candidate_pool)
            
            for y in candidate_pool:
                if time.time() - ts > time_limit:
                    break
                if can_add_fast(chosen_list, counts, y):
                    add_to_set(chosen_list, chosen_set, counts, y)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps - remove one, add another
            # Sample a few elements to try removing
            if len(chosen_list) > 0:
                sample_size = min(len(chosen_list), 30)
                sampled_indices = random.sample(range(len(chosen_list)), sample_size)
                
                for idx in sampled_indices:
                    if time.time() - ts > time_limit:
                        break
                    
                    x = chosen_list[idx]
                    
                    # Temporarily remove x
                    remove_from_set(chosen_list, chosen_set, counts, x)
                    
                    # Try to find a replacement
                    # Focus on candidates that were previously blocked by x
                    # or just try a random subset
                    candidate_pool = []
                    for y in range(1, N + 1):
                        if y not in chosen_set:
                            if counts[2 * y] < 2:
                                candidate_pool.append(y)
                    
                    random.shuffle(candidate_pool)
                    
                    found = False
                    for y in candidate_pool:
                        if time.time() - ts > time_limit:
                            break
                        if can_add_fast(chosen_list, counts, y):
                            add_to_set(chosen_list, chosen_set, counts, y)
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Restore x
                        add_to_set(chosen_list, chosen_set, counts, x)
                    
                    if improved:
                        break
        
        return chosen_list, chosen_set, counts
    
    def generate_orders(rng, strategy):
        """Generate candidate orderings."""
        if strategy == 'ascending':
            return list(range(1, N + 1))
        elif strategy == 'descending':
            return list(range(N, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            return candidates
        elif strategy == 'sparse_first':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            return odds + evens
        elif strategy == 'weighted_small':
            candidates = list(range(1, N + 1))
            # Bias towards smaller numbers
            weights = [1.0 / (1 + i * 0.005) for i in range(N)]
            indexed = list(zip(weights, candidates))
            rng.shuffle(indexed)
            return [x[1] for x in indexed]
        elif strategy == 'mod_based':
            # Group by mod 3, shuffle within groups
            groups = [[] for _ in range(3)]
            for i in range(1, N + 1):
                groups[i % 3].append(i)
            for g in groups:
                rng.shuffle(g)
            return groups[1] + groups[2] + groups[0]
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            return candidates
    
    # Strategy 1: Quick greedy baselines
    best_list = []
    best_set = set()
    best_counts = [0] * (max_sum + 1)
    best_len = 0
    
    # Try a few quick greedy constructions
    quick_strategies = ['ascending', 'descending', 'random', 'sparse_first']
    rng = random.Random(42)
    
    for strat in quick_strategies:
        if time.time() > deadline - 30:
            break
        order = generate_orders(rng, strat)
        cl, cs, cc = greedy_build(order)
        if len(cl) > best_len:
            best_len = len(cl)
            best_list = cl[:]
            best_set = cs.copy()
            best_counts = cc[:]
    
    # Strategy 2: Local search on best
    if best_list and time.time() < deadline - 10:
        remaining = deadline - time.time()
        cl, cs, cc = local_search(best_list[:], best_set.copy(), best_counts[:], time_limit=min(remaining, 20.0))
        if len(cl) > best_len:
            best_len = len(cl)
            best_list = cl[:]
            best_set = cs.copy()
            best_counts = cc[:]
    
    # Strategy 3: Multiple restarts with local search
    while time.time() < deadline - 5:
        strat = random.choice(['random', 'sparse_first', 'weighted_small', 'mod_based'])
        order = generate_orders(rng, strat)
        cl, cs, cc = greedy_build(order)
        
        # Quick local search
        remaining = deadline - time.time() - 2
        if remaining > 2:
            cl, cs, cc = local_search(cl, cs, cc, time_limit=min(remaining, 8.0))
        
        if len(cl) > best_len:
            best_len = len(cl)
            best_list = cl[:]
            best_set = cs.copy()
            best_counts = cc[:]
    
    return best_list
