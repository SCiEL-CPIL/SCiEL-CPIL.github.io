import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    N = 1000
    
    def greedy_build(order):
        """Build set greedily following given order."""
        S = set()
        counts = np.zeros(2*N+1, dtype=np.int32)
        for x in order:
            valid = True
            # Check all sums x+y for y in S
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid and counts[2*x] >= 2:
                valid = False
            if valid:
                S.add(x)
                counts[2*x] += 1
                for y in S:
                    if y == x:
                        continue
                    s = x + y
                    counts[s] += 1
        return S, counts
    
    def try_insert(S, counts, x, N):
        """Try to insert x into S. Returns True if successful."""
        if x in S:
            return False
        # Check validity
        for y in S:
            s = x + y
            if counts[s] >= 2:
                return False
        if counts[2*x] >= 2:
            return False
        # Insert
        S.add(x)
        counts[2*x] += 1
        for y in S:
            if y == x:
                continue
            s = x + y
            counts[s] += 1
        return True
    
    def remove_from_S(S, counts, x):
        """Remove x from S and update counts."""
        S.remove(x)
        counts[2*x] -= 1
        for y in S:
            s = x + y
            counts[s] -= 1
    
    def local_search(S, counts, time_limit):
        """Local search: try insertions, then swaps."""
        best_S = frozenset(S)
        best_size = len(S)
        current_S = set(S)
        current_counts = counts.copy()
        
        random.shuffle(list(range(N+1)))
        candidates_order = [x for x in range(1, N+1) if x not in current_S]
        random.shuffle(candidates_order)
        
        while time.time() < time_limit:
            made_progress = False
            
            # Phase 1: Try to insert any candidate
            for x in candidates_order:
                if time.time() > time_limit:
                    break
                if x in current_S:
                    continue
                if try_insert(current_S, current_counts, x, N):
                    made_progress = True
                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
                    # Remove from candidates
                    try:
                        candidates_order.remove(x)
                    except ValueError:
                        pass
            
            # Phase 2: If no insertions, try swaps
            if not made_progress:
                # Try random swap: remove one element, try to add another
                if current_S:
                    x_remove = random.choice(list(current_S))
                    remove_from_S(current_S, current_counts, x_remove)
                    
                    # Try to add a different element
                    swapped = False
                    for x_add in candidates_order:
                        if time.time() > time_limit:
                            break
                        if x_add in current_S:
                            continue
                        if try_insert(current_S, current_counts, x_add, N):
                            swapped = True
                            if len(current_S) > best_size:
                                best_S = frozenset(current_S)
                                best_size = len(current_S)
                            # Update candidates
                            if x_remove not in candidates_order:
                                candidates_order.append(x_remove)
                            try:
                                candidates_order.remove(x_add)
                            except ValueError:
                                pass
                            break
                    
                    if not swapped:
                        # Put x_remove back
                        try_insert(current_S, current_counts, x_remove, N)
            
            # If no progress after a full pass, break
            if not made_progress:
                break
        
        return best_S, best_size
    
    best_overall = set()
    best_overall_size = 0
    
    # Multiple greedy strategies
    strategies = []
    
    # 1. Ascending
    strategies.append(list(range(1, N+1)))
    
    # 2. Descending
    strategies.append(list(range(N, 0, -1)))
    
    # 3-8. Weighted random orderings
    for i in range(6):
        random.seed(100 + i * 7)
        order = list(range(1, N+1))
        # Weight towards sparse regions
        weights = [1.0 / (1 + abs(x - 500)/500) for x in order]
        # Use random selection with weights
        selected = []
        remaining = list(range(1, N+1))
        remaining_weights = [1.0 / (1 + abs(x - 500)/500) for x in remaining]
        for _ in range(N):
            total_w = sum(remaining_weights)
            r = random.random() * total_w
            cum = 0
            for idx, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    selected.append(remaining[idx])
                    remaining.pop(idx)
                    remaining_weights.pop(idx)
                    break
        strategies.append(selected)
    
    # 9-12. Sparse-focused: prioritize numbers with fewer small-sum conflicts
    for i in range(4):
        random.seed(200 + i * 13)
        order = list(range(1, N+1))
        random.shuffle(order)
        # Sort by: prefer numbers that are far from each other
        # Use a heuristic: sort by random key but favor spread
        keys = [random.random() * 0.5 + abs(x - 500)/1000 for x in order]
        order.sort(key=lambda x: keys[order.index(x)])
        strategies.append(order)
    
    # Run each strategy
    for i, order in enumerate(strategies):
        if time.time() > deadline - 10:
            break
        
        S, counts = greedy_build(order)
        size = len(S)
        
        if size > best_overall_size:
            best_overall = set(S)
            best_overall_size = size
        
        # Local search with time budget
        remaining_time = deadline - time.time() - 2  # 2s safety
        if remaining_time > 5:
            time_per_ls = max(2, remaining_time / max(1, len(strategies) - i))
            ls_start = time.time()
            best_S_ls, best_size_ls = local_search(S, counts, ls_start + time_per_ls)
            
            if best_size_ls > best_overall_size:
                best_overall = set(best_S_ls)
                best_overall_size = best_size_ls
    
    # Final: try to improve best_overall with more local search
    if time.time() < deadline - 5:
        # Rebuild counts for best_overall
        counts_final = np.zeros(2*N+1, dtype=np.int32)
        for x in best_overall:
            counts_final[2*x] += 1
            for y in best_overall:
                if y == x:
                    continue
                s = x + y
                counts_final[s] += 1
        
        ls_start = time.time()
        best_S_final, best_size_final = local_search(set(best_overall), counts_final, deadline - 3)
        
        if best_size_final > best_overall_size:
            best_overall = set(best_S_final)
            best_overall_size = best_size_final
    
    return list(best_overall)
