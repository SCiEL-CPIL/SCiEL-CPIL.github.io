import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Step 1: Filter candidates based on diagonal constraint (2*a not square)
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    
    # Build adjacency list for the conflict graph
    # conflict[a] is a set of b such that a+b is a perfect square
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    def greedy_min_degree(candidates, conflict):
        """Greedy: repeatedly pick the vertex with minimum degree in remaining graph."""
        remaining = set(candidates)
        selected = []
        while remaining:
            # Find vertex with minimum degree among remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflict[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            selected.append(best_v)
            remaining.remove(best_v)
        return selected
    
    def greedy_randomized(candidates, conflict, alpha=0.5):
        """Greedy with randomization: pick from top alpha fraction of min-degree vertices."""
        remaining = set(candidates)
        selected = []
        while remaining:
            # Compute degrees
            degrees = {}
            for v in remaining:
                degrees[v] = len(conflict[v] & remaining)
            min_deg = min(degrees.values())
            # Vertices within alpha fraction of min degree
            threshold = min_deg + alpha * (max(degrees.values()) - min_deg)
            eligible = [v for v in remaining if degrees[v] <= threshold]
            best_v = random.choice(eligible)
            selected.append(best_v)
            remaining.remove(best_v)
        return selected
    
    def local_search_improve(selected, candidates, conflict):
        """Try to improve by adding vertices that don't conflict, then try swaps."""
        selected_set = set(selected)
        # First pass: add any non-conflicting vertex
        changed = True
        while changed:
            changed = False
            for a in candidates:
                if a not in selected_set:
                    if not (conflict[a] & selected_set):
                        selected.append(a)
                        selected_set.add(a)
                        changed = True
        # Second pass: try single swaps
        improved = False
        for i in range(len(selected)):
            rem = selected[i]
            remaining = [x for x in selected if x != rem]
            remaining_set = set(remaining)
            new_added = []
            for a in candidates:
                if a not in remaining_set and a != rem:
                    if not (conflict[a] & remaining_set):
                        new_added.append(a)
            if len(remaining) + len(new_added) > len(selected):
                selected = remaining + new_added
                selected_set = set(selected)
                improved = True
        return selected
    
    # Run multiple strategies
    results = []
    
    # Strategy 1: Greedy min-degree
    r1 = greedy_min_degree(candidates, conflict)
    r1 = local_search_improve(r1, candidates, conflict)
    results.append(r1)
    
    # Strategy 2: Greedy randomized with multiple restarts
    best_rand = []
    num_restarts = 30
    for _ in range(num_restarts):
        r = greedy_randomized(candidates, conflict, alpha=0.3)
        r = local_search_improve(r, candidates, conflict)
        if len(r) > len(best_rand):
            best_rand = r
    results.append(best_rand)
    
    # Strategy 3: Greedy by ascending conflict count (original)
    by_conflict_asc = sorted(candidates, key=lambda x: len(conflict[x]))
    r3 = []
    selected_set = set()
    for a in by_conflict_asc:
        if not (conflict[a] & selected_set):
            r3.append(a)
            selected_set.add(a)
    r3 = local_search_improve(r3, candidates, conflict)
    results.append(r3)
    
    # Strategy 4: Greedy by descending conflict count
    by_conflict_desc = sorted(candidates, key=lambda x: -len(conflict[x]))
    r4 = []
    selected_set = set()
    for a in by_conflict_desc:
        if not (conflict[a] & selected_set):
            r4.append(a)
            selected_set.add(a)
    r4 = local_search_improve(r4, candidates, conflict)
    results.append(r4)
    
    # Pick the best so far
    best = max(results, key=len)
    
    # Check time
    if time.time() - start_time > deadline - 15:
        return best
    
    # More randomized restarts with different alpha values
    for alpha in [0.1, 0.2, 0.4, 0.6, 0.8]:
        if time.time() - start_time > deadline - 10:
            break
        for _ in range(10):
            r = greedy_randomized(candidates, conflict, alpha=alpha)
            r = local_search_improve(r, candidates, conflict)
            if len(r) > len(best):
                best = r
    
    # Try 2-swap local search: remove two elements, add back if net gain
    if time.time() - start_time > deadline - 5:
        return best
    
    best_set = set(best)
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            if time.time() - start_time > deadline - 2:
                break
            rem1, rem2 = best[i], best[j]
            remaining = [x for x in best if x != rem1 and x != rem2]
            remaining_set = set(remaining)
            new_added = []
            for a in candidates:
                if a not in remaining_set and a != rem1 and a != rem2:
                    if not (conflict[a] & remaining_set):
                        new_added.append(a)
            if len(remaining) + len(new_added) > len(best):
                best = remaining + new_added
                best_set = set(best)
    
    return best
