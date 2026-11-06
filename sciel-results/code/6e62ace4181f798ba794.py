import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Perfect squares up to 512
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Diagonal constraint: 2a must not be a perfect square
    # 2a = k^2 => a = k^2/2, k must be even: a = 2*m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
    
    # Candidates: numbers not violating diagonal
    candidates = [a for a in range(1, n + 1) if a not in diagonal_bad]
    candidate_set = set(candidates)
    
    # Build conflict graph: edge between a,b (a<b) if a+b is a perfect square
    # For each candidate, find its conflicts
    conflicts = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_valid(s):
        s_set = set(s)
        for a in s:
            if 2 * a in squares:
                return False
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                if a + b in squares:
                    return False
        return True
    
    def score(s):
        return len(s)
    
    # Greedy: sort by degree (number of conflicts), pick lowest degree first
    # This tends to give a good independent set
    def greedy_is():
        remaining = set(candidates)
        result = []
        # Use a priority: at each step, pick the vertex with minimum conflicts in remaining
        while remaining:
            # Find vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if best_deg == 0:
                        break
            result.append(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v from remaining
            for nb in conflicts[best_v]:
                remaining.discard(nb)
        return result
    
    # Initial solution
    best = greedy_is()
    best_set = set(best)
    
    # Local search: try to add elements, try swaps
    def try_add(s_set):
        # Try adding each candidate not in s_set
        added = False
        for a in candidates:
            if a in s_set:
                continue
            if all((a + b) not in squares for b in s_set):
                s_set.add(a)
                added = True
        return added
    
    def local_search(s_set, time_limit):
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try adding
            for a in list(candidates):
                if a in s_set:
                    continue
                if all((a + b) not in squares for b in s_set):
                    s_set.add(a)
                    improved = True
            # Try swaps: remove one, add one or more
            if (time.time() - start_time) >= time_limit:
                break
            for a in list(s_set):
                s_set.remove(a)
                # Try to add back multiple
                added_any = False
                for b in candidates:
                    if b in s_set:
                        continue
                    if all((b + c) not in squares for c in s_set):
                        s_set.add(b)
                        added_any = True
                if not added_any:
                    s_set.add(a)  # put back
                else:
                    improved = True
        return s_set
    
    best_set = local_search(best_set, deadline - 5)
    
    # Multiple restarts with randomization
    best_size = len(best_set)
    
    while (time.time() - start_time) < deadline - 5:
        # Randomized greedy
        random.seed(int(time.time() * 1000) % (2**31))
        remaining = set(candidates)
        result = []
        while remaining:
            # Pick randomly among lowest-degree vertices
            min_deg = min(len(conflicts[v] & remaining) for v in remaining)
            low_deg = [v for v in remaining if len(conflicts[v] & remaining) == min_deg]
            v = random.choice(low_deg)
            result.append(v)
            remaining.discard(v)
            for nb in conflicts[v]:
                remaining.discard(nb)
        
        r_set = set(result)
        r_set = local_search(r_set, deadline - 5)
        
        if len(r_set) > best_size:
            best_size = len(r_set)
            best_set = r_set
    
    return list(best_set)
