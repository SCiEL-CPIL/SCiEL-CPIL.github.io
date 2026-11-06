"""Minimum-bottleneck Hamilton cycle via min-cost perfect matching on 2n ports."""
import time
import random
import sys

def construct():
    # Edge arrival indices (1-based)
    edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]
    
    n = 24
    # Build cost matrix: cost[i][j] = arrival index of edge (i,j)
    cost = [[0] * n for _ in range(n)]
    for idx, (a, b) in enumerate(edge_order, 1):
        cost[a][b] = idx
        cost[b][a] = idx
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Greedy + local search approach
    # First, try to find a good Hamilton cycle using greedy construction
    # then improve with 2-opt and Or-opt moves
    
    def greedy_cycle(seed=42):
        """Build a Hamilton cycle greedily, preferring early-arriving edges."""
        rng = random.Random(seed)
        visited = [False] * n
        cycle = [0]
        visited[0] = True
        
        for _ in range(n - 1):
            last = cycle[-1]
            # Sort unvisited vertices by cost from last
            candidates = [(cost[last][v], v) for v in range(n) if not visited[v]]
            candidates.sort()
            # Pick the best (earliest arrival)
            _, next_v = candidates[0]
            cycle.append(next_v)
            visited[next_v] = True
        
        return cycle
    
    def cycle_score(cycle):
        """Compute the max arrival index of edges in the cycle."""
        max_idx = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            max_idx = max(max_idx, cost[u][v])
        return max_idx
    
    def two_opt(cycle, max_iters=10000):
        """Improve cycle using 2-opt moves."""
        best = cycle[:]
        best_score = cycle_score(best)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n - 1):
                if time.time() > deadline:
                    return best
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue  # This would be the same as no move
                    # Remove edges (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                    # Add edges (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                    a, b = best[i], best[i + 1]
                    c, d = best[j], best[(j + 1) % n]
                    old_cost = cost[a][b] + cost[c][d]
                    new_cost = cost[a][c] + cost[b][d]
                    if new_cost < old_cost:
                        # Apply 2-opt
                        best[i + 1:j + 1] = reversed(best[i + 1:j + 1])
                        best_score = cycle_score(best)
                        improved = True
                        break
                if improved:
                    break
        return best
    
    def or_opt(cycle, max_iters=10000):
        """Improve cycle using Or-opt moves (relocate segments of length 1-3)."""
        best = cycle[:]
        best_score = cycle_score(best)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    if time.time() > deadline:
                        return best
                    # Remove segment starting at i of length seg_len
                    if seg_len >= n:
                        continue
                    # The segment is best[i:i+seg_len]
                    # Edges to remove: (best[i-1], best[i]) and (best[i+seg_len-1], best[i+seg_len])
                    prev_idx = (i - 1) % n
                    next_idx = (i + seg_len) % n
                    # If removing from position 0, prev is n-1
                    a = best[prev_idx]
                    b = best[i]
                    c = best[(i + seg_len - 1) % n]
                    d = best[next_idx]
                    
                    if seg_len == n:
                        continue
                    
                    old_cost = cost[a][b] + cost[c][d]
                    
                    # Try inserting segment at each position
                    for j in range(n - seg_len):
                        if j == i or j == (i + seg_len) % n:
                            continue
                        # Insert after position j
                        # New edges: (best[j], best[i]) and (best[i+seg_len-1], best[(j+1)%n])
                        # But we need to be careful about the cycle structure
                        # Let's just try all insertions
                        new_cycle = best[:i] + best[i+seg_len:]
                        # Insert segment at position j in new_cycle
                        if j > i:
                            j -= seg_len
                        seg = best[i:i+seg_len]
                        candidate = new_cycle[:j] + seg + new_cycle[j:]
                        # Check it's a valid permutation
                        if len(candidate) != n:
                            continue
                        if len(set(candidate)) != n:
                            continue
                        cand_score = cycle_score(candidate)
                        if cand_score < best_score:
                            best = candidate
                            best_score = cand_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best
    
    # Try multiple seeds and find the best
    best_cycle = None
    best_score = float('inf')
    
    for seed in range(200):
        if time.time() > deadline:
            break
        cycle = greedy_cycle(seed)
        cycle = two_opt(cycle, max_iters=5000)
        cycle = or_opt(cycle, max_iters=2000)
        s = cycle_score(cycle)
        if s < best_score:
            best_score = s
            best_cycle = cycle[:]
    
    # Also try a few random starting points
    rng = random.Random(12345)
    for _ in range(50):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        # Build cycle starting from random vertex
        visited = [False] * n
        cycle = [start]
        visited[start] = True
        for _ in range(n - 1):
            last = cycle[-1]
            candidates = [(cost[last][v], v) for v in range(n) if not visited[v]]
            candidates.sort()
            _, next_v = candidates[0]
            cycle.append(next_v)
            visited[next_v] = True
        
        cycle = two_opt(cycle, max_iters=5000)
        cycle = or_opt(cycle, max_iters=2000)
        s = cycle_score(cycle)
        if s < best_score:
            best_score = s
            best_cycle = cycle[:]
    
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle
