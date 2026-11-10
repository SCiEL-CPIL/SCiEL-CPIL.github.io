"""Solve: fewest colors for degree-4 graph with no bicolored cycles."""

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

import time
import random
import sys
from collections import defaultdict

DEADLINE = time.time() + 50  # Safety margin


def _is_valid(colors, edges):
    """Check proper coloring + no bicolored cycles via union-find."""
    n = len(colors)
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check no bicolored cycles
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu > cv:
            cu, cv = cv, cu
        pair_edges[(cu, cv)].append((u, v))
    
    # For each color pair, check if the subgraph is a forest
    for (cu, cv), pel in pair_edges.items():
        if not pel:
            continue
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        for u, v in pel:
            if not union(u, v):
                return False
    return True


def _count_distinct(colors):
    return len(set(colors))


def construct():
    n = P['n']
    edges = P['edges']
    random.seed(42)

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Initial valid coloring: Use distinct colors for all vertices to be safe?
    # No, that's 50 colors. We want fewer.
    # Let's try a greedy coloring with a small number of colors first.
    # If we can't find a valid one with K colors, we increase K.
    
    best_colors = None
    best_k = n + 1

    # Try to find a valid coloring with k colors
    # Start with a reasonable guess, e.g., 4 or 5.
    # If that fails, go up.
    
    # Strategy: 
    # 1. Generate a random proper coloring (ignoring bicolored cycles) with k colors.
    # 2. Use local search to fix bicolored cycles and reduce color count.
    
    # Actually, let's just do a local search starting from a "high" color count 
    # that is definitely proper, then reduce.
    # Or start from a greedy proper coloring with ~5 colors.
    
    # Let's try multiple restarts with different starting K.
    
    for max_k in range(3, 8):
        if time.time() > DEADLINE:
            break
            
        found_valid = False
        for restart in range(100):
            if time.time() > DEADLINE:
                break
                
            # Random greedy proper coloring
            colors = [-1] * n
            valid_greedy = True
            order = list(range(n))
            random.shuffle(order)
            
            for v in order:
                used = set(colors[u] for u in adj[v] if colors[u] != -1)
                avail = [c for c in range(max_k) if c not in used]
                if not avail:
                    valid_greedy = False
                    break
                colors[v] = random.choice(avail)
            
            if not valid_greedy:
                continue
                
            # Now we have a proper coloring with <= max_k colors.
            # Check if it satisfies the bicolored cycle constraint.
            if _is_valid(colors, edges):
                k = _count_distinct(colors)
                if k < best_k:
                    best_k = k
                    best_colors = list(colors)
                found_valid = True
                
                # Try to improve (reduce k) via local search
                # We can try to recolor vertices to eliminate rare colors
                current_colors = list(colors)
                current_k = k
                improved = True
                while improved and time.time() < DEADLINE:
                    improved = False
                    # Count frequencies
                    from collections import Counter
                    cnt = Counter(current_colors)
                    # Sort colors by frequency
                    sorted_colors = sorted(cnt.keys(), key=lambda c: cnt[c])
                    
                    # Try to eliminate the least frequent color
                    for c_to_elim in sorted_colors:
                        if cnt[c_to_elim] == 0:
                            continue
                        if time.time() > DEADLINE:
                            break
                            
                        # Try to recolor all vertices with c_to_elim to other colors
                        verts = [v for v in range(n) if current_colors[v] == c_to_elim]
                        random.shuffle(verts)
                        
                        save = list(current_colors)
                        success = True
                        for v in verts:
                            # Find a valid new color
                            # Must not be c_to_elim (we are eliminating it)
                            # Must not be neighbor's color
                            used = set(current_colors[u] for u in adj[v])
                            avail = [c for c in range(max_k) if c not in used and c != c_to_elim]
                            if not avail:
                                success = False
                                break
                            current_colors[v] = random.choice(avail)
                        
                        if success and _is_valid(current_colors, edges):
                            new_k = _count_distinct(current_colors)
                            if new_k < best_k:
                                best_k = new_k
                                best_colors = list(current_colors)
                            improved = True
                            break # Restart loop with new state
                        else:
                            current_colors = save
                            # If we failed to eliminate this color, try next
                            # But we might be stuck. 
                            # For now, just break the while loop if we can't improve easily.
                            # To avoid infinite loops, we only accept if valid.
                            # If we can't eliminate any color, we stop improving this instance.
                            pass 
                    if not improved:
                        break

            # If we found a valid one, we can break restarts for this max_k 
            # if we found a very good one? 
            # Actually, we want the global minimum. 
            # But if we found k=3, we can't do better than 3 (unless 2, which is impossible for 4-reg non-bipartite).
            # Let's check if best_k <= max_k. If so, we might still find fewer in other restarts? 
            # No, we are iterating max_k upwards. 
            # If we found a valid coloring with max_k=3, best_k is 3. 
            # We should still check max_k=4,5... to see if we can find a coloring with *fewer* distinct colors?
            # No, if we used 3 colors, distinct count is <= 3. 
            # The loop `for max_k in range(3, 8)` tries to find *any* valid coloring using at most `max_k` colors.
            # The objective is to minimize *distinct* colors used.
            # So if we find a valid coloring with max_k=5 that uses only 3 distinct colors, that's great.
            # But if we find a valid coloring with max_k=3, it uses at most 3.
            # So we should prioritize finding valid colorings with small max_k?
            # Actually, the number of distinct colors is bounded by max_k.
            # So if we find a valid coloring with max_k=4, it might use 4 colors.
            # If we find a valid coloring with max_k=5, it might use 3 colors (if one color is unused in the greedy step, or we eliminate it).
            # So we should run local search aggressively to reduce distinct colors.
            
            # To save time, if we have found a solution with best_k <= 3, we can stop early?
            # 3 is likely the minimum for this graph (4-regular, not bipartite).
            if best_k <= 3:
                return [int(c) for c in best_colors]

    if best_colors is None:
        # Fallback: Greedy with many colors
        colors = [-1] * n
        for v in range(n):
            used = set(colors[u] for u in adj[v] if colors[u] != -1)
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return [int(c) for c in colors]
        
    return [int(c) for c in best_colors]
