"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    _t0 = time.time()
    _DEADLINE = 85.0
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def is_triangle_free(added):
        for u, v in added:
            if (u - v) % 2 == 0:
                return False
        return True
    
    def compute_diameter(added):
        edge_set = set(base_edges)
        for u, v in added:
            edge_set.add((min(u,v), max(u,v)))
        full_adj = [set() for _ in range(n)]
        for u, v in edge_set:
            full_adj[u].add(v)
            full_adj[v].add(u)
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nb in full_adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            for d in dist:
                if d == -1:
                    return float('inf')
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d

    def time_left():
        return (time.time() - _t0) < _DEADLINE

    # Known good: hub 12 with 10 edges
    initial_added = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    best = initial_added[:]

    # Try to reduce current best
    improved = True
    while improved and time_left():
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break

    # Systematic search for 2-hub solutions with minimum edges
    # For two hubs h1, h2 (both even), each odd vertex v must be within dist<=4 of all others
    # Key: odd vertex v needs edge to h1 or h2 (or is adjacent on path)
    # For diameter <= 4: any two odd vertices must share a common even neighbor within 2 steps
    # With hubs h1, h2: odd v connected to h1, odd w connected to h2 -> path v-h1-...-h2-w
    # dist(v,w) via hubs: 1 + dist(h1,h2) + 1 = 2 + dist(h1,h2) on path
    # So need dist_path(h1,h2) <= 2 for odd-odd pairs where one connects to each hub
    # But if both connect to same hub: dist = 2 (v-h1-w)
    
    # Better approach: enumerate all possible 2-hub configs and find min edges
    for h1 in range(0, n, 2):
        if not time_left():
            break
        for h2 in range(h1+2, n, 2):
            if not time_left():
                break
            # For each odd vertex, determine which hubs it CAN connect to (not adjacent on path)
            # and find minimum assignment
            odd_verts = list(range(1, n, 2))
            # For each odd v, can it connect to h1? (v not adjacent to h1 on path)
            # v adjacent to h1 on path if abs(v-h1)==1
            can_h1 = {v: (abs(v - h1) != 1) for v in odd_verts}
            can_h2 = {v: (abs(v - h2) != 1) for v in odd_verts}
            
            # Every odd vertex must be connected to at least one hub (or be adjacent to both hubs on path)
            # Actually, odd vertices adjacent to a hub on path don't need an edge to that hub
            # But they still need to be reachable within dist 4 of all other vertices
            
            # For minimum edges: try all subsets of edges (each odd vertex assigned to h1, h2, or both)
            # This is 3^12 which is too large. Instead, use the constraint:
            # dist between any two odd vertices <= 4
            
            # With hubs h1, h2: if odd v connects to h1 and odd w connects to h2,
            # dist(v,w) = 1 + dist_path(h1,h2) + 1 = 2 + |h1-h2|
            # Need this <= 4, so |h1-h2| <= 2
            # If |h1-h2| > 2, then every odd vertex must connect to the SAME hub
            # which means it's effectively a 1-hub solution
            
            if abs(h1 - h2) > 2:
                # Both hubs must cover all odd vertices, so min edges = min(odd not adj to h1, odd not adj to h2)
                # Actually each odd must connect to at least one, and if |h1-h2|>2,
                # cross-hub paths are too long, so effectively need all to connect to one hub
                # OR: odd v to h1, odd w to h2: dist = 2+|h1-h2| > 4 -> INVALID
                # So all odd must connect to same hub -> 1-hub case
                continue
            
            # |h1-h2| <= 2, so cross-hub paths work
            # Now find min edges: each odd v must connect to h1 or h2 (whichever it can)
            # Min edges = number of odd vertices that need at least one edge
            # An odd v needs NO edge if it's adjacent to BOTH h1 and h2 on path
            # (then it's within dist 1 of both hubs)
            
            edges_needed = 0
            trial = []
            for v in odd_verts:
                adj_h1 = (abs(v - h1) == 1)
                adj_h2 = (abs(v - h2) == 1)
                if adj_h1 and adj_h2:
                    # No edge needed
                    continue
                elif adj_h1:
                    # Already adjacent to h1, but needs to reach h2
                    # dist to h2: 1 (to h1) + |h1-h2| = 1+|h1-h2|
                    # For dist to odd w connected to h2: 1+|h1-h2|+1 = 2+|h1-h2| <= 4 OK
                    # So no edge to h2 needed if |h1-h2|<=2
                    continue
                elif adj_h2:
                    continue
                else:
                    # Needs edge to h1 or h2
                    edges_needed += 1
                    # Assign to closer hub for now
                    if abs(v - h1) <= abs(v - h2):
                        trial.append((h1, v))
                    else:
                        trial.append((h2, v))
            
            if len(trial) < len(best) and is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial[:]

    # Try 3-hub solutions
    for h1 in range(0, n, 2):
        if not time_left():
            break
        for h2 in range(h1+2, n, 2):
            if not time_left():
                break
            for h3 in range(h2+2, n, 2):
                if not time_left():
                    break
                odd_verts = list(range(1, n, 2))
                trial = []
                for v in odd_verts:
                    # v needs edge unless adjacent to at least one hub
                    hubs = [h1, h2, h3]
                    adj_any = False
                    for h in hubs:
                        if abs(v - h) == 1:
                            adj_any = True
                            break
                    if adj_any:
                        continue
                    # Assign to nearest hub
                    best_h = min(hubs, key=lambda h: abs(v - h))
                    trial.append((best_h, v))
                
                if len(trial) < len(best) and is_triangle_free(trial) and compute_diameter(trial) <= 4:
                    best = trial[:]

    # Try 4-hub solutions
    even_verts = list(range(0, n, 2))
    for hub_combo in combinations(even_verts, 4):
        if not time_left():
            break
        h1, h2, h3, h4 = hub_combo
        odd_verts = list(range(1, n, 2))
        trial = []
        for v in odd_verts:
            hubs = [h1, h2, h3, h4]
            adj_any = False
            for h in hubs:
                if abs(v - h) == 1:
                    adj_any = True
                    break
            if adj_any:
                continue
            best_h = min(hubs, key=lambda h: abs(v - h))
            trial.append((best_h, v))
        
        if len(trial) < len(best) and is_triangle_free(trial) and compute_diameter(trial) <= 4:
            best = trial[:]

    # Final reduction pass
    improved = True
    while improved and time_left():
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break

    # Fallback
    if not best or not (is_triangle_free(best) and compute_diameter(best) <= 4):
        best = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]

    # Final verification
    base_set = set(base_edges)
    seen = set()
    final = []
    for u, v in best:
        a, b = min(u, v), max(u, v)
        if (a, b) in base_set or (a, b) in seen:
            continue
        seen.add((a, b))
        final.append((a, b))
    if not (is_triangle_free(final) and compute_diameter(final) <= 4):
        final = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]

    result = [[int(u), int(v)] for u, v in final]
    return result
# EVOLVE-BLOCK-END
