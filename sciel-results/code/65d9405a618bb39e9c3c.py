"""Triangle-free diameter-four augmentation via dominating set approach."""
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 24
    t0 = time.time()
    deadline = t0 + 80
    
    # Base path edges
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # All valid added edges (even-odd pairs not in base path)
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    all_edges = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e, o), max(e, o))
            if key not in base_edges:
                all_edges.append([e, o])
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for a in range(n):
            for b in adj[a]:
                if b > a:
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            return False
        return True
    
    def bfs_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for i in range(n):
                if dist[i] == -1:
                    return -1
            max_d = max(max_d, max(dist.values()))
        return max_d
    
    def validate(added_edges):
        # Check no duplicates, no existing edges, valid vertices
        seen = set()
        for u, v in added_edges:
            if not (isinstance(u, int) and isinstance(v, int)):
                return False
            if not (0 <= u < n and 0 <= v < n) or u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in base_edges or key in seen:
                return False
            seen.add(key)
        
        adj = build_adj(added_edges)
        
        # Triangle-free
        if not is_triangle_free(adj):
            return False
        
        # Diameter <= 4
        if bfs_diameter(adj) > 4:
            return False
        
        return True
    
    # Initial known good solution
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    best = [list(e) for e in fallback]
    best_val = len(best)
    
    # Strategy 1: Dominating set approach
    # In P24^2, vertex u dominates {u-2, u-1, u, u+1, u+2}
    # We want to find a set of "hubs" such that every vertex is within distance 2
    # of some hub or its path neighbor.
    
    # For diameter <= 4, we need: for every pair (u, v), dist(u, v) <= 4
    # This is equivalent to: the graph has diameter <= 4
    
    # Key insight: If we add edges from hubs to far vertices, we can reduce diameter
    # A hub at position h can reach vertices within distance 2 on the path
    # Adding edge (h, o) where o is far makes o reachable in 2 steps from h
    
    # Let's try systematic hub configurations
    def try_single_hub(hub):
        edges = []
        # Hub connects to vertices at distance > 2
        for v in range(n):
            if v != hub and abs(v - hub) > 2:
                # Check if edge is valid (even-odd)
                if (hub % 2) != (v % 2):
                    key = (min(hub, v), max(hub, v))
                    if key not in base_edges:
                        edges.append([hub, v])
        return edges
    
    def try_two_hubs(h1, h2):
        edges = []
        # Connect h1 to far odd/even vertices
        # Connect h2 to far odd/even vertices
        # Ensure h1-h2 edge if needed
        key12 = (min(h1, h2), max(h1, h2))
        if key12 not in base_edges:
            if (h1 % 2) != (h2 % 2):
                edges.append([h1, h2])
        
        for v in range(n):
            d1 = abs(v - h1)
            d2 = abs(v - h2)
            if d1 > 2 and d2 > 2:
                # v needs direct connection to h1 or h2
                if (v % 2) != (h1 % 2):
                    key = (min(h1, v), max(h1, v))
                    if key not in base_edges:
                        edges.append([h1, v])
                elif (v % 2) != (h2 % 2):
                    key = (min(h2, v), max(h2, v))
                    if key not in base_edges:
                        edges.append([h2, v])
        return edges
    
    # Try single hubs
    for hub in range(n):
        config = try_single_hub(hub)
        if len(config) < best_val and validate(config):
            best = [list(e) for e in config]
            best_val = len(best)
            if best_val <= 7:
                break
    
    if time.time() > deadline:
        result = [[int(e[0]), int(e[1])] for e in best]
        return result
    
    # Try two hubs
    for h1 in even_verts:
        for h2 in odd_verts:
            if abs(h1 - h2) <= 1:
                continue  # already adjacent
            config = try_two_hubs(h1, h2)
            if len(config) < best_val and validate(config):
                best = [list(e) for e in config]
                best_val = len(best)
                if best_val <= 6:
                    break
        if best_val <= 6:
            break
    
    if time.time() > deadline:
        result = [[int(e[0]), int(e[1])] for e in best]
        return result
    
    # Strategy 2: Greedy set cover
    # Start empty, add edges that fix most diameter violations
    
    def get_violations(adj):
        violations = []
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for i in range(n):
                if dist[i] > 4:
                    violations.append((start, i))
        return violations
    
    greedy = []
    while time.time() < deadline:
        adj = build_adj(greedy)
        violations = get_violations(adj)
        if not violations:
            break
        
        best_edge = None
        best_fix = 0
        for edge in all_edges:
            if edge in greedy:
                continue
            u, v = edge
            trial = greedy + [edge]
            trial_adj = build_adj(trial)
            if not is_triangle_free(trial_adj):
                continue
            trial_viol = get_violations(trial_adj)
            fixed = len(violations) - len(trial_viol)
            if fixed > best_fix:
                best_fix = fixed
                best_edge = edge
        
        if best_edge is None or best_fix == 0:
            break
        greedy.append(best_edge)
    
    if greedy and len(greedy) < best_val and validate(greedy):
        best = [list(e) for e in greedy]
        best_val = len(best)
    
    # Strategy 3: Local search - edge removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:i] + best[i+1:]
            if validate(trial):
                best = trial
                best_val = len(best)
                changed = True
                break
    
    # Strategy 4: 1-swap local search
    if time.time() < deadline:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            remaining = best[:i] + best[i+1:]
            for alt in all_edges:
                if time.time() > deadline:
                    break
                if alt in remaining:
                    continue
                trial = remaining + [alt]
                if len(trial) < best_val and validate(trial):
                    best = trial
                    best_val = len(best)
                    break
            if len(best) < best_val:
                break
    
    # Final validation
    if not validate(best):
        best = [list(e) for e in fallback]
    
    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])
    
    return result
# EVOLVE-BLOCK-END
