"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = defaultdict(list)
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


def has_bicolored_cycle(colors, k):
    """Check if any pair of colors induces a cycle. Uses union-find per color pair."""
    for a in range(k):
        for b in range(a + 1, k):
            # Build subgraph induced by colors a and b
            parent = {}
            rank = {}
            
            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Process edges between vertices colored a or b
            for u, v in EDGES:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not union(u, v):
                        return True
    return False


def count_distinct_colors(colors):
    return len(set(colors))


def is_valid(colors, k):
    """Check adjacency constraint and bicolored cycle constraint."""
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    if has_bicolored_cycle(colors, k):
        return False
    return True


def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    best_colors = list(range(N))
    best_score = count_distinct_colors(best_colors)  # 50
    
    # Try greedy coloring with increasing k
    for k in range(2, 15):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this k
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initial coloring with k colors
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Local search: try to fix adjacency violations and bicolored cycles
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Check for adjacency violations
                adj_violations = []
                for u, v in EDGES:
                    if colors[u] == colors[v]:
                        adj_violations.append((u, v))
                
                # Try to fix adjacency violations
                if adj_violations:
                    for u, v in adj_violations:
                        # Try recoloring u to a different color
                        for c in range(k):
                            if c != colors[u]:
                                # Check if c is allowed for u (no neighbor has color c)
                                ok = True
                                for nb in ADJ[u]:
                                    if colors[nb] == c:
                                        ok = False
                                        break
                                if ok:
                                    colors[u] = c
                                    improved = True
                                    break
                        if improved:
                            break
                
                # Check for bicolored cycles
                if not adj_violations:
                    cycle_found = False
                    for a in range(k):
                        if cycle_found:
                            break
                        for b in range(a + 1, k):
                            if cycle_found:
                                break
                            # Find a cycle in the subgraph induced by colors a and b
                            # BFS to find a cycle
                            visited = set()
                            parent = {}
                            cycle = None
                            
                            for start in range(N):
                                if colors[start] not in (a, b) or start in visited:
                                    continue
                                queue = deque([start])
                                visited.add(start)
                                parent[start] = None
                                while queue and cycle is None:
                                    u = queue.popleft()
                                    for v in ADJ[u]:
                                        if colors[v] not in (a, b):
                                            continue
                                        if v not in visited:
                                            visited.add(v)
                                            parent[v] = u
                                            queue.append(v)
                                        elif v != parent.get(u, -1):
                                            # Found a back edge, reconstruct cycle
                                            cycle = [u, v]
                                            cur = u
                                            while cur != v and parent.get(cur) is not None:
                                                cur = parent[cur]
                                                cycle.append(cur)
                                            break
                            
                            if cycle:
                                # Recolor a vertex in the cycle to a third color
                                for vertex in cycle:
                                    for c in range(k):
                                        if c != colors[vertex] and c not in (a, b):
                                            # Check if c is allowed
                                            ok = True
                                            for nb in ADJ[vertex]:
                                                if colors[nb] == c:
                                                    ok = False
                                                    break
                                            if ok:
                                                colors[vertex] = c
                                                cycle_found = True
                                                improved = True
                                                break
                                    if cycle_found:
                                        break
                            
                            if cycle_found:
                                break
                
                # If no violations, check total validity
                if not adj_violations and not cycle_found:
                    if is_valid(colors, k):
                        score = count_distinct_colors(colors)
                        if score < best_score:
                            best_score = score
                            best_colors = colors[:]
                        break  # Found valid coloring for this k, move to next k
        
        if best_score <= k:
            break  # Can't do better than k with k colors
    
    # Try to reduce colors further with local search on best solution
    if time.time() < deadline - 5:
        colors = best_colors[:]
        k = best_score
        for iteration in range(2000):
            if time.time() > deadline - 3:
                break
            
            # Try to reduce k by 1
            new_k = k - 1
            if new_k < 2:
                break
            
            # Try to map colors to new_k
            # Simple approach: try each vertex, see if we can change its color
            # to one of the existing colors in [0, new_k)
            changed = False
            for u in range(N):
                for c in range(new_k):
                    if c == colors[u]:
                        continue
                    # Check adjacency
                    ok = True
                    for nb in ADJ[u]:
                        if colors[nb] == c:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Try the change
                    old_color = colors[u]
                    colors[u] = c
                    if is_valid(colors, new_k):
                        changed = True
                        break
                    else:
                        colors[u] = old_color
                if changed:
                    break
            
            if changed:
                k = new_k
                best_score = k
                best_colors = colors[:]
            else:
                break
    
    # Ensure we return Python ints in a list
    result = [int(c) for c in best_colors]
    return result
# EVOLVE-BLOCK-END
