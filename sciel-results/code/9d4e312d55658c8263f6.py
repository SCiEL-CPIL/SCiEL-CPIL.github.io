import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    def compute_diameter(adj):
        """Compute diameter of graph using BFS from each node."""
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam
    
    def build_adj(added_edges):
        """Build adjacency list from base + added edges."""
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_valid_solution(added_edges):
        """Check if solution is valid: triangle-free, connected, diameter <= 4."""
        adj = build_adj(added_edges)
        # Check triangle-free: for each edge (u,v), check no common neighbor
        for u in range(n):
            for v in adj[u]:
                if u < v and (u, v) not in base_edges:
                    # This is an added edge
                    if adj[u] & adj[v]:
                        return False
                elif u < v:
                    # Base edge, still need to check
                    if adj[u] & adj[v]:
                        return False
        # Check connected and diameter
        diam = compute_diameter(adj)
        if diam > 4:
            return False
        return True
    
    def is_valid_edge(u, v, added_edges):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        a, b = min(u, v), max(u, v)
        if (a, b) in base_edges:
            return False
        if any((min(x, y), max(x, y)) == (a, b) for x, y in added_edges):
            return False
        # Check triangle: u and v must have no common neighbor
        adj = build_adj(added_edges)
        if adj[u] & adj[v]:
            return False
        return True
    
    deadline = time.time() + 85.0
    
    # Phase 1: Build a valid solution using a hub-based approach
    # Partition: A = even, B = odd
    A = [i for i in range(n) if i % 2 == 0]
    B = [i for i in range(n) if i % 2 == 1]
    
    # Try different hub configurations
    best_solution = None
    best_count = n * n  # large number
    
    # Strategy: Pick hub(s) and connect to many vertices in opposite partition
    # Try single hub first
    for hub_candidates in [A, B]:
        for hub in hub_candidates:
            other_partition = B if hub in A else A
            # Connect hub to all in other partition
            added = []
            for v in other_partition:
                a, b = min(hub, v), max(hub, v)
                if (a, b) not in base_edges:
                    added.append((hub, v))
            # Remove base edges that are now redundant? No, keep them for connectivity
            added_list = [list(e) for e in added]
            if is_valid_solution(added_list):
                if len(added_list) < best_count:
                    best_count = len(added_list)
                    best_solution = added_list[:]
    
    # Try 2-hub configurations
    for h1 in A:
        for h2 in B:
            added = []
            # Connect h1 to some in B, h2 to some in A
            # Distribute: h1 connects to first half of B, h2 to second half of A
            for v in B:
                if (min(h1, v), max(h1, v)) not in base_edges:
                    added.append((h1, v))
            for v in A:
                if (min(h2, v), max(h2, v)) not in base_edges:
                    added.append((h2, v))
            # Try to be selective: only add if it helps
            added_list = [list(e) for e in added]
            if is_valid_solution(added_list):
                if len(added_list) < best_count:
                    best_count = len(added_list)
                    best_solution = added_list[:]
    
    # Phase 2: Greedy removal - try to remove edges from best solution
    if best_solution is not None:
        current = best_solution[:]
        removed = True
        while removed:
            removed = False
            for i in range(len(current)):
                trial = current[:i] + current[i+1:]
                if is_valid_solution(trial):
                    current = trial
                    removed = True
                    break
        if len(current) < best_count:
            best_count = len(current)
            best_solution = current[:]
    
    # Phase 3: Try to add edges that allow further removal
    # For each missing valid edge, add it, then try removal again
    if best_solution is not None:
        for u in A:
            for v in B:
                if (min(u, v), max(u, v)) in base_edges:
                    continue
                if any(min(x, y) == min(u, v) and max(x, y) == max(u, v) for x, y in best_solution):
                    continue
                trial = best_solution + [[u, v]]
                if not is_valid_solution(trial):
                    continue
                # Try removal
                current = trial[:]
                removed = True
                while removed:
                    removed = False
                    for i in range(len(current)):
                        t2 = current[:i] + current[i+1:]
                        if is_valid_solution(t2):
                            current = t2
                            removed = True
                            break
                if len(current) < best_count:
                    best_count = len(current)
                    best_solution = current[:]
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
    
    return [[int(u), int(v)] for u, v in best_solution]
# EVOLVE-BLOCK-END
