"""Rewritten program: packing edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import sys
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s kill

    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def ekey(u, v):
        return (min(u, v), max(u, v))

    def find_ham_cycle(used_edges, start_vertex=0):
        """DFS for a Hamiltonian cycle avoiding used_edges."""
        path = [start_vertex]
        visited = {start_vertex}
        
        def dfs():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                e = ekey(last, start_vertex)
                if is_edge(last, start_vertex) and e not in used_edges:
                    return path[:]
                return None

            last = path[-1]
            # Candidates: unvisited neighbors with available edge
            candidates = []
            for nb in adj[last]:
                if nb in visited:
                    continue
                e = ekey(last, nb)
                if e in used_edges:
                    continue
                candidates.append(nb)
            
            if not candidates:
                return None

            # Heuristic: Sort by number of available moves (Warnsdorff-like)
            # Lower degree in remaining graph is preferred
            def rem_deg(v):
                c = 0
                for w in adj[v]:
                    if w not in visited or w == start_vertex:
                        if ekey(v, w) not in used_edges:
                            c += 1
                return c
            
            candidates.sort(key=lambda v: rem_deg(v))
            
            for nb in candidates:
                path.append(nb)
                visited.add(nb)
                res = dfs()
                if res is not None:
                    return res
                path.pop()
                visited.remove(nb)
            
            return None

        return dfs()

    used_edges = set()
    cycles = []
    
    # Target: 3 cycles
    for i in range(3):
        if time.time() > deadline:
            break
        
        found = False
        # Try different start vertices if the first one fails
        starts = list(range(n))
        random.Random(42 + i).shuffle(starts)
        
        for s in starts:
            c = find_ham_cycle(used_edges, start_vertex=s)
            if c is not None:
                # Add edges to used set
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    used_edges.add(ekey(u, v))
                cycles.append(c)
                found = True
                break
        
        if not found:
            break

    # Validate
    def valid_solution(cycles):
        if not cycles:
            return False
        seen = set()
        for c in cycles:
            if len(c) != n:
                return False
            if len(set(c)) != n:
                return False
            for j in range(n):
                u = c[j]
                v = c[(j + 1) % n]
                e = ekey(u, v)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
        return True

    if not valid_solution(cycles):
        # Fallback: try to find at least 1
        c = find_ham_cycle(set(), start_vertex=0)
        if c is not None:
            cycles = [c]
        else:
            cycles = []

    # Convert to list of lists of ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
