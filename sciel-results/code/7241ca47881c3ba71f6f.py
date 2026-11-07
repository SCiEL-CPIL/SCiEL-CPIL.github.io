import time
import sys
from collections import defaultdict
import random

# EVOLVE-BLOCK-START

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and edge set for fast lookup
    adj = defaultdict(set)
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Target: Maximize number of cycles.
    # Theoretical max for n=21, m=75 is 3 (since 3*21 = 63 <= 75, 4*21 = 84 > 75).
    # We aim for 3.
    
    # --- Attempt 1: Fast Path Construction (Circulant-style) ---
    # If the graph is a circulant or contains circulant subgraphs, we can find
    # edge-disjoint Hamiltonian cycles using steps that are coprime to n and distinct.
    # Steps 1, 2, 5 are coprime to 21 and mutually distinct modulo 21.
    # Cycle for step k: 0 -> k -> 2k -> ... -> (n-1)k -> 0
    
    def try_fast_path(steps):
        cycles_found = []
        used_edges = set()
        for k in steps:
            if k % n == 0: continue
            cycle = [(i * k) % n for i in range(n)]
            # Verify this cycle is valid and edge-disjoint
            valid = True
            for i in range(n):
                u = cycle[i]
                v = cycle[(i+1) % n]
                eu, ev = min(u, v), max(u, v)
                e = (eu, ev)
                if e not in edge_set:
                    valid = False
                    break
                if e in used_edges:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    u = cycle[i]
                    v = cycle[(i+1) % n]
                    used_edges.add((min(u, v), max(u, v)))
                cycles_found.append(cycle)
            else:
                break # One failure means this specific set of steps doesn't work as a block
        return cycles_found

    # Try common small steps
    fast_result = try_fast_path([1, 2, 5])
    if len(fast_result) >= 3:
        return [tuple(c) for c in fast_result[:3]]
    
    # If fast path didn't yield 3, we need to search.
    # We will search for 3 edge-disjoint Hamiltonian cycles.
    
    # State for search
    best_cycles = []
    
    # Heuristic: Start with the full vertex set.
    vertices = list(range(n))
    
    # We use a backtracking search to find the first cycle, then remove edges, repeat.
    # To ensure speed, we use a greedy heuristic for the first few cycles if possible,
    # but strict backtracking is safer for "disjointness" constraints.
    
    def find_hamiltonian_cycle(used_edges, timeout=5.0):
        """
        Finds a Hamiltonian cycle in the residual graph (V, E \ used_edges).
        Returns the list of vertices in order, or None.
        """
        start = min(vertices)
        path = [start]
        visited = {start}
        t_start = time.time()
        
        # Precompute available neighbors for start to help ordering
        def get_unvisited_neighbors(cur):
            nbs = []
            for nb in adj[cur]:
                if nb not in visited and nb in set(vertices):
                    e = (min(cur, nb), max(cur, nb))
                    if e not in used_edges:
                        nbs.append(nb)
            # Warnsdorff's rule: sort by number of onward options (fewer first)
            nbs.sort(key=lambda x: len([y for y in adj[x] if y not in visited and y in set(vertices) and (min(x,y), max(x,y)) not in used_edges]))
            return nbs

        def dfs(cur):
            if time.time() - t_start > timeout:
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                if (min(cur, start), max(cur, start)) in used_edges:
                    return None
                # Also ensure start is adjacent (implied by edge check)
                return list(path)
            
            nbs = get_unvisited_neighbors(cur)
            # Connectivity check (optional but helpful): 
            # The unvisited nodes must be connected to the current node in the residual graph.
            # For simplicity and speed, we rely on Warnsdorff first.
            
            for nb in nbs:
                e = (min(cur, nb), max(cur, nb))
                # Edge is guaranteed not in used_edges by get_unvisited_neighbors
                path.append(nb)
                visited.add(nb)
                
                # Mark edge as temporarily used? No, we just pass used_edges.
                # But wait, if we use edge (cur, nb), we must not use it again in this path.
                # The path is simple, so vertices are unique. Edges are unique in a simple path.
                # The only edge that might be reused is the closing edge, which we check at the end.
                # So we don't need to update used_edges during the path construction for internal edges.
                
                res = dfs(nb)
                if res is not None:
                    return res
                
                visited.remove(nb)
                path.pop()
            return None
        
        return dfs(start)

    # Greedy extraction with backtracking fallback for robustness
    # We try to extract 3 cycles.
    
    # To improve success rate, we can try different starting vertices for the first cycle.
    # However, Hamiltonian cycles are rotationally symmetric, so the start vertex doesn't change the set of edges used.
    
    # Let's try to find 3 cycles.
    # We'll store the edges used by the cycles we've found so far.
    
    # Since the fast path might have failed partially (e.g., found 1 or 2 valid cycles),
    # we should check if we can extend the fast result.
    # But for simplicity and distinctness, we will restart the search if fast path < 3.
    
    # Optimization: If the graph is dense enough, a simple greedy with backtracking on the last step works well.
    
    # Let's implement a robust search for 3 cycles.
    # We will try to find Cycle 1, then Cycle 2 (using remaining edges), then Cycle 3.
    
    def extract_k_cycles(k_target):
        current_used = set()
        cycles = []
        for i in range(k_target):
            # Try to find a cycle in the residual graph
            # We can randomize the starting vertex or the order of neighbors to break symmetries if stuck
            found = None
            # Try a few random shuffles of neighbor order if first attempt fails?
            # For now, try deterministic DFS.
            found = find_hamiltonian_cycle(current_used, timeout=3.0)
            if not found:
                return cycles
            # Add edges to current_used
            for i in range(n):
                u = found[i]
                v = found[(i+1) % n]
                current_used.add((min(u, v), max(u, v)))
            cycles.append(found)
        return cycles

    found_cycles = extract_k_cycles(3)
    
    if len(found_cycles) == 3:
        return [tuple(c) for c in found_cycles]
    
    # If we only found < 3, try with randomization to escape local minima?
    # Given the graph size (21), if 3 exist, they are likely easy to find.
    # If the fast path [1, 2, 5] failed, maybe other steps work?
    # Try steps 1, 2, 4? (4 is not coprime to 21? 21=3*7, 4 is coprime).
    # Try steps 1, 2, 4
    fast_result_2 = try_fast_path([1, 2, 4])
    if len(fast_result_2) >= 3:
        return [tuple(c) for c in fast_result_2[:3]]
        
    # Try steps 1, 2, 8?
    fast_result_3 = try_fast_path([1, 2, 8])
    if len(fast_result_3) >= 3:
        return [tuple(c) for c in fast_result_3[:3]]

    # If all fails, return what we have (max 3).
    # The problem states "Maximize number of cycles".
    # If we can only find 2, we return 2.
    # But we believe 3 is possible.
    
    # One more attempt: Randomized DFS for the 3rd cycle if the first 2 were found.
    if len(found_cycles) == 2:
        current_used = set()
        for c in found_cycles:
            for i in range(n):
                u = c[i]
                v = c[(i+1) % n]
                current_used.add((min(u, v), max(u, v)))
        
        # Try a randomized search for the 3rd
        random.seed(42)
        for attempt in range(100):
            # Shuffle the neighbor order logic by using a random start or random tie-breaking?
            # We can simulate randomization by picking a random starting vertex for the cycle.
            start = random.choice(vertices)
            # The DFS function uses min(vertices) as start. 
            # We need a version that takes a start parameter.
            
            def find_ham_start(start_v, used, timeout=1.0):
                path = [start_v]
                visited = {start_v}
                t_start = time.time()
                
                def dfs(cur):
                    if time.time() - t_start > timeout:
                        return None
                    if len(path) == n:
                        if (min(cur, start_v), max(cur, start_v)) not in used:
                            return list(path)
                        return None
                    
                    nbs = list(adj[cur])
                    random.shuffle(nbs) # Randomize order
                    
                    for nb in nbs:
                        if nb in visited or nb not in set(vertices):
                            continue
                        e = (min(cur, nb), max(cur, nb))
                        if e in used:
                            continue
                        
                        path.append(nb)
                        visited.add(nb)
                        
                        res = dfs(nb)
                        if res is not None:
                            return res
                        
                        visited.remove(nb)
                        path.pop()
                    return None
                return dfs(start_v)
            
            res = find_ham_start(start, current_used, timeout=1.0)
            if res:
                found_cycles.append(res)
                return [tuple(c) for c in found_cycles]

    return [tuple(c) for c in found_cycles]

# EVOLVE-BLOCK-END
