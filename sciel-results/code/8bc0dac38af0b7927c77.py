import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    best_cycles = []
    best_count = 0
    
    def find_hamiltonian_cycle(vertices, used_edges, deadline):
        """
        Find a Hamiltonian cycle on vertices avoiding used_edges.
        Uses greedy + backtracking with strong pruning.
        """
        if time.time() > deadline:
            return None
        
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Pre-compute available neighbors for each vertex
        # avail[v] = set of neighbors u such that (v,u) is not in used_edges and u in vertices
        avail = {v: set() for v in vertices}
        for v in vertices:
            for u in adj[v]:
                if u in vertices:
                    e = (min(u, v), max(u, v))
                    if e not in used_edges:
                        avail[v].add(u)
        
        # Check feasibility: all vertices must have degree >= 2 in the available graph
        for v in vertices:
            if len(avail[v]) < 2:
                return None
        
        # Try different start vertices
        # Heuristic: start with vertices that have lower degree in the available graph
        start_candidates = sorted(vertices, key=lambda v: len(avail[v]))
        
        # Limit attempts to avoid excessive time on bad starts
        max_starts = min(len(start_candidates), 5)
        
        for start in start_candidates[:max_starts]:
            if time.time() > deadline:
                return None
            
            path = [start]
            remaining = set(vertices) - {start}
            
            # We need to close the cycle, so 'start' must be connected to the last node.
            # Effectively, we are finding a path from start to some node that connects back to start.
            
            def backtrack(path, remaining, depth=0):
                if time.time() > deadline:
                    return None
                
                if len(path) == k:
                    last = path[-1]
                    # Check if last is connected to start
                    e = (min(last, start), max(last, start))
                    if e not in used_edges:
                        # Verify it's a simple cycle (distinct vertices)
                        # Since we track 'remaining', all vertices in path are distinct.
                        return path[:]
                    return None
                
                last = path[-1]
                
                # Get candidates for next vertex
                candidates = []
                for v in avail[last]:
                    if v in remaining:
                        # Heuristic: prefer vertices with lower remaining degree
                        # Calculate degree of v in the subgraph induced by remaining + {last} (approx)
                        # Actually, just count neighbors in 'remaining'
                        rem_deg = sum(1 for w in avail[v] if w in remaining or w == last)
                        # Also, we must ensure that after picking v, 
                        # no other vertex in 'remaining' gets isolated.
                        
                        # Pruning: 
                        # 1. If we pick v, the edge (last, v) is used.
                        # 2. For any u in remaining \ {v}, if u's only neighbor in 
                        #    (remaining + {last, v}) is v or last, and we don't pick it next, 
                        #    it might get stuck.
                        
                        # Simple strong prune:
                        # Check if any vertex in remaining \ {v} has degree 0 or 1 
                        # in the graph formed by remaining \ {v} plus connections to path ends?
                        # This is expensive. Let's stick to degree ordering first.
                        
                        candidates.append((rem_deg, v))
                
                if not candidates:
                    return None
                
                # Sort by degree (most constrained first)
                candidates.sort()
                
                for _, v in candidates:
                    # Additional prune:
                    # If we go to v, does any other vertex in remaining become isolated?
                    # Specifically, if there is a vertex u in remaining \ {v}
                    # such that all its neighbors in (remaining + {last}) are in {v} 
                    # and v is not the only option? 
                    # Actually, if u has only 1 neighbor in the *available* graph 
                    # restricted to (remaining + {last}), and that neighbor is not v, 
                    # then u is stuck because we are moving away from last.
                    
                    stuck = False
                    for u in remaining:
                        if u == v:
                            continue
                        # Check if u is isolated in the future
                        # u's potential neighbors are:
                        # 1. Neighbors in remaining \ {v}
                        # 2. 'last' (if u is adjacent to last)
                        # Note: u cannot connect to other path vertices yet.
                        
                        neighbors_in_rem = [w for w in avail[u] if w in remaining and w != v]
                        can_connect_last = (last in avail[u])
                        
                        # If u has no neighbors in remaining \ {v} AND cannot connect to last, it's stuck.
                        # Wait, if u can connect to last, it's not stuck *yet*, but it must connect to last 
                        # either now or at the end.
                        # If u has 0 neighbors in remaining \ {v} and cannot connect to last, stuck.
                        # If u has 1 neighbor in remaining \ {v} and cannot connect to last, it's forced to that neighbor.
                        
                        if not neighbors_in_rem and not can_connect_last:
                            stuck = True
                            break
                        
                        # Also, if u is forced to connect to last, and last is already "used up" 
                        # (last can only connect to 2 vertices in total: prev and next), 
                        # we need to be careful.
                        # But 'last' is the current end. It will connect to 'v' (next) and 'prev' (already done).
                        # So 'last' cannot connect to 'u' unless u == v.
                        # So if u != v and u needs to connect to last, it's impossible 
                        # because last is going to v.
                        # UNLESS u is the *last* vertex in the path, which connects back to start.
                        # But we are building the path forward.
                        
                        # Correction:
                        # The path is start -> ... -> last -> v -> ...
                        # 'last' is connected to 'prev' and 'v'.
                        # So 'last' has no capacity to connect to any other vertex in remaining.
                        # Therefore, if u != v and u's only available neighbor in (remaining + {last}) is 'last', 
                        # then u is stuck.
                        
                        if not neighbors_in_rem:
                            # u has no neighbors in remaining \ {v}
                            # u must connect to last to be saved?
                            # But last is connecting to v.
                            # So u is stuck.
                            stuck = True
                            break
                        
                    if stuck:
                        continue

                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
                
                return None
            
            result = backtrack(path, remaining)
            if result is not None:
                return result
        
        return None

    def try_vertex_subset(vertices, deadline):
        """Try to find max edge-disjoint Hamiltonian cycles on this vertex set."""
        nonlocal best_cycles, best_count
        if time.time() > deadline:
            return
        
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return
        
        # Count edges in induced subgraph
        induced_edges = set()
        for u in vertices:
            for v in adj[u]:
                if v in vertices:
                    edge = (min(u, v), max(u, v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily
        for attempt in range(max_possible):
            if time.time() > deadline:
                break
            
            cycle = find_hamiltonian_cycle(vertices, used_edges, deadline)
            if cycle is None:
                break
            
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
            
            # Optional: If we found a good set, try to extend it?
            # Or just return.
            
    all_vertices = set(range(n))
    
    # 1. Try full vertex set
    try_vertex_subset(all_vertices, deadline)
    
    # 2. Try all single-vertex removals (20 vertices)
    for removed in range(n):
        if time.time() > deadline:
            break
        subset = all_vertices - {removed}
        try_vertex_subset(subset, deadline)
        
    # 3. Try all double-vertex removals (19 vertices)
    # This is 21*20/2 = 210 subsets.
    # If time permits.
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            subset = all_vertices - {i, j}
            try_vertex_subset(subset, deadline)
            
    # 4. If still low, try some random 18-vertex subsets
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        subset = set(random.sample(range(n), 18))
        try_vertex_subset(subset, deadline)

    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: try to find any cycle on full vertex set
        cycle = find_hamiltonian_cycle(all_vertices, set(), deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
