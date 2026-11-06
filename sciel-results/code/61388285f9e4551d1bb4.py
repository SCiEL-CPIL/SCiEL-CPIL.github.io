import time
import sys
from collections import defaultdict, deque
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        original_edges = set(tuple(sorted(e)) for e in edges)
        
        # 1. Calculate degrees and identify odd vertices
        degree = defaultdict(int)
        for u, v in original_edges:
            degree[u] += 1
            degree[v] += 1
            
        odd_vertices = [v for v, d in degree.items() if d % 2 == 1]
        
        # 2. Augment graph to make all degrees even (Chinese Postman)
        # We need to add a multiset of edges such that all degrees become even.
        # This is equivalent to finding a minimum weight T-join where T is the set of odd vertices.
        # For small graphs, we can approximate this by pairing odd vertices with shortest paths.
        # Since we want to minimize the number of paths, minimizing the number of added edges is key.
        # Actually, the number of paths in the final decomposition will be |odd_vertices| / 2 
        # (if we don't merge them). 
        # Wait, Gallai's conjecture is about simple paths.
        # If we make the graph Eulerian, we get 1 circuit. 
        # Breaking a circuit at k odd vertices gives k paths.
        # So the number of paths is determined by the number of odd vertices / 2?
        # No. An Eulerian circuit in a connected graph can be broken into paths by cutting edges.
        # If we start with an Eulerian circuit, and we want simple paths, we can just take the whole circuit 
        # and if it repeats vertices, it's not simple.
        
        # Let's rethink.
        # The current approach extracts simple paths.
        # The "Eulerian" approach usually deals with trails (edges distinct, vertices can repeat).
        # The task requires SIMPLE paths (vertices distinct).
        
        # However, if we find an Eulerian circuit in the augmented graph, it is a trail.
        # We can decompose the trail into simple paths by cutting at repeated vertices.
        # But this might create many short paths.
        
        # Alternative Structural Change:
        # Use a **Minimum Path Cover** approach based on matching?
        # No, Path Cover in directed graphs is matching. In undirected, it's harder.
        
        # Let's try a different structural change: **Iterative Deepening A* for Path Extraction**?
        # No, too slow.
        
        # Let's try **Simulated Annealing on the Set of Paths**.
        # State: A set of paths covering all edges.
        # Move: Remove a path, redistribute its edges into other paths, or merge two paths.
        # This is complex.
        
        # Let's go back to the "Eulerian" idea but adapted for Simple Paths.
        # If we find a spanning tree, we have n-1 edges. The remaining edges are chords.
        # The current code does something similar to "extract longest path".
        
        # New Strategy: **Degree-Driven Pairing and Path Stitching**
        # 1. Identify odd-degree vertices.
        # 2. Pair them up to minimize the total length of connecting paths (T-join approximation).
        # 3. Add these paths to the graph (conceptually).
        # 4. Now all vertices have even degree. Find an Eulerian circuit.
        # 5. The Eulerian circuit is a sequence of edges.
        # 6. We need to convert this circuit into a set of edge-disjoint SIMPLE paths.
        #    A circuit is a closed trail. If we break the circuit at every vertex that appears more than once,
        #    we get a set of simple paths?
        #    No, if a vertex appears 3 times, breaking it creates 3 segments.
        #    Actually, any closed trail can be decomposed into simple cycles.
        #    And any cycle can be a path (if we break it).
        #    But we need to cover ALL edges.
        
        # Let's try a different angle: **Randomized Greedy with Lookahead (Beam Search)**.
        # The current greedy is "pick max degree neighbor".
        # Beam search: Keep top K partial paths.
        
        # Let's try **Local Search on the Path Count**.
        # Start with a valid decomposition (e.g., from current greedy).
        # Try to reduce the number of paths by:
        # 1. Taking two paths P1 and P2.
        # 2. Finding a path Q in the remaining graph (edges not in P1 or P2) that connects an endpoint of P1 to an endpoint of P2?
        #    No, the edges must be edge-disjoint.
        #    If we remove P1 and P2, the remaining graph has edges.
        #    If we can find a path R in the remaining graph that connects end(P1) to start(P2),
        #    we can merge P1 + R + P2 into one path?
        #    Wait, P1 and P2 are already using edges. R uses other edges.
        #    The new path would be P1, then R, then P2.
        #    This is valid if all vertices in P1, R, P2 are distinct (simple path).
        #    And edges are disjoint (they are, by construction).
        
        # This "Path Merging" local search is a strong structural change.
        
        # Implementation Plan:
        # 1. Generate an initial decomposition using the existing greedy (fast).
        # 2. While time remains:
        #    a. Pick two random paths (or paths with high potential to merge).
        #    b. Remove their edges from the graph to get G_rem.
        #    c. Try to find a simple path in G_rem connecting an endpoint of Path A to an endpoint of Path B.
        #       (Try all 4 combinations of endpoints).
        #    d. If found, check if the merged path (A + Connector + B) is simple (no repeated vertices).
        #    e. If valid, replace A and B with the merged path.
        #    f. If not, revert.
        
        return _decompose_with_merging(original_edges, deadline)

    def _decompose_with_merging(edges, deadline):
        # 1. Initial Decomposition (Greedy)
        def greedy_decompose(edges):
            remaining = set(tuple(sorted(e)) for e in edges)
            paths = []
            while remaining:
                if time.time() > deadline:
                    # Fallback: single edges
                    for e in remaining:
                        paths.append([e[0], e[1]])
                    break
                adj = defaultdict(set)
                for u, v in remaining:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Find a long path
                # Start from a high-degree vertex
                if not adj: break
                start = max(adj.keys(), key=lambda x: len(adj[x]))
                path = [start]
                visited = {start}
                curr = start
                while True:
                    neighbors = [n for n in adj[curr] if n not in visited]
                    if not neighbors: break
                    # Warnsdorff
                    next_v = min(neighbors, key=lambda n: len([x for x in adj[n] if x not in visited]))
                    path.append(next_v)
                    visited.add(next_v)
                    curr = next_v
                
                # Extract edges
                path_edges = set()
                for i in range(len(path)-1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e in remaining:
                        path_edges.add(e)
                
                if not path_edges:
                    # Safety: take one edge
                    e = next(iter(remaining))
                    paths.append([e[0], e[1]])
                    remaining.remove(e)
                else:
                    paths.append(path)
                    remaining -= path_edges
            return paths

        paths = greedy_decompose(edges)
        
        # 2. Local Search: Path Merging
        # To merge Path A and Path B, we need a path C in G \ (A ∪ B) connecting end(A) to start(B) (or similar).
        # The merged path is A + C + B.
        # Constraint: A, C, B must be vertex-disjoint except at the connection points.
        
        # Optimization: Only try to merge paths that are "close" or short?
        # For now, random sampling.
        
        rng = random.Random(42)
        max_iter = 20000
        iter_count = 0
        
        # Precompute adjacency for the whole graph? No, it changes.
        # But we can precompute the set of all edges.
        all_edges = set(tuple(sorted(e)) for e in edges)
        
        while time.time() < deadline - 5 and iter_count < max_iter:
            iter_count += 1
            
            if len(paths) <= 1: break
            
            # Pick two paths to try merging
            # Bias towards shorter paths? Or random?
            i, j = rng.sample(range(len(paths)), 2)
            P1 = paths[i]
            P2 = paths[j]
            
            # Get edges of P1 and P2
            def get_path_edges(p):
                es = set()
                for k in range(len(p)-1):
                    es.add(tuple(sorted((p[k], p[k+1]))))
                return es
            
            E1 = get_path_edges(P1)
            E2 = get_path_edges(P2)
            
            # Remaining edges
            E_rem = all_edges - E1 - E2
            
            if not E_rem: continue
            
            # Try 4 connections:
            # 1. end(P1) to start(P2) -> New Path: P1 + C + P2
            # 2. end(P1) to end(P2)   -> New Path: P1 + C + reverse(P2)
            # 3. start(P1) to start(P2) -> New Path: reverse(P1) + C + P2
            # 4. start(P1) to end(P2)   -> New Path: reverse(P1) + C + reverse(P2)
            
            candidates = [
                (P1[-1], P2[0], P1, P2),
                (P1[-1], P2[-1], P1, list(reversed(P2))),
                (P1[0], P2[0], list(reversed(P1)), P2),
                (P1[0], P2[-1], list(reversed(P1)), list(reversed(P2)))
            ]
            
            merged = False
            for start_v, end_v, A, B in candidates:
                # Check vertex disjointness first (cheap)
                set_A = set(A)
                set_B = set(B)
                # Connection vertices must be in both? No.
                # A ends at start_v. B starts at end_v.
                # The connector C must start at start_v and end at end_v.
                # So start_v is in A. end_v is in B.
                # A and B must be disjoint.
                if set_A & set_B:
                    continue
                
                # Find path C in E_rem from start_v to end_v
                # Use BFS
                parent = {start_v: None}
                q = deque([start_v])
                found = False
                adj_rem = defaultdict(set)
                for u, v in E_rem:
                    adj_rem[u].add(v)
                    adj_rem[v].add(u)
                
                while q:
                    curr = q.popleft()
                    if curr == end_v:
                        found = True
                        break
                    for nb in adj_rem[curr]:
                        if nb not in parent:
                            parent[nb] = curr
                            q.append(nb)
                
                if not found:
                    continue
                
                # Reconstruct C
                C = []
                curr = end_v
                while curr is not None:
                    C.append(curr)
                    curr = parent[curr]
                C.reverse()
                
                # Check if C is simple (BFS guarantees simple)
                # Check if C is vertex-disjoint from A and B (except endpoints)
                set_C = set(C)
                # Endpoints of C are start_v and end_v.
                # start_v is in A. end_v is in B.
                # Other vertices in C must not be in A or B.
                internal_C = set(C[1:-1])
                if internal_C & (set_A | set_B):
                    continue
                
                # Construct new path
                # A ends at start_v. C starts at start_v.
                # So A + C[1:] + B
                new_path = A + C[1:] + B
                
                # Verify new_path is simple
                if len(new_path) != len(set(new_path)):
                    continue
                
                # Verify edges of new_path are valid and disjoint
                # Edges of A are in E1. Edges of B are in E2. Edges of C are in E_rem.
                # Since E1, E2, E_rem are disjoint, edges are disjoint.
                # We just need to ensure C uses edges in E_rem. (BFS on adj_rem ensures this).
                
                # Replace P1 and P2 with new_path
                # Remove i and j
                # Be careful with indices if i > j
                if i > j:
                    i, j = j, i
                    # Swap P1, P2 logic? 
                    # We already constructed new_path based on original P1, P2.
                    # We just need to remove the two old paths and add the new one.
                
                del paths[i]
                del paths[j-1] # Adjust index
                paths.append(new_path)
                merged = True
                break # Try next pair
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time budget per graph
    # Graph 3 is largest, give it more time?
    # Or equal time?
    # Let's give 45s each, with 15s buffer.
    time_per_graph = 45
    
    for idx, spec in enumerate(specs):
        graph_start = time.time()
        edges = instance(*spec)
        
        # Adjust deadline for this graph
        local_deadline = graph_start + time_per_graph
        # But also respect global deadline
        local_deadline = min(local_deadline, deadline - 5)
        
        paths = decompose_graph(edges)
        results.append(paths)
        
        # If we ran out of global time, fill rest
        if time.time() > deadline - 5:
             for remaining_spec in specs[idx+1:]:
                 e = instance(*remaining_spec)
                 results.append([[u, v] for u, v in e])
             break
             
    return results
