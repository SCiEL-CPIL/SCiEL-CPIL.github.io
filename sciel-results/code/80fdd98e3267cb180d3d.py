import time
import sys
from collections import defaultdict
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def verify_decomposition(paths, edges, n):
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        
        return True

    def solve_graph(n, p, seed, time_budget=40):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        adj = defaultdict(list)
        edge_map = {}
        for idx, (u, v) in enumerate(edges):
            adj[u].append(v)
            adj[v].append(u)
            edge_map[(u, v)] = idx
            edge_map[(v, u)] = idx
            
        # Map edge index to endpoints
        edge_ends = [(u, v) for u, v in edges]
        
        # Initialize pairings
        # For each vertex, we have a list of incident edge indices.
        # We maintain a list of "connections" which are pairs of edge indices or single edge indices.
        # Actually, simpler: For each vertex, maintain a list of incident edges.
        # A state is defined by how these edges are paired.
        
        # Let's use a representation where for each vertex v, we have a list of incident edge IDs.
        # We partition this list into pairs and singles.
        
        incident_edges = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            incident_edges[u].append(idx)
            incident_edges[v].append(idx)

        # Initial random pairing
        # For each vertex, shuffle incident edges and pair them up.
        # If odd number, one is left single.
        
        def random_pairing():
            pairings = {}
            for v in range(n):
                e_ids = incident_edges[v][:]
                random.shuffle(e_ids)
                pairs = []
                i = 0
                while i < len(e_ids):
                    if i + 1 < len(e_ids) and random.random() < 0.9: # High prob of pairing
                        pairs.append((e_ids[i], e_ids[i+1]))
                        i += 2
                    else:
                        pairs.append((e_ids[i],))
                        i += 1
                pairings[v] = pairs
            return pairings

        def get_trails(pairings):
            # Build a graph of edge-to-edge connections
            # Node: edge ID
            # Edge: (e1, e2) if they are paired at some vertex
            
            # Actually, we can just walk the trails.
            # Start from any edge that is "single" at one end?
            # Or just start from any edge and follow.
            
            visited_edges = set()
            trails = []
            
            # We need to traverse the "line graph" of the pairing structure.
            # Let's build an adjacency for the edges based on the pairings.
            edge_adj = defaultdict(list)
            
            for v in range(n):
                for pair in pairings[v]:
                    if len(pair) == 2:
                        e1, e2 = pair
                        edge_adj[e1].append(e2)
                        edge_adj[e2].append(e1)
                    # Singles don't create internal connections, they are endpoints
            
            # Now find paths in this edge-adjacency graph.
            # Each node is an edge.
            # The graph is a set of disjoint paths and cycles?
            # Since it's a pairing, the degree of each edge node is at most 2 (connected to at most 2 other edges, one at each endpoint).
            # So the components are paths or cycles.
            
            for e in range(len(edges)):
                if e in visited_edges:
                    continue
                
                trail = []
                current = e
                prev = -1
                
                while True:
                    if current in visited_edges:
                        # Cycle or closed loop?
                        # If we return to start, it's a cycle.
                        # If we hit a visited node from another trail, something is wrong (should be disjoint).
                        break
                    
                    visited_edges.add(current)
                    trail.append(current)
                    
                    # Find next edge
                    next_e = -1
                    for nb in edge_adj[current]:
                        if nb != prev:
                            next_e = nb
                            break
                    
                    if next_e == -1:
                        # End of trail
                        break
                    
                    prev, current = current, next_e
            
            if trail:
                # Convert edge trail to vertex trail
                v_trail = []
                for i, e_id in enumerate(trail):
                    u, v = edge_ends[e_id]
                    if i == 0:
                        v_trail.extend([u, v])
                    else:
                        # Check continuity
                        last_v = v_trail[-1]
                        if u == last_v:
                            v_trail.append(v)
                        elif v == last_v:
                            v_trail.append(u)
                        else:
                            # Discontinuity? Should not happen if pairing is consistent.
                            # But wait, the pairing ensures continuity at the vertex where they are paired.
                            # So this should be fine.
                            v_trail.append(u)
                trails.append(v_trail)
                
            return trails

        def split_trails_to_simple_paths(trails):
            paths = []
            for trail in trails:
                if not trail:
                    continue
                # Check for repeated vertices
                seen = set()
                current_path = []
                for v in trail:
                    if v in seen:
                        # Split
                        if len(current_path) > 1:
                            paths.append(current_path)
                        current_path = [v]
                    else:
                        current_path.append(v)
                    seen.add(v)
                if len(current_path) > 1:
                    paths.append(current_path)
            return paths

        def count_paths(pairings):
            trails = get_trails(pairings)
            paths = split_trails_to_simple_paths(trails)
            return len(paths), paths

        # Local Search
        best_k, best_paths = count_paths(random_pairing())
        
        # Simulated Annealing
        T = 1.0
        cool_rate = 0.99
        min_T = 0.01
        
        current_pairings = random_pairing()
        current_k, _ = count_paths(current_pairings)
        
        # If initial is already good, we might not need much search
        # Lower bound is number of odd vertices / 2
        odd_count = sum(1 for v in range(n) if len(incident_edges[v]) % 2 != 0)
        lower_bound = odd_count // 2
        
        if current_k <= lower_bound:
            # Found optimal
            return best_paths

        end_time = start_time + time_budget
        while time.time() < end_time and T > min_T:
            # Generate neighbor
            new_pairings = [list(p) for p in current_pairings] # Deep copy
            
            # Pick a random vertex and modify its pairing
            v = random.randint(0, n - 1)
            if not new_pairings[v]:
                T *= cool_rate
                continue
                
            # Pick a random pair or single to modify
            # Simple move: Pick two pairs (a,b) and (c,d) and swap to (a,c), (b,d)
            # Or pick a pair (a,b) and single (c) and make (a,c), (b)
            
            # Let's just re-randomize the pairing for one vertex
            # This is a big move, but effective for escaping local minima
            e_ids = incident_edges[v][:]
            random.shuffle(e_ids)
            new_pairs = []
            i = 0
            while i < len(e_ids):
                if i + 1 < len(e_ids):
                    # 50% chance to pair, 50% to single?
                    # Or just shuffle and pair sequentially
                    new_pairs.append((e_ids[i], e_ids[i+1]))
                    i += 2
                else:
                    new_pairs.append((e_ids[i],))
                    i += 1
            
            new_pairings[v] = new_pairs
            
            new_k, _ = count_paths(new_pairings)
            
            delta = new_k - current_k
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_pairings = new_pairings
                current_k = new_k
                if current_k < best_k:
                    best_k = current_k
                    _, best_paths = count_paths(current_pairings)
                    if best_k <= lower_bound:
                        break
            
            T *= cool_rate
        
        # Verify best_paths
        if not verify_decomposition(best_paths, edges, n):
            # Fallback
            # Re-derive from best_pairings
            trails = get_trails(best_paths and current_pairings or random_pairing()) 
            # Wait, best_paths is derived from a pairing. We need to keep the pairing.
            # Let's just return the paths we have. If invalid, use a simple fallback.
            # Simple fallback: each edge is a path.
            return [list(e) for e in edges]
            
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        paths = solve_graph(n, p, seed, time_budget=min(40, deadline - time.time() - 10))
        results.append(paths)

    return results

# EVOLVE-BLOCK-END
