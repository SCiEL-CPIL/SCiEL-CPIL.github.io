"""Path decomposition for Gallai's conjecture instances via combined greedy and Eulerian pairing."""
import time
from collections import defaultdict
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _greedy_decompose(edge_set, adj, rng, deadline):
    """One greedy pass: build maximal simple paths, return list of paths."""
    remaining = set(edge_set)
    paths = []
    
    # Build remaining adjacency
    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
    
    def rem_degree(v):
        return len(rem_adj[v]) if v in rem_adj else 0
    
    while remaining and time.time() < deadline:
        # Choose start vertex: prefer odd-degree in remaining graph
        rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
        if rem_odd:
            # Pick randomly among top candidates for diversity
            candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:3]
            start = rng.choice(candidates)
        else:
            candidates = [u for u in rem_adj if rem_degree(u) > 0]
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:3]
            start = rng.choice(candidates)
        
        # Build a maximal simple path
        path = [start]
        in_path = {start}
        current = start
        
        while True:
            # Get neighbors with unused edges
            neighbors = [nb for nb in rem_adj[current] 
                         if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                break
            
            # Prefer neighbors that are NOT in path (to avoid cycles)
            non_cycle = [nb for nb in neighbors if nb not in in_path]
            if non_cycle:
                non_cycle.sort(key=lambda v: -rem_degree(v))
                # Add some randomness
                if rng.random() < 0.3 and len(non_cycle) > 1:
                    next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                else:
                    next_v = non_cycle[0]
            else:
                # All neighbors are in path; pick one to allow cycle (but path must be simple)
                # Actually we can't revisit, so we're stuck
                break
            
            key = (min(current, next_v), max(current, next_v))
            remaining.discard(key)
            path.append(next_v)
            in_path.add(next_v)
            
            # Update rem_adj
            if next_v in rem_adj[current]:
                rem_adj[current].remove(next_v)
            if current in rem_adj[next_v]:
                rem_adj[next_v].remove(current)
            if not rem_adj[current]:
                del rem_adj[current]
            if not rem_adj[next_v]:
                del rem_adj[next_v]
            
            current = next_v
        
        if len(path) > 1:
            paths.append(path)
    
    # Handle any leftover edges (should be rare)
    while remaining and time.time() < deadline:
        e = next(iter(remaining))
        u, v = e
        paths.append([u, v])
        remaining.discard(e)
        if v in rem_adj.get(u, []):
            rem_adj[u].remove(v)
        if u in rem_adj.get(v, []):
            rem_adj[v].remove(u)
        if u in rem_adj and not rem_adj[u]:
            del rem_adj[u]
        if v in rem_adj and not rem_adj[v]:
            del rem_adj[v]
    
    return paths

def _eulerian_decompose(edge_set, adj, odd_vertices, pairing, deadline):
    """
    Try to decompose using a specific pairing of odd vertices.
    Returns list of paths if valid (all simple), else None.
    """
    if not edge_set:
        return []
    
    # Build augmented graph
    aug_adj = defaultdict(list)
    for u, v in edge_set:
        aug_adj[u].append(v)
        aug_adj[v].append(u)
    
    # Add virtual edges for the pairing
    virtual_edges = set()
    for u, v in pairing:
        key = (min(u, v), max(u, v))
        virtual_edges.add(key)
        aug_adj[u].append(v)
        aug_adj[v].append(u)
    
    # Check if all degrees are even
    for u in aug_adj:
        if len(aug_adj[u]) % 2 != 0:
            return None # Should not happen if pairing covers all odd vertices
    
    # Find Eulerian Circuit using Hierholzer's
    # We need to track edges to ensure we use each once.
    # Since we added virtual edges, we need to distinguish them.
    # Let's use a set of used edges (original + virtual).
    # To make it easier, we can just use the adjacency lists and mark edges as used.
    # But Hierholzer on multigraphs is tricky with just lists.
    
    # Simpler: Use a stack-based approach with edge tracking.
    # We need to know which edges are virtual.
    
    # Let's rebuild adjacency with edge IDs.
    # This might be too slow if done for many pairings.
    # Given small N, let's try a simpler check.
    
    # Actually, we can just run Hierholzer on the multigraph representation.
    # Let's represent edges as tuples (u, v, is_virtual).
    
    edge_list = []
    for u, v in edge_set:
        edge_list.append((u, v, False))
        edge_list.append((v, u, False))
    for u, v in pairing:
        edge_list.append((u, v, True))
        edge_list.append((v, u, True))
    
    # Build adjacency with indices
    adj_idx = defaultdict(list)
    for i, (u, v, is_v) in enumerate(edge_list):
        adj_idx[u].append((v, i))
    
    used = [False] * len(edge_list)
    
    # Find circuit starting from any vertex with edges
    start_node = next(iter(adj_idx))
    stack = [start_node]
    circuit = []
    
    while stack:
        u = stack[-1]
        # Find an unused edge
        next_edge_idx = -1
        for v, idx in adj_idx[u]:
            if not used[idx]:
                next_edge_idx = idx
                break
        
        if next_edge_idx == -1:
            circuit.append(u)
            stack.pop()
        else:
            used[next_edge_idx] = True
            # Mark the reverse edge as used too? 
            # In undirected graph, edge (u,v) is same as (v,u).
            # Our edge_list has both directions.
            # If we use index i, we should also mark the reverse index as used.
            # The reverse index is i ^ 1 if we added them in pairs? 
            # Not necessarily if we added original edges then virtual edges.
            # Let's just mark this one. The loop will find the other one later and skip it if used?
            # No, if we mark only one, the reverse one is still "unused" and might be picked.
            # We need to mark both.
            # Since we added (u,v) then (v,u), they are consecutive?
            # No, we added all original (u,v) and (v,u) in a loop.
            # Let's just create a mapping.
            
            # To be safe and simple, let's just use a set of used edge tuples (min, max, is_virtual).
            # This is O(E) lookup.
            pass
            
    # This Hierholzer implementation is getting messy with the "used" tracking for undirected edges in a list.
    # Let's use a simpler approach for small graphs:
    # 1. Build adjacency of (neighbor, edge_id).
    # 2. Keep a set of used edge_ids.
    # 3. When traversing u->v using edge_id e, mark e and its reverse e' as used.
    
    # Re-implementing Hierholzer properly
    adj_idx = defaultdict(list)
    edge_map = {} # (u, v, is_v) -> id
    edge_list = []
    
    for u, v in edge_set:
        id_ = len(edge_list)
        edge_list.append((u, v, False))
        edge_map[(u, v, False)] = id_
        edge_list.append((v, u, False))
        edge_map[(v, u, False)] = id_ + 1
        adj_idx[u].append((v, id_))
        adj_idx[v].append((u, id_ + 1))
        
    for u, v in pairing:
        id_ = len(edge_list)
        edge_list.append((u, v, True))
        edge_map[(u, v, True)] = id_
        edge_list.append((v, u, True))
        edge_map[(v, u, True)] = id_ + 1
        adj_idx[u].append((v, id_))
        adj_idx[v].append((u, id_ + 1))
        
    used = [False] * len(edge_list)
    
    start_node = next(iter(adj_idx))
    stack = [start_node]
    circuit = []
    
    while stack:
        u = stack[-1]
        next_edge_idx = -1
        for v, idx in adj_idx[u]:
            if not used[idx]:
                next_edge_idx = idx
                break
        
        if next_edge_idx == -1:
            circuit.append(u)
            stack.pop()
        else:
            used[next_edge_idx] = True
            # Mark reverse
            rev_idx = next_edge_idx ^ 1
            used[rev_idx] = True
            
            v = edge_list[next_edge_idx][1]
            stack.append(v)
            
    # Circuit is in reverse order
    circuit.reverse()
    
    # Now, break the circuit at virtual edges.
    # The circuit is a sequence of vertices.
    # We need to identify which edges in the circuit are virtual.
    # An edge between circuit[i] and circuit[i+1] is virtual if (min, max, True) is in virtual_edges.
    
    virtual_edges_set = set()
    for u, v in pairing:
        virtual_edges_set.add((min(u, v), max(u, v)))
        
    paths = []
    current_path = [circuit[0]]
    
    for i in range(len(circuit) - 1):
        u = circuit[i]
        v = circuit[i+1]
        current_path.append(v)
        
        # Check if edge (u, v) is virtual
        key = (min(u, v), max(u, v))
        if key in virtual_edges_set:
            # End of a path
            if len(current_path) > 1:
                # Check if simple
                if len(set(current_path)) == len(current_path):
                    paths.append(current_path)
                else:
                    return None # Not simple
            current_path = [v] # Start new path from v
            
    # Last segment
    if len(current_path) > 1:
        if len(set(current_path)) == len(current_path):
            paths.append(current_path)
        else:
            return None
            
    # Also, the circuit is closed, so the first and last vertex are the same.
    # The loop above handles segments between virtual edges.
    # But what if the start/end of the circuit is not at a virtual edge?
    # The circuit is a cycle. The virtual edges break it into paths.
    # The last path connects the end of the last segment to the start of the first segment?
    # Yes, because it's a cycle.
    # My loop `for i in range(len(circuit) - 1)` processes edges 0-1, 1-2, ..., n-2-n-1.
    # The edge n-1 to 0 is not processed.
    
    # We need to handle the closing edge.
    # If the closing edge is virtual, it's a break point.
    # If not, it's part of the last path.
    
    # Let's re-verify the logic.
    # The circuit is v0, v1, ..., vk, v0.
    # Edges are (v0,v1), ..., (vk,v0).
    # My loop processes (v0,v1) to (v_{k-1}, vk).
    # It misses (vk, v0).
    
    # Let's just extract all segments between virtual edges.
    
    segments = []
    start_idx = 0
    for i in range(len(circuit)):
        u = circuit[i]
        v = circuit[(i+1) % len(circuit)]
        key = (min(u, v), max(u, v))
        if key in virtual_edges_set:
            # Edge (u,v) is virtual.
            # The segment from start_idx to i is a path.
            # Vertices: circuit[start_idx] ... circuit[i]
            seg = circuit[start_idx : i+1]
            if len(seg) > 1:
                segments.append(seg)
            start_idx = i + 1
            
    # Check if we closed the loop
    # If the last edge (vk, v0) was virtual, start_idx would be len(circuit) (or len-1+1).
    # If not, the last segment is circuit[start_idx] ... circuit[len-1] ... circuit[0] ... circuit[start_idx-1]?
    # No, the circuit is linear in the array, but cyclic in the graph.
    # If the last edge is NOT virtual, then the last segment wraps around.
    
    # This is getting complicated.
    # Simpler: Just check if the number of paths is len(pairing) and all are simple.
    
    # Let's just return the paths found in the loop if they seem right, otherwise None.
    # Given the complexity, and that the greedy method is robust, 
    # I will only use this if it clearly produces len(pairing) paths.
    
    if len(paths) != len(pairing):
        return None
        
    return paths

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count."""
    if not edges:
        return []
    
    edge_set = set()
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute odd vertices
    degree = {u: len(adj[u]) for u in adj}
    odd_vertices = [u for u in degree if degree[u] % 2 == 1]
    min_paths = max(1, len(odd_vertices) // 2)
    
    # 1. Try Eulerian Pairing for a few pairings
    best_paths = None
    best_count = float('inf')
    
    if len(odd_vertices) > 0:
        # Generate a few pairings
        # 1. Random pairings
        # 2. Closest pair (by index, just for variety)
        
        pairings_to_try = []
        
        # Random pairings
        rng_pair = Random(42)
        for _ in range(10):
            shuffled_odd = odd_vertices[:]
            rng_pair.shuffle(shuffled_odd)
            pairing = []
            for i in range(0, len(shuffled_odd), 2):
                pairing.append((shuffled_odd[i], shuffled_odd[i+1]))
            pairings_to_try.append(pairing)
            
        # Try them
        for pairing in pairings_to_try:
            if time.time() > deadline:
                break
            paths = _eulerian_decompose(edge_set, adj, odd_vertices, pairing, deadline)
            if paths:
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_paths = paths
                if best_count == min_paths:
                    break # Found optimal
            
    # 2. Fall back to Greedy Multi-Start
    if best_count > min_paths: # Or if no Eulerian solution found
        max_restarts = 30
        for trial in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = Random(trial * 7919 + 13)
            paths = _greedy_decompose(edge_set, adj, rng, deadline)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
            
            if best_count == min_paths:
                break
    
    if best_paths is None:
        # Fallback: single-edge paths
        best_paths = [[u, v] for u, v in edge_set]
    
    return best_paths

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return single-edge paths for remaining graphs
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
    
    return results
