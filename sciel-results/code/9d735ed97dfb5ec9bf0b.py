import time
import random
import sys
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        cycles = []
        
        def get_cycle_edges(cycle):
            es = set()
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                es.add(tuple(sorted((u, v))))
            u, v = cycle[-1], cycle[0]
            es.add(tuple(sorted((u, v))))
            return es

        def remove_cycle_edges(cycle):
            es = get_cycle_edges(cycle)
            for e in es:
                remaining_edges.discard(e)
            # Update adjacency
            for u in cycle:
                adj[u] = set()
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)

        def find_longest_cycle(time_budget):
            """
            Find the longest cycle in the remaining graph within time_budget.
            """
            if not remaining_edges:
                return None
            
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            search_deadline = time.time() + time_budget
            
            # Try starting from each candidate
            for start in candidates[:5]: # Limit starts
                if time.time() > search_deadline:
                    break
                
                visited = {start}
                path = [start]
                
                def dfs(current, path):
                    nonlocal best_cycle, best_len, search_deadline
                    
                    if time.time() > search_deadline:
                        return
                    
                    # Check if we can close the cycle
                    if len(path) >= 3:
                        if (current, start) in remaining_edges or (start, current) in remaining_edges:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                    
                    # Extend path
                    # Heuristic: visit neighbors with higher degree first? Or lower?
                    # Higher degree might lead to longer cycles.
                    neighbors = list(adj[current])
                    neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                    
                    for neighbor in neighbors:
                        if neighbor == start:
                            continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path)
                            path.pop()
                            visited.remove(neighbor)
                            if time.time() > search_deadline:
                                return

                dfs(start, path)
                
            return best_cycle

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            # Allocate time for finding the next cycle
            # As edges decrease, finding cycles might be faster, but let's keep it steady
            cycle_time_budget = 0.5
            cycle = find_longest_cycle(cycle_time_budget)
            
            if cycle:
                cycles.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found. Remaining edges are single pieces.
                break

        # Local Search Improvement
        # Check if we can swap segments of cycles with paths in remaining edges
        # to reduce the total number of pieces.
        
        def find_path_in_remaining(u, v, max_len):
            """Find a path from u to v in remaining_edges of length <= max_len."""
            # BFS
            if u == v:
                return []
            if (u, v) in remaining_edges or (v, u) in remaining_edges:
                return [u, v]
            
            visited = {u}
            queue = [(u, [u])]
            while queue:
                if time.time() > time_limit:
                    return None
                curr, path = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor in visited:
                        continue
                    # Check if edge (curr, neighbor) is in remaining
                    if (curr, neighbor) not in remaining_edges and (neighbor, curr) not in remaining_edges:
                        continue
                    new_path = path + [neighbor]
                    if len(new_path) - 1 > max_len:
                        continue
                    if neighbor == v:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
            return None

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # For each cycle, try to improve it
            for i, cycle in enumerate(cycles):
                if time.time() > time_limit:
                    break
                
                # Get edges of this cycle
                cycle_edges = get_cycle_edges(cycle)
                
                # Try to replace a segment of the cycle with a path from remaining edges
                # A cycle is a sequence of vertices.
                # Let's try to find a chord or external path.
                # If there is a path P in remaining_edges from u to v (u, v in cycle)
                # such that len(P) < len(segment(u, v) in cycle), we can swap.
                
                # To be efficient, only check a few pairs (u, v)
                # Or check all pairs if cycle is small.
                n_cycle = len(cycle)
                for i_v in range(n_cycle):
                    u = cycle[i_v]
                    for j_v in range(i_v + 1, n_cycle):
                        v = cycle[j_v]
                        
                        # Calculate length of segment from u to v in cycle
                        # Path 1: u -> ... -> v (forward)
                        len1 = j_v - i_v
                        # Path 2: v -> ... -> u (forward, which is backward from u)
                        len2 = n_cycle - len1
                        
                        # We want to replace the LONGER segment with a path in remaining edges?
                        # No, we replace a segment with a path.
                        # If we replace segment of length L_seg with path of length L_path.
                        # We remove L_seg edges from cycle, add L_path edges to cycle.
                        # We remove L_path edges from remaining, add L_seg edges to remaining.
                        # Change in pieces:
                        # Old: 1 (cycle) + L_path (remaining)
                        # New: 1 (new cycle) + L_seg (remaining)
                        # Delta = L_seg - L_path.
                        # We want Delta < 0, so L_path < L_seg.
                        
                        # So we look for a path in remaining edges between u and v
                        # that is shorter than the segment of the cycle connecting them.
                        
                        # Check both segments?
                        # Actually, if we replace segment 1, the new cycle uses segment 2 + path.
                        # If we replace segment 2, the new cycle uses segment 1 + path.
                        # In both cases, the condition is Path < Segment.
                        
                        # Let's try to find a path shorter than len1
                        if len1 > 1: # If len1 is 1, it's an edge, can't be shorter
                            path = find_path_in_remaining(u, v, len1 - 1)
                            if path:
                                # Perform swap
                                # New cycle: path + segment 2
                                # Segment 2 is from v to u in the original cycle direction
                                # Original cycle: u ... v ... u
                                # Segment 1: u ... v
                                # Segment 2: v ... u
                                # New cycle: path(u->v) + segment 2(v->u)
                                # But path is u->v. Segment 2 is v->...->u.
                                # So new cycle is [path] + [cycle[j_v:] + cycle[:i_v+1]]?
                                # Let's construct the new cycle carefully.
                                
                                # Original cycle: [c0, c1, ..., c_{n-1}]
                                # u = c_i, v = c_j
                                # Segment 1: c_i, c_{i+1}, ..., c_j
                                # Segment 2: c_j, c_{j+1}, ..., c_{n-1}, c_0, ..., c_i
                                
                                # Path: u = p_0, ..., p_k = v
                                # New cycle: p_0, ..., p_k, c_{j+1}, ..., c_{n-1}, c_0, ..., c_i
                                # Wait, p_k is v. So we connect v to c_{j+1}.
                                # And c_i connects to p_0 (u).
                                
                                new_cycle = path + cycle[j_v+1:] + cycle[:i_v+1]
                                
                                # Verify validity
                                if len(set(new_cycle)) != len(new_cycle):
                                    continue
                                if len(new_cycle) < 3:
                                    continue
                                
                                # Check if edges exist
                                valid = True
                                new_cycle_edges = get_cycle_edges(new_cycle)
                                # All edges must be in (cycle_edges U remaining_edges)
                                # And specifically, the path edges must be in remaining_edges
                                # And the segment 2 edges must be in cycle_edges
                                
                                # Remove old cycle and path edges from respective sets
                                # Add new cycle edges to cycles set?
                                # We are modifying the list of cycles and remaining_edges.
                                
                                # 1. Remove cycle_edges from remaining? No, they are not in remaining.
                                # 2. Remove path edges from remaining.
                                # 3. Add segment 1 edges to remaining.
                                # 4. Replace cycle[i] with new_cycle.
                                
                                path_edges = get_cycle_edges(path) # path is a path, not cycle, but get_cycle_edges works for edges
                                # Wait, get_cycle_edges closes the loop. Path does not close.
                                # Let's write a helper for path edges.
                                def get_path_edges(path):
                                    es = set()
                                    for i in range(len(path) - 1):
                                        u, v = path[i], path[i+1]
                                        es.add(tuple(sorted((u, v))))
                                    return es
                                    
                                p_edges = get_path_edges(path)
                                
                                # Check if p_edges are subset of remaining_edges
                                if not p_edges.issubset(remaining_edges):
                                    continue
                                    
                                # Check if new_cycle is valid
                                # Edges in new_cycle must be distinct and valid.
                                # Since path is simple and segment 2 is simple and disjoint vertices (except u,v),
                                # and we checked distinct vertices, it should be fine.
                                
                                # Execute swap
                                remaining_edges.difference_update(p_edges)
                                remaining_edges.update(get_path_edges(cycle[i_v:j_v+1])) # Segment 1 edges
                                
                                # Update adj
                                for u in cycle:
                                    adj[u] = set()
                                for u in path:
                                    adj[u] = set()
                                for u, v in remaining_edges:
                                    adj[u].add(v)
                                    adj[v].add(u)
                                
                                cycles[i] = new_cycle
                                improved = True
                                break # Restart loop for this cycle
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        # Final pieces
        pieces = list(cycles)
        for e in remaining_edges:
            pieces.append(list(e))
            
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time
        time_limits = [20.0, 30.0, 80.0]
        current_limit = time_limits[i]
        global_limit = deadline - time.time()
        if global_limit < current_limit:
            current_limit = max(1.0, global_limit)
            
        time_limit = time.time() + current_limit
        
        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
