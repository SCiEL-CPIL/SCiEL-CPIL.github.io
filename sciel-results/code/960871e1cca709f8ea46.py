"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        remaining = set(edges)
        cycles = []
        singles = set()
        
        # Build dynamic adjacency for remaining edges
        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        adj = build_adj(remaining)

        def find_longest_cycle(start, max_len=50):
            """Find a long cycle starting and ending at start using randomized DFS."""
            if len(remaining) < 3:
                return None
            
            best_cycle = None
            
            # Try multiple random walks to find a long cycle
            for _ in range(100):  # 100 attempts
                path = [start]
                visited = {start}
                current = start
                success = False
                
                for step in range(max_len):
                    neighbors = [nb for nb in adj.get(current, []) 
                                if (min(current, nb), max(current, nb)) in remaining]
                    
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if start in neighbors and len(path) >= 3:
                        # Close the cycle
                        path.append(start)
                        success = True
                        break
                    
                    if unvisited:
                        # Randomly pick an unvisited neighbor
                        next_node = random.choice(unvisited)
                    else:
                        # Only visited neighbors left, try to close cycle
                        if start in neighbors and len(path) >= 3:
                            path.append(start)
                            success = True
                            break
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if success and (best_cycle is None or len(path) > len(best_cycle)):
                    best_cycle = path[:]
            
            return best_cycle

        def find_cycle_greedy():
            """Find a cycle using a greedy approach that prefers longer cycles."""
            if len(remaining) < 3:
                return None
            
            # Find vertices with degree >= 2
            vertex_degrees = {}
            for u, v in remaining:
                vertex_degrees[u] = vertex_degrees.get(u, 0) + 1
                vertex_degrees[v] = vertex_degrees.get(v, 0) + 1
            
            candidates = [v for v in vertex_degrees if vertex_degrees[v] >= 2]
            if not candidates:
                return None
            
            best_cycle = None
            
            # Try each candidate vertex to find the best cycle
            for start in candidates:
                if time.time() > deadline - 10:
                    break
                cycle = find_longest_cycle(start)
                if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                    best_cycle = cycle
            
            return best_cycle

        # Greedily extract cycles, preferring longer ones
        while remaining:
            if time.time() > deadline - 20:
                break
                
            cycle = find_cycle_greedy()
            
            if cycle:
                # Extract this cycle
                cycle_edges = []
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    if edge in remaining:
                        cycle_edges.append(edge)
                        remaining.discard(edge)
                    else:
                        # Should not happen if logic is correct, but break if it does
                        break
                if len(cycle_edges) == len(cycle) - 1:
                    cycles.append(cycle[:-1][:])
                    # Rebuild adjacency
                    adj = build_adj(remaining)
                else:
                    # Recovery: put edges back
                    for e in cycle_edges:
                        remaining.add(e)
                    break
            else:
                # No more cycles; take remaining edges as singles
                for e in sorted(remaining):
                    singles.add(e)
                remaining = set()
                break

        # If we hit the deadline or finished loop, add remaining edges as singles
        if remaining:
            for e in sorted(remaining):
                singles.add(e)
            remaining = set()

        # Refinement: try to merge single edges into cycles
        # Move: If (u,v) and (v,w) are singles, and (u,w) is in a cycle, 
        # replace (u,w) in the cycle with u-v-w.
        
        def edge_in_cycle(edge, cycles):
            """Find index of cycle containing the edge."""
            u, v = edge
            for i, c in enumerate(cycles):
                # Check if edge (u,v) is in cycle c
                # Cycle is list of vertices [v1, v2, ..., vk]
                # Edges are (v1,v2), ..., (vk,v1)
                found = False
                for j in range(len(c)):
                    a = c[j]
                    b = c[(j + 1) % len(c)]
                    if (min(a,b), max(a,b)) == (u,v):
                        return i, j
                # Also check reverse? No, edge is undirected.
            return -1, -1

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Build a list of singles to iterate over (since we modify it)
            single_list = list(singles)
            for e1 in single_list:
                if e1 not in singles: continue # Might have been removed
                u1, v1 = e1
                # Find other single edges sharing v1 or u1
                # We need pairs (u,v) and (v,w)
                # So for vertex v1, look for another single (v1, w)
                neighbors_in_singles = set()
                for e2 in singles:
                    if e1 == e2: continue
                    a2, b2 = e2
                    if a2 == v1 or b2 == v1:
                        if a2 == v1: neighbors_in_singles.add(b2)
                        else: neighbors_in_singles.add(a2)
                    if a2 == u1 or b2 == u1:
                        if a2 == u1: neighbors_in_singles.add(b2)
                        else: neighbors_in_singles.add(a2)
                
                # For each neighbor w in neighbors_in_singles:
                # We have singles (u1, v1) and (v1, w) OR (w, u1) and (u1, v1)
                # Case 1: (u1, v1) and (v1, w). Check if (u1, w) is in a cycle.
                # Case 2: (w, u1) and (u1, v1). Check if (w, v1) is in a cycle.
                
                for w in neighbors_in_singles:
                    # Check if (v1, w) is a single
                    e2 = (min(v1, w), max(v1, w))
                    if e2 in singles:
                        # We have (u1, v1) and (v1, w)
                        target_edge = (min(u1, w), max(u1, w))
                        ci, pos = edge_in_cycle(target_edge, cycles)
                        if ci != -1:
                            # Perform merge
                            # Remove singles
                            singles.discard(e1)
                            singles.discard(e2)
                            # Update cycle
                            c = cycles[ci]
                            # Find the two vertices u1, w in the cycle adjacent to each other
                            # The edge was at pos.
                            # c[pos] and c[(pos+1)%len(c)] are the endpoints.
                            a = c[pos]
                            b = c[(pos + 1) % len(c)]
                            # We need to replace edge (a,b) with a-v1-w or w-v1-b
                            # Since target_edge is (u1,w), and e1 is (u1,v1), e2 is (v1,w)
                            # The path is u1 - v1 - w
                            # We need to orient the cycle such that we go from u1 to w via v1
                            # If a == u1 and b == w:
                            #   Replace segment a...b with a, v1, b
                            #   New cycle: ... a, v1, b ...
                            #   This means inserting v1 between a and b.
                            #   But we also need to ensure the rest of the cycle is intact.
                            #   The cycle is a list.
                            #   If we insert v1 at pos+1, the edge (a, v1) and (v1, b) are formed.
                            #   Wait, the original edge was (a,b).
                            #   We remove edge (a,b) from the cycle's edge set.
                            #   We add edges (a,v1) and (v1,b).
                            #   This is valid if (a,v1) and (v1,b) are the singles we removed.
                            #   Yes.
                            
                            # Insert v1 into the cycle list at position pos+1
                            # Be careful with indices.
                            # c = [v0, v1, ..., vk-1]
                            # Edge at pos is (c[pos], c[pos+1]).
                            # We want to replace this edge with path c[pos] - v_mid - c[pos+1].
                            # So insert v_mid at pos+1.
                            
                            # Check orientation
                            if a == u1 and b == w:
                                # Insert v1 after a
                                c.insert(pos + 1, v1)
                            elif a == w and b == u1:
                                # Insert v1 after a (which is w)
                                # Path w - v1 - u1
                                c.insert(pos + 1, v1)
                            else:
                                # This shouldn't happen if edge_in_cycle found the edge
                                pass
                            
                            improved = True
                            break
                    # Check other pairing: (w, u1) and (u1, v1)
                    e2_alt = (min(w, u1), max(w, u1))
                    if e2_alt in singles:
                        # We have (w, u1) and (u1, v1)
                        target_edge = (min(w, v1), max(w, v1))
                        ci, pos = edge_in_cycle(target_edge, cycles)
                        if ci != -1:
                            # Remove singles
                            singles.discard(e1)
                            singles.discard(e2_alt)
                            # Update cycle
                            c = cycles[ci]
                            a = c[pos]
                            b = c[(pos + 1) % len(c)]
                            # Path w - u1 - v1
                            # Edge (w, v1) is replaced by w - u1 - v1
                            if a == w and b == v1:
                                c.insert(pos + 1, u1)
                            elif a == v1 and b == w:
                                c.insert(pos + 1, u1)
                            else:
                                pass
                            improved = True
                            break
                if improved:
                    break
            # If no improvement in this pass, break
            if not improved:
                break

        # Combine cycles and singles into final list of pieces
        pieces = []
        for c in cycles:
            pieces.append(c[:])
        for e in singles:
            pieces.append([e[0], e[1]])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
