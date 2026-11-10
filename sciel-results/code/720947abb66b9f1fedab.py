"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _beam_search_cycle(start, adj, remaining, beam_width=20, max_depth=30):
    """
    Find a long cycle starting from 'start' using beam search.
    Returns the cycle as a list of vertices or None.
    """
    if start not in adj or len(adj[start]) < 2:
        return None
        
    # State: (node, path_tuple, visited_set)
    # We use tuples for paths for hashing/efficiency in beam
    beams = [(start, (start,), {start})]
    best_cycle = None
    best_len = 0
    
    for depth in range(max_depth):
        next_beams = []
        for node, path, visited in beams:
            for nb in adj[node]:
                if nb not in remaining:
                    continue
                # Check if edge (node, nb) is in remaining
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                if nb == start:
                    # Cycle found
                    if len(path) >= 3:
                        # path is (start, ..., node)
                        # Cycle is path + (start) but we store vertices
                        # The cycle vertices are path
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                elif nb not in visited:
                    new_visited = visited | {nb}
                    next_beams.append((nb, path + (nb,), new_visited))
        
        if not next_beams:
            break
        
        # Prune to keep only top beam_width states based on path length
        # Longer paths are generally better
        next_beams.sort(key=lambda x: len(x[1]), reverse=True)
        beams = next_beams[:beam_width]
        
        if not beams:
            break
            
    return best_cycle

def _decompose_graph(edges, deadline):
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edges_set)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Greedy Cycle Extraction with Beam Search
    # We iterate until no more cycles are found
    while remaining and time.time() < deadline:
        # Find candidates: vertices with degree >= 2 in remaining graph
        # We can just pick any vertex with deg >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree descending to prioritize high connectivity
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try a subset of candidates to find the best cycle
        # We don't need to try ALL if beam search is good
        for start in candidates[:20]:
            if len(adj[start]) < 2:
                continue
            cycle = _beam_search_cycle(start, adj, remaining, beam_width=10, max_depth=25)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            if best_len >= 15: # Good enough, stop searching
                break
        
        if best_cycle and best_len >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            # No more cycles found
            break
    
    # Phase 2: Local Search Improvement
    # Try to merge cycles or absorb single edges into cycles
    # We represent pieces as lists. Cycles have len >= 3, edges have len == 2.
    # Note: Single edges are stored as [u, v].
    
    # First, add remaining edges as single pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    # Local Search: 
    # 1. Try to merge two cycles if their union is a single cycle.
    # 2. Try to absorb a single edge into a cycle if the union is a single cycle.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Build index of pieces
        # To make this efficient, we only check promising pairs.
        # However, N is small (<= 30 vertices, ~100-150 edges).
        # Number of pieces is small.
        
        # Re-identify cycles and edges
        cycles = [p for p in pieces if len(p) >= 3]
        singles = [p for p in pieces if len(p) == 2]
        
        # Try merging two cycles
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Check if union is a cycle
                # Union graph has vertices V1 U V2
                # If it's a cycle, every vertex in the union must have degree 2 in the union graph
                # And it must be connected.
                
                # Build union edges
                edges_c1 = set()
                for k in range(len(c1)):
                    u, v = c1[k], c1[(k+1)%len(c1)]
                    edges_c1.add(tuple(sorted((u, v))))
                    
                edges_c2 = set()
                for k in range(len(c2)):
                    u, v = c2[k], c2[(k+1)%len(c2)]
                    edges_c2.add(tuple(sorted((u, v))))
                    
                if edges_c1 & edges_c2: # Edge disjoint required, they are from decomposition so they are
                    continue
                    
                union_edges = edges_c1 | edges_c2
                # Check if union is a single cycle
                # Degrees must be 2
                deg = defaultdict(int)
                for u, v in union_edges:
                    deg[u] += 1
                    deg[v] += 1
                    
                if all(d == 2 for d in deg.values()):
                    # It is a cycle (or disjoint cycles, but we need to check connectivity)
                    # Since c1 and c2 were connected, if they share vertices, union is connected.
                    # If they don't share vertices, union is two disjoint cycles -> not a single cycle.
                    # So they must share at least one vertex to form a single cycle?
                    # Actually, if they are disjoint, union is two cycles. Degree 2 everywhere.
                    # We want to replace 2 pieces with 1 piece.
                    # So union MUST be a single connected cycle.
                    
                    # Check connectivity
                    start_v = next(iter(deg))
                    visited = {start_v}
                    stack = [start_v]
                    adj_union = defaultdict(list)
                    for u, v in union_edges:
                        adj_union[u].append(v)
                        adj_union[v].append(u)
                        
                    while stack:
                        node = stack.pop()
                        for nb in adj_union[node]:
                            if nb not in visited:
                                visited.add(nb)
                                stack.append(nb)
                                
                    if len(visited) == len(deg):
                        # It is a single cycle
                        # Extract the cycle order
                        # Start from start_v, follow neighbors
                        cycle_order = [start_v]
                        prev = None
                        curr = start_v
                        while True:
                            nexts = [x for x in adj_union[curr] if x != prev]
                            if not nexts:
                                break
                            next_node = nexts[0]
                            if next_node == start_v:
                                break
                            cycle_order.append(next_node)
                            prev, curr = curr, next_node
                            
                        # Replace pieces
                        # Remove i and j from cycles list (conceptually)
                        # We will rebuild pieces list
                        new_cycle = cycle_order
                        # Update pieces
                        # This is tricky with indices. Let's just mark for deletion and add new.
                        # For simplicity, we'll do a full rebuild in the next iteration if improved.
                        # But we need to update 'pieces' immediately.
                        
                        # Find indices in 'pieces'
                        idx_i = pieces.index(c1)
                        idx_j = pieces.index(c2)
                        
                        # Remove them
                        # Be careful with indices if i > j
                        if idx_i > idx_j:
                            del pieces[idx_i]
                            del pieces[idx_j]
                        else:
                            del pieces[idx_j]
                            del pieces[idx_i]
                        
                        pieces.append(new_cycle)
                        improved = True
                        break # Restart loops
            if improved:
                break
                
        if improved:
            continue
            
        # Try absorbing a single edge into a cycle
        for i in range(len(cycles)):
            c = cycles[i]
            edges_c = set()
            for k in range(len(c)):
                u, v = c[k], c[(k+1)%len(c)]
                edges_c.add(tuple(sorted((u, v))))
                
            for j in range(len(singles)):
                e = singles[j]
                u, v = e[0], e[1]
                e_t = tuple(sorted((u, v)))
                
                # If edge is already in cycle (shouldn't be), skip
                if e_t in edges_c:
                    continue
                    
                # Check if C U {e} is a single cycle
                union_edges = edges_c | {e_t}
                deg = defaultdict(int)
                for a, b in union_edges:
                    deg[a] += 1
                    deg[b] += 1
                    
                # In a cycle, all degrees must be 2.
                # If e connects two vertices in C, their degrees become 3.
                # So this is only possible if e connects two vertices NOT in C? No, then not connected.
                # Or if e connects two vertices in C, and we REMOVE an edge from C?
                # Local search usually involves swapping.
                # Here we just add e. If deg > 2, it's not a simple cycle.
                # So this specific move (adding an edge to a cycle to make a larger cycle)
                # is only possible if the cycle was actually a path? No, it's a cycle.
                # Adding an edge to a cycle creates a "lollipop" or adds a chord.
                # It does not form a single simple cycle unless we remove an edge.
                
                # So, "absorbing" a single edge into a cycle to form a larger cycle
                # requires that the single edge connects two vertices in the cycle,
                # and we choose one of the two resulting cycles as the new piece,
                # and the other part of the original cycle becomes single edges.
                
                # Let's implement this swap:
                # If u, v in C.
                # C has two paths between u and v.
                # P1 and P2.
                # New cycle C1 = P1 + e.
                # Remaining edges from P2 become singles.
                # Old pieces: 1 (C) + 1 (e) = 2.
                # New pieces: 1 (C1) + len(P2) (singles).
                # Gain if 1 + len(P2) < 2 => len(P2) < 1 => len(P2) = 0. Impossible for distinct u,v.
                # So this move NEVER reduces piece count?
                # Wait.
                # If len(P2) = 1, then 1 + 1 = 2. No change.
                # If len(P2) > 1, we increase pieces.
                
                # So this move is useless for minimizing piece count.
                # The only useful merge is Cycle + Cycle -> Cycle (if union is cycle).
                # Or Cycle + Path -> Cycle?
                # We don't have paths, we have singles.
                
                pass
        
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time: ~45s each
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
