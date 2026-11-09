"""Decompose each fixed instance graph into edge-disjoint cycles + single edges."""
import time
import random
from random import Random
from collections import defaultdict


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _edge_key(a, b):
    return (min(a, b), max(a, b))


def _find_all_cycles(adj, avail, max_cycles=50, deadline=None):
    """Find many edge-disjoint cycles using iterative DFS with randomization."""
    cycles = []
    remaining_adj = defaultdict(set)
    for (u, v) in avail:
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    remaining_edges = set(avail)
    n = max(remaining_adj.keys()) + 1 if remaining_adj else 0
    
    for _ in range(max_cycles):
        if deadline and time.time() > deadline:
            break
        if not remaining_edges:
            break
        
        # Find vertices with degree >= 2
        candidates = [v for v in remaining_adj if len(remaining_adj[v]) >= 2]
        if not candidates:
            break
        
        # Try multiple random starts
        best_cycle = None
        random.shuffle(candidates)
        for start in candidates[:min(len(candidates), 20)]:
            if deadline and time.time() > deadline:
                break
            
            # DFS to find cycle back to start
            stack = [(start, [start], {start})]
            found = None
            visit_count = 0
            max_visits = 5000
            
            while stack and visit_count < max_visits:
                if deadline and time.time() > deadline:
                    return cycles
                cur, path, visited = stack.pop()
                visit_count += 1
                
                if len(path) >= 3 and start in remaining_adj.get(cur, set()):
                    if _edge_key(start, cur) in remaining_edges:
                        found = path[:]
                        break
                
                if len(path) >= 20:
                    continue
                
                nbrs = [nb for nb in remaining_adj.get(cur, set()) 
                        if _edge_key(cur, nb) in remaining_edges]
                # Sort by degree descending for better cycles
                nbrs.sort(key=lambda x: len(remaining_adj.get(x, set())), reverse=True)
                
                for nb in reversed(nbrs):
                    if nb not in visited:
                        stack.append((nb, path + [nb], visited | {nb}))
                    elif nb == start and len(path) >= 3:
                        if _edge_key(start, cur) in remaining_edges:
                            found = path[:]
                            break
                if found:
                    break
            
            if found and len(found) > (len(best_cycle) if best_cycle else 0):
                best_cycle = found
                if len(best_cycle) >= 8:
                    break
        
        if best_cycle:
            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = _edge_key(a, b)
                remaining_edges.discard(e)
                remaining_adj[a].discard(b)
                remaining_adj[b].discard(a)
    
    return cycles


def _local_search(pieces, edges_set, deadline):
    """Try to merge adjacent single edges into cycles via local search."""
    # Build edge-to-piece mapping
    edge_to_piece = {}
    for idx, piece in enumerate(pieces):
        if len(piece) == 2:
            e = _edge_key(piece[0], piece[1])
            edge_to_piece[e] = idx
        else:
            for i in range(len(piece)):
                a, b = piece[i], piece[(i + 1) % len(piece)]
                e = _edge_key(a, b)
                edge_to_piece[e] = idx
    
    # Try to find 3-cycles among single edges
    single_edges = [p for p in pieces if len(p) == 2]
    adj = defaultdict(set)
    for (u, v) in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find triangles in the single-edge subgraph
    single_edge_set = set()
    for p in single_edges:
        e = _edge_key(p[0], p[1])
        single_edge_set.add(e)
    
    merged = False
    new_pieces = []
    used_single = set()
    
    for i, p1 in enumerate(single_edges):
        if i in used_single:
            continue
        if deadline and time.time() > deadline:
            break
        u, v = p1[0], p1[1]
        e1 = _edge_key(u, v)
        
        # Look for a third vertex w such that (v,w) and (u,w) are also single edges
        for w in adj.get(v, set()):
            if w == u:
                continue
            e2 = _edge_key(v, w)
            e3 = _edge_key(u, w)
            if e2 in single_edge_set and e3 in single_edge_set:
                # Found a triangle of single edges
                # Find their indices
                for j, p2 in enumerate(single_edges):
                    if j in used_single:
                        continue
                    if _edge_key(p2[0], p2[1]) == e2:
                        for k, p3 in enumerate(single_edges):
                            if k in used_single:
                                continue
                            if _edge_key(p3[0], p3[1]) == e3:
                                # Merge into cycle
                                used_single.add(i)
                                used_single.add(j)
                                used_single.add(k)
                                new_pieces.append([u, v, w])
                                merged = True
                                break
                        break
                if merged:
                    break
        if merged:
            continue
    
    if merged:
        # Rebuild pieces list
        final_pieces = []
        for idx, piece in enumerate(pieces):
            if len(piece) == 2:
                e = _edge_key(piece[0], piece[1])
                # Check if this single edge was merged
                # We need to track which singles were used
                pass
        
        # Simpler approach: just return the improvement count
        # Actually, let's do it properly
        single_set = set()
        for p in single_edges:
            single_set.add(_edge_key(p[0], p[1]))
        
        # Rebuild
        result = []
        for piece in pieces:
            if len(piece) == 2:
                e = _edge_key(piece[0], piece[1])
                # Keep it (we'll handle merging separately)
                result.append(piece)
            else:
                result.append(piece)
        
        # This local search is complex; skip for now and rely on better cycle finding
        return pieces
    
    return pieces


def _decompose(edges, deadline):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    avail = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Phase 1: Find cycles using randomized DFS
    cycles = _find_all_cycles(adj, avail, max_cycles=100, deadline=deadline)
    for cycle in cycles:
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            e = _edge_key(a, b)
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(cycle)
    
    # Phase 2: Try to find more cycles in remaining edges
    if avail and time.time() < deadline:
        more_cycles = _find_all_cycles(adj, avail, max_cycles=50, deadline=deadline)
        for cycle in more_cycles:
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = _edge_key(a, b)
                avail.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(cycle)
    
    # Phase 3: Extract 3-cycles from remaining
    if avail and time.time() < deadline:
        rem_adj = defaultdict(set)
        for (u, v) in avail:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        for u in sorted(rem_adj.keys()):
            if time.time() > deadline:
                break
            nbrs_u = sorted(rem_adj[u])
            for i in range(len(nbrs_u)):
                v = nbrs_u[i]
                if v <= u:
                    continue
                for w in nbrs_u[i+1:]:
                    if w in rem_adj.get(v, set()):
                        e1 = _edge_key(u, v)
                        e2 = _edge_key(v, w)
                        e3 = _edge_key(u, w)
                        if e1 in avail and e2 in avail and e3 in avail:
                            avail.discard(e1)
                            avail.discard(e2)
                            avail.discard(e3)
                            adj[u].discard(v)
                            adj[v].discard(u)
                            adj[v].discard(w)
                            adj[w].discard(v)
                            adj[u].discard(w)
                            adj[w].discard(u)
                            pieces.append([u, v, w])
    
    # Phase 4: Remaining edges as singletons
    for e in sorted(avail):
        pieces.append(list(e))
    
    return pieces


def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    total_size = sum(n * n for n, _, _ in specs)
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue
        frac = (n * n) / total_size
        graph_deadline = time.time() + max(5.0, remaining_time * frac * 0.95)
        results.append(_decompose(edges, graph_deadline))
    return results
