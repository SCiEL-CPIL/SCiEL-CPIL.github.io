import time
import random
from collections import defaultdict
import heapq

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_greedy(edges, n, rng, deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find a long cycle
        best_cycle = None
        best_len = 0
        
        # Heuristic: Try starting from high-degree vertices first
        candidates = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        for start in candidates:
            if not adj[start]:
                continue
            
            # Randomized DFS to find a long cycle
            # We limit the number of random restarts per start to keep it fast
            for _ in range(5):
                path = [start]
                visited = {start}
                current = start
                
                # Backtracking search for a long cycle
                # Simple greedy walk with randomization
                while True:
                    neighbors = list(adj[current])
                    unvisited = [v for v in neighbors if v not in visited]
                    
                    if not unvisited:
                        if len(path) >= 3 and start in adj[current]:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break
                    else:
                        # Prefer unvisited nodes with higher degree?
                        # Or just random
                        next_node = rng.choice(unvisited)
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                        
        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            # No cycle found, take a single edge
            edge_removed = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_removed = True
                        break
                if edge_removed:
                    break
            if not edge_removed:
                break
                
    return pieces

def _polish_pieces(pieces, n, deadline):
    """
    Polish the decomposition by finding triangles and 4-cycles 
    composed of single edges.
    """
    # Map edge to piece index
    edge_to_piece = {}
    for i, p in enumerate(pieces):
        if len(p) == 2:
            u, v = p
            edge_to_piece[(u, v)] = i
            edge_to_piece[(v, u)] = i
        else:
            for j in range(len(p)):
                u = p[j]
                v = p[(j + 1) % len(p)]
                edge_to_piece[(u, v)] = i
                edge_to_piece[(v, u)] = i
                
    # Build adjacency for single edges only?
    # We need to find triangles in the graph of single edges.
    single_edges = []
    for i, p in enumerate(pieces):
        if len(p) == 2:
            single_edges.append((p[0], p[1], i))
            
    # Build adj for single edges
    single_adj = defaultdict(set)
    for u, v, idx in single_edges:
        single_adj[u].add(v)
        single_adj[v].add(u)
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find triangles in single_adj
        for u in single_adj:
            if u not in single_adj: continue
            for v in list(single_adj[u]):
                if v <= u: continue
                for w in list(single_adj[v]):
                    if w <= v: continue
                    if w in single_adj[u]:
                        # Triangle (u, v, w)
                        # Check if edges are still single
                        # Get indices
                        idx_uv = edge_to_piece.get((u, v))
                        idx_vw = edge_to_piece.get((v, w))
                        idx_wu = edge_to_piece.get((w, u))
                        
                        if idx_uv is not None and idx_vw is not None and idx_wu is not None:
                            # Verify they are still single edges
                            if len(pieces[idx_uv]) == 2 and len(pieces[idx_vw]) == 2 and len(pieces[idx_wu]) == 2:
                                # Replace with cycle
                                new_cycle = [u, v, w]
                                pieces[idx_uv] = new_cycle
                                # Remove other pieces
                                pieces[idx_vw] = []
                                pieces[idx_wu] = []
                                
                                # Update edge_to_piece
                                for edge in [(u, v), (v, u), (v, w), (w, v), (w, u), (u, w)]:
                                    edge_to_piece[edge] = idx_uv
                                    
                                # Update single_adj
                                for a in [u, v, w]:
                                    for b in [u, v, w]:
                                        if a != b:
                                            single_adj[a].discard(b)
                                            
                                changed = True
                                break
                if changed: break
            if changed: break
        if changed: break
        
    # Clean up empty pieces
    final_pieces = [p for p in pieces if p]
    return final_pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    num_attempts = 50
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 0.5:
            break
        
        rng = random.Random(base_seed + i)
        pieces = _decompose_greedy(edges, n, rng, deadline)
        
        # Polish
        if time.time() < deadline - 0.5:
            pieces = _polish_pieces(pieces, n, deadline)
            
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
    
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 3.0:
            graph_deadline = start_time + total_budget - 3.0
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
