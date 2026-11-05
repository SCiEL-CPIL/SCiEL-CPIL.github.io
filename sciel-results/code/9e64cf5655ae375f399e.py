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

def _find_long_cycle(adj, n, rng, max_starts=None):
    """Find a long cycle using greedy with degree-aware scoring."""
    best_cycle = None
    best_len = 0
    
    # Start from vertices with higher degree first
    candidates = [v for v in range(n) if adj[v]]
    if not candidates:
        return None
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    if max_starts:
        candidates = candidates[:max_starts]
    
    for start in candidates:
        # Multiple randomized trials from this start
        for trial in range(8):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                
                if not unvisited:
                    # Check if we can close the cycle
                    if len(path) >= 3 and start in adj[current]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Score candidates: prefer unvisited neighbors that lead to more options
                    # and prefer closing if close to start
                    scored = []
                    for v in unvisited:
                        score = len(adj[v])
                        # Bonus if v is connected to start (can close soon)
                        if start in adj[v] and len(path) >= 3:
                            score += 10
                        # Penalty if v has only one unvisited neighbor (dead end risk)
                        v_unvisited = [w for w in adj[v] if w not in visited]
                        if len(v_unvisited) == 1:
                            score -= 2
                        scored.append((score, v))
                    
                    # Pick with some randomness
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = scored[:min(3, len(scored))]
                    if rng.random() < 0.7:
                        next_node = top[0][1]
                    else:
                        next_node = rng.choice([v for _, v in top])
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
    
    return best_cycle

def _polish_pieces(pieces, adj, n, rng, deadline):
    """Try to merge single-edge pieces into cycles by finding cycles in the edge-pieces graph."""
    # Build a graph where nodes are edge-pieces, edges connect pieces that share a vertex
    edge_pieces = []
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            edge_pieces.append(i)
    
    if len(edge_pieces) < 2:
        return pieces
    
    # Build adjacency among edge pieces: two edge pieces are adjacent if they share a vertex
    piece_adj = defaultdict(set)
    for i in edge_pieces:
        u, v = pieces[i]
        for j in edge_pieces:
            if i == j:
                continue
            a, b = pieces[j]
            if u in (a, b) or v in (a, b):
                piece_adj[i].add(j)
    
    # Try to find cycles in this meta-graph and merge
    new_pieces = list(pieces)
    used = set()
    
    for _ in range(20):
        if time.time() > deadline:
            break
        # Find a cycle in the meta-graph
        meta_cycle = None
        for start in edge_pieces:
            if start in used:
                continue
            path = [start]
            visited = {start}
            current = start
            while True:
                unvisited_neighbors = [v for v in piece_adj.get(current, []) if v not in visited]
                if not unvisited_neighbors:
                    if len(path) >= 3 and start in piece_adj.get(current, []):
                        meta_cycle = path[:]
                    break
                next_node = rng.choice(unvisited_neighbors)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            if meta_cycle:
                break
        
        if not meta_cycle:
            break
        
        # Merge these edge pieces into a single cycle
        # The cycle in meta-graph means the edges form a cycle in the original graph
        # Extract the actual cycle
        actual_cycle = []
        for idx in meta_cycle:
            u, v = new_pieces[idx]
            if not actual_cycle:
                actual_cycle.append(u)
                actual_cycle.append(v)
            else:
                # Find which endpoint connects to the last vertex
                last = actual_cycle[-1]
                if u == last:
                    actual_cycle.append(v)
                elif v == last:
                    actual_cycle.append(u)
                else:
                    # Try the other orientation
                    actual_cycle.append(u)
        
        # Verify it's a valid cycle
        m = len(actual_cycle)
        if m >= 3 and len(set(actual_cycle)) == m and actual_cycle[0] in adj.get(actual_cycle[-1], set()):
            # Remove the old edge pieces and add the cycle
            for idx in meta_cycle:
                u, v = new_pieces[idx]
                adj[u].discard(v)
                adj[v].discard(u)
                used.add(idx)
            for i in range(m):
                a = actual_cycle[i]
                b = actual_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            new_pieces = [p for i, p in enumerate(new_pieces) if i not in used]
            new_pieces.append(actual_cycle)
            # Rebuild edge_pieces list
            edge_pieces = [i for i, p in enumerate(new_pieces) if len(p) == 2]
            used = set()
            piece_adj = defaultdict(set)
            for i in edge_pieces:
                u, v = new_pieces[i]
                for j in edge_pieces:
                    if i == j:
                        continue
                    a, b = new_pieces[j]
                    if u in (a, b) or v in (a, b):
                        piece_adj[i].add(j)
        else:
            break
    
    return new_pieces

def _decompose_once(edges, n, rng, deadline):
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
        cycle = _find_long_cycle(adj, n, rng, max_starts=15)
        
        if cycle:
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cycle)
        else:
            # Remove a single edge
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
    
    # Polish: try to merge edge pieces into cycles
    pieces = _polish_pieces(pieces, adj, n, rng, deadline)
    
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    num_attempts = 60
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 0.5:
            break
        
        rng = random.Random(base_seed + i * 7 + 13)
        pieces = _decompose_once(edges, n, rng, deadline)
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
