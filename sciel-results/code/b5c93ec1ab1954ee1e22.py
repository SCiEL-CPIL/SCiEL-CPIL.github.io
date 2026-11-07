"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 15

def _is_valid(adj, n):
    """Check if no 3-edge path s-u-v-t has deg(s)==deg(t)."""
    deg = [sum(adj[i]) for i in range(n)]
    # For every edge (u,v), for every s in N(u)\{v}, for every t in N(v)\{u}, s!=t:
    #   if deg[s] == deg[t], invalid
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]:
                continue
            for s in range(n):
                if s == u or s == v:
                    continue
                if not adj[s][u]:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s:
                        continue
                    if not adj[t][v]:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _add_edge(adj, u, v):
    adj[u][v] = 1
    adj[v][u] = 1

def _remove_edge(adj, u, v):
    adj[u][v] = 0
    adj[v][u] = 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = N
    
    # Start with empty graph
    adj = [[0]*n for _ in range(n)]
    
    # Greedy: try to add all possible edges in a smart order
    # Order edges by "benefit" - prefer edges that connect low-degree to high-degree
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    
    # First pass: greedy add
    for u, v in all_edges:
        if time.time() > deadline:
            break
        _add_edge(adj, u, v)
        if _is_valid(adj, n):
            pass  # keep
        else:
            _remove_edge(adj, u, v)
    
    best_adj = [row[:] for row in adj]
    best_count = _count_edges(adj, n)
    
    # Local search: try to improve by swapping edges
    while time.time() < deadline:
        improved = False
        # Try adding a missing edge
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            _add_edge(adj, u, v)
            if _is_valid(adj, n):
                improved = True
                best_adj = [row[:] for row in adj]
                best_count = _count_edges(adj, n)
            else:
                _remove_edge(adj, u, v)
        
        # Try removing one edge and adding two
        if not improved and time.time() < deadline:
            present = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u][v]]
            random.shuffle(present)
            for rem_u, rem_v in present[:50]:  # limit for speed
                if time.time() > deadline:
                    break
                _remove_edge(adj, rem_u, rem_v)
                missing2 = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
                random.shuffle(missing2)
                added = 0
                for add_u, add_v in missing2:
                    if time.time() > deadline:
                        break
                    _add_edge(adj, add_u, add_v)
                    if _is_valid(adj, n):
                        added += 1
                        if added >= 2:
                            break
                    else:
                        _remove_edge(adj, add_u, add_v)
                if added > 1:
                    # Net gain
                    best_adj = [row[:] for row in adj]
                    best_count = _count_edges(adj, n)
                    improved = True
                else:
                    # Revert
                    for add_u, add_v in missing2:
                        if adj[add_u][add_v]:
                            _remove_edge(adj, add_u, add_v)
                    _add_edge(adj, rem_u, rem_v)
        
        if not improved:
            break
    
    # Build result from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
