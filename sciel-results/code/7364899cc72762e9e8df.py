"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
_N = 16
_M = 65
_DEADLINE = 145.0  # safety margin before 150s kill

def _make_adj():
    """Build K_{8,8} + edge (0,1) as adjacency matrix (list of bytearrays)."""
    adj = [bytearray(_N) for _ in range(_N)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
    adj[0][1] = 1
    adj[1][0] = 1
    return adj

def _degrees(adj):
    return [sum(adj[i]) for i in range(_N)]

def _find_triangles(adj):
    """Return list of (u,v,w) with u<v<w that form triangles."""
    tris = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if adj[u][v]:
                for w in range(v + 1, _N):
                    if adj[u][w] and adj[v][w]:
                        tris.append((u, v, w))
    return tris

def _score(adj, deg):
    """Largest degree sum over all triangles; returns (max_sum, num_triangles)."""
    best = 0
    ntri = 0
    for u in range(_N):
        for v in range(u + 1, _N):
            if adj[u][v]:
                for w in range(v + 1, _N):
                    if adj[u][w] and adj[v][w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        ntri += 1
    return best, ntri

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _swap_edges(adj, deg, e1, e2):
    """Swap: remove e1, add e2. Returns True if valid, False otherwise.
    e1, e2 are (u,v) with u<v."""
    u1, v1 = e1
    u2, v2 = e2
    # e1 must be an edge, e2 must be a non-edge
    if not adj[u1][v1] or adj[u2][v2]:
        return False
    # Also ensure e1 != e2
    if (u1, v1) == (u2, v2):
        return False
    # Apply
    adj[u1][v1] = 0
    adj[v1][u1] = 0
    adj[u2][v2] = 1
    adj[v2][u2] = 1
    deg[u1] -= 1
    deg[v1] -= 1
    deg[u2] += 1
    deg[v2] += 1
    return True

def _undo_swap(adj, deg, e1, e2):
    u1, v1 = e1
    u2, v2 = e2
    adj[u1][v1] = 1
    adj[v1][u1] = 1
    adj[u2][v2] = 0
    adj[v2][u2] = 0
    deg[u1] += 1
    deg[v1] += 1
    deg[u2] -= 1
    deg[v2] -= 1

def _random_edge(adj):
    """Pick a random edge (u,v) with u<v."""
    # Collect all edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if adj[u][v]:
                edges.append((u, v))
    return random.choice(edges)

def _random_nonedge(adj):
    """Pick a random non-edge (u,v) with u<v."""
    nonedges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if not adj[u][v]:
                nonedges.append((u, v))
    return random.choice(nonedges)

def _soft_max_triangle_score(adj, deg, beta=5.0):
    """Smooth surrogate: soft-max of triangle degree sums.
    Lower beta -> smoother. Higher beta -> closer to true max."""
    tris = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if adj[u][v]:
                for w in range(v + 1, _N):
                    if adj[u][w] and adj[v][w]:
                        tris.append(deg[u] + deg[v] + deg[w])
    if not tris:
        return 0.0
    m = max(tris)
    # log-sum-exp for stability
    shifted = [s - m for s in tris]
    lse = m + math.log(sum(math.exp(beta * s) for s in shifted)) / beta
    return lse

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Initial: K_{8,8} + (0,1)
    adj = _make_adj()
    deg = _degrees(adj)
    best_score, best_ntri = _score(adj, deg)
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    
    # Precompute all possible edges and nonedges for faster sampling
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    def _is_edge(u, v):
        return adj[u][v]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    # Count steps for progress
    steps = 0
    stuck = 0
    best_stuck = 0
    
    while time.time() - t0 < _DEADLINE:
        # Pick random edge and non-edge
        e1 = _random_edge(adj)
        e2 = _random_nonedge(adj)
        
        if not _swap_edges(adj, deg, e1, e2):
            continue
        
        new_score, new_ntri = _score(adj, deg)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            stuck = 0
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - best_score
            if delta > 0 and random.random() < math.exp(-delta / T):
                stuck += 1
            else:
                _undo_swap(adj, deg, e1, e2)
                stuck += 1
        
        T = max(T_min, T * cooling)
        steps += 1
        
        # If stuck too long, restart with higher temperature
        if stuck > 2000:
            # Reset to best and reheat
            for i in range(_N):
                for j in range(_N):
                    adj[i][j] = best_adj[i][j]
            for i in range(_N):
                deg[i] = best_deg[i]
            T = 5.0
            stuck = 0
            best_stuck += 1
            if best_stuck > 20:
                break
    
    # Final polish: try all single swaps from best state
    if time.time() - t0 < _DEADLINE - 10:
        for i in range(_N):
            for j in range(_N):
                adj[i][j] = best_adj[i][j]
        for i in range(_N):
            deg[i] = best_deg[i]
        
        # Local search: try every possible swap
        for e1 in all_pairs:
            if not best_adj[e1[0]][e1[1]]:
                continue
            for e2 in all_pairs:
                if best_adj[e2[0]][e2[1]]:
                    continue
                # Apply
                u1, v1 = e1
                u2, v2 = e2
                best_adj[u1][v1] = 0
                best_adj[v1][u1] = 0
                best_adj[u2][v2] = 1
                best_adj[v2][u2] = 1
                best_deg[u1] -= 1
                best_deg[v1] -= 1
                best_deg[u2] += 1
                best_deg[v2] += 1
                
                s, _ = _score(best_adj, best_deg)
                
                if s < best_score:
                    best_score = s
                else:
                    # Undo
                    best_adj[u1][v1] = 1
                    best_adj[v1][u1] = 1
                    best_adj[u2][v2] = 0
                    best_adj[v2][u2] = 0
                    best_deg[u1] += 1
                    best_deg[v1] += 1
                    best_deg[u2] -= 1
                    best_deg[v2] -= 1
        
        # Second polish pass
        for e1 in all_pairs:
            if not best_adj[e1[0]][e1[1]]:
                continue
            for e2 in all_pairs:
                if best_adj[e2[0]][e2[1]]:
                    continue
                u1, v1 = e1
                u2, v2 = e2
                best_adj[u1][v1] = 0
                best_adj[v1][u1] = 0
                best_adj[u2][v2] = 1
                best_adj[v2][u2] = 1
                best_deg[u1] -= 1
                best_deg[v1] -= 1
                best_deg[u2] += 1
                best_deg[v2] += 1
                
                s, _ = _score(best_adj, best_deg)
                
                if s < best_score:
                    best_score = s
                else:
                    best_adj[u1][v1] = 1
                    best_adj[v1][u1] = 1
                    best_adj[u2][v2] = 0
                    best_adj[v2][u2] = 0
                    best_deg[u1] += 1
                    best_deg[v1] += 1
                    best_deg[u2] -= 1
                    best_deg[v2] -= 1
    
    # Build edge list from best_adj
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return {'n': _N, 'edges': edges}
# EVOLVE-BLOCK-END
