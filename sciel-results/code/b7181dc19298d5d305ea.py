"""Combined approach: hill-climbing + simulated annealing for triangle degree sums."""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE_MARGIN = 10  # seconds before hard kill (150s total - 140s safety)


def _edges_to_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _degrees(adj):
    return [len(s) for s in adj]


def _triangles(adj):
    tris = []
    for u in range(N):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        tris.append((u, v, w))
    return tris


def _score_graph(adj, tris, deg):
    if not tris:
        return 999
    return max(deg[u] + deg[v] + deg[w] for u, v, w in tris)


def _initial_edges():
    # K_{8,8} plus one intra-part edge
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges


def _all_pairs():
    return [(u, v) for u in range(N) for v in range(u + 1, N)]


def _edge_index(edges):
    m = {}
    for i, (u, v) in enumerate(edges):
        a, b = (u, v) if u < v else (v, u)
        m[(a, b)] = i
    return m


def _quick_triangle_check(adj, deg, u, v):
    """Quick check: if adding edge (u,v) creates triangles, what's the max degree sum?"""
    common = adj[u] & adj[v]
    if not common:
        return 0, []
    max_sum = 0
    new_tris = []
    for w in common:
        s = (deg[u] + 1) + (deg[v] + 1) + deg[w]
        if s > max_sum:
            max_sum = s
        new_tris.append((min(u,v,w), min(max(u,v),w), max(u,v,w)))
    return max_sum, new_tris


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with the seed
    edges = _initial_edges()
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    best_edges = [e[:] for e in edges]
    best_score = current_score
    
    all_pairs = _all_pairs()
    
    # Simulated annealing parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # Phase 1: Simulated annealing
    while time.time() < deadline and T > T_min:
        # Try a random swap
        u, v = random.choice(all_pairs)
        if (min(u,v), max(u,v)) in ei:
            T *= cooling
            continue
        
        # Pick a random edge to remove
        idx = random.randint(0, len(edges) - 1)
        rem_u, rem_v = edges[idx]
        
        a, b = (u, v) if u < v else (v, u)
        ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
        
        # Perform swap
        adj[u].add(v)
        adj[v].add(u)
        adj[rem_u].remove(rem_v)
        adj[rem_v].remove(rem_u)
        
        deg[u] += 1
        deg[v] += 1
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        
        edges[idx] = [u, v]
        del ei[(ra, rb)]
        ei[(a, b)] = idx
        
        # Recompute score
        tris_new = _triangles(adj)
        score_new = _score_graph(adj, tris_new, deg)
        
        delta = score_new - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_score = score_new
            tris = tris_new
            if current_score < best_score:
                best_score = current_score
                best_edges = [e[:] for e in edges]
        else:
            # Revert
            adj[u].remove(v)
            adj[v].remove(u)
            adj[rem_u].add(rem_v)
            adj[rem_v].add(rem_u)
            
            deg[u] -= 1
            deg[v] -= 1
            deg[rem_u] += 1
            deg[rem_v] += 1
            
            edges[idx] = [rem_u, rem_v]
            del ei[(a, b)]
            ei[(ra, rb)] = idx
        
        T *= cooling
    
    # Phase 2: Hill climbing from best found
    edges = best_edges[:]
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(all_pairs)
        for pair in all_pairs:
            if time.time() > deadline:
                break
            u, v = pair
            if (min(u,v), max(u,v)) in ei:
                continue
            
            # Try removing each existing edge
            for idx in range(len(edges)):
                if time.time() > deadline:
                    break
                rem_u, rem_v = edges[idx]
                a, b = (u, v) if u < v else (v, u)
                ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
                
                # Quick check: will adding (u,v) create bad triangles?
                max_new, _ = _quick_triangle_check(adj, deg, u, v)
                if max_new > current_score:
                    continue
                
                # Perform swap
                adj[u].add(v)
                adj[v].add(u)
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                
                deg[u] += 1
                deg[v] += 1
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                edges[idx] = [u, v]
                del ei[(ra, rb)]
                ei[(a, b)] = idx
                
                tris_new = _triangles(adj)
                score_new = _score_graph(adj, tris_new, deg)
                
                if score_new < current_score:
                    current_score = score_new
                    tris = tris_new
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [e[:] for e in edges]
                    improved = True
                    break  # Restart pair loop
                else:
                    # Revert
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[rem_u].add(rem_v)
                    adj[rem_v].add(rem_u)
                    
                    deg[u] -= 1
                    deg[v] -= 1
                    deg[rem_u] += 1
                    deg[rem_v] += 1
                    
                    edges[idx] = [rem_u, rem_v]
                    del ei[(a, b)]
                    ei[(ra, rb)] = idx
    
    # Phase 3: Final polish - try all single-edge swaps
    edges = best_edges[:]
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    for pair in all_pairs:
        if time.time() > deadline:
            break
        u, v = pair
        if (min(u,v), max(u,v)) in ei:
            continue
        
        for idx in range(len(edges)):
            if time.time() > deadline:
                break
            rem_u, rem_v = edges[idx]
            a, b = (u, v) if u < v else (v, u)
            ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
            
            adj[u].add(v)
            adj[v].add(u)
            adj[rem_u].remove(rem_v)
            adj[rem_v].remove(rem_u)
            
            deg[u] += 1
            deg[v] += 1
            deg[rem_u] -= 1
            deg[rem_v] -= 1
            
            edges[idx] = [u, v]
            del ei[(ra, rb)]
            ei[(a, b)] = idx
            
            tris_new = _triangles(adj)
            score_new = _score_graph(adj, tris_new, deg)
            
            if score_new < current_score:
                current_score = score_new
                tris = tris_new
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in edges]
            else:
                # Revert
                adj[u].remove(v)
                adj[v].remove(u)
                adj[rem_u].add(rem_v)
                adj[rem_v].add(rem_u)
                
                deg[u] -= 1
                deg[v] -= 1
                deg[rem_u] += 1
                deg[rem_v] += 1
                
                edges[idx] = [rem_u, rem_v]
                del ei[(a, b)]
                ei[(ra, rb)] = idx
    
    # Verify edge count
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    if len(edge_set) != TARGET_EDGES:
        # Fall back to baseline
        best_edges = _initial_edges()
    
    return {'n': N, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    adj = _edges_to_adj(res['edges'])
    deg = _degrees(adj)
    tris = _triangles(adj)
    print("Score:", _score_graph(adj, tris, deg))
    print("Edges:", len(res['edges']))
