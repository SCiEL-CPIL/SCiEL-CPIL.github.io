import time
import random
import sys
import numpy as np
from itertools import combinations

def construct():
    n = 16
    target = 65
    deadline = time.time() + 110.0
    rng = random.Random(42)

    # Precompute all triangles as (u,v,w) with u<v<w
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)

    # Precompute triangle indices as numpy arrays for fast access
    tri_u = np.array([t[0] for t in triangles], dtype=np.int8)
    tri_v = np.array([t[1] for t in triangles], dtype=np.int8)
    tri_w = np.array([t[2] for t in triangles], dtype=np.int8)

    def _score_from_adj_fast(adj_np, deg_np):
        """Fast score computation using vectorized triangle detection."""
        # adj_np is 16x16 uint8, deg_np is 16 int32
        # Triangle exists if adj[u,v] & adj[v,w] & adj[u,w]
        tri_exists = adj_np[tri_u, tri_v] & adj_np[tri_v, tri_w] & adj_np[tri_u, tri_w]
        if not np.any(tri_exists):
            return 0
        deg_sums = deg_np[tri_u] + deg_np[tri_v] + deg_np[tri_w]
        return int(np.max(deg_sums[tri_exists]))

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def _valid(edges, n, target):
        if len(edges) != target:
            return False
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True

    def _seed_edges(n):
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges

    best_edges = _seed_edges(n)
    best_score = 24  # Known score of seed

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    def local_search(adj_np, deg_np, score):
        """Improved local search with vectorized scoring."""
        try_improve(_edges_from_adj(adj_np.astype(bool)), score)
        
        max_iter = 500
        no_improve_count = 0
        max_no_improve = 30
        
        for it in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Find all triangles and their degree sums
            tri_exists = adj_np[tri_u, tri_v] & adj_np[tri_v, tri_w] & adj_np[tri_u, tri_w]
            if not np.any(tri_exists):
                break
            
            deg_sums = deg_np[tri_u] + deg_np[tri_v] + deg_np[tri_w]
            mask = tri_exists
            current_max = int(np.max(deg_sums[mask]))
            
            if current_max != score:
                score = current_max
                try_improve(_edges_from_adj(adj_np.astype(bool)), score)
            
            # Find worst triangle
            worst_idx = int(np.argmax(np.where(mask, deg_sums, -1)))
            u, v, w = int(tri_u[worst_idx]), int(tri_v[worst_idx]), int(tri_w[worst_idx])
            
            improved = False
            
            # Try swapping each edge of the worst triangle
            for a, b in [(u, v), (v, w), (u, w)]:
                # Find non-edges (c,d) not involving a or b
                # Prioritize non-edges where c and d have low degrees
                candidates = []
                for c in range(n):
                    if c == a or c == b:
                        continue
                    for d in range(c + 1, n):
                        if d == a or d == b:
                            continue
                        if not adj_np[c][d]:
                            # Score candidate by degree sum (lower is better)
                            candidates.append((deg_np[c] + deg_np[d], c, d))
                
                # Sort by degree sum, try lowest first
                candidates.sort()
                
                best_new_score = score
                best_swap = None
                
                for _, c, d in candidates[:15]:
                    # Simulate swap
                    new_adj = adj_np.copy()
                    new_deg = deg_np.copy()
                    
                    new_adj[a][b] = 0
                    new_adj[b][a] = 0
                    new_adj[c][d] = 1
                    new_adj[d][c] = 1
                    
                    new_deg[a] -= 1
                    new_deg[b] -= 1
                    new_deg[c] += 1
                    new_deg[d] += 1
                    
                    # Check if new edge creates bad triangles
                    ok = True
                    for x in range(n):
                        if x == c or x == d:
                            continue
                        if new_adj[c][x] and new_adj[d][x]:
                            s_new = new_deg[c] + new_deg[d] + new_deg[x]
                            if s_new > current_max:
                                ok = False
                                break
                    
                    if not ok:
                        continue
                    
                    # Compute full score
                    ns = _score_from_adj_fast(new_adj, new_deg)
                    
                    if ns < best_new_score:
                        best_new_score = ns
                        best_swap = (a, b, c, d, new_adj, new_deg)
                
                if best_swap:
                    a2, b2, c2, d2, new_adj, new_deg = best_swap
                    adj_np = new_adj
                    deg_np = new_deg
                    score = best_new_score
                    improved = True
                    try_improve(_edges_from_adj(adj_np.astype(bool)), score)
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
                
                # Try random edge swaps
                all_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj_np[u][v]]
                all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj_np[u][v]]
                
                if all_edges and all_non_edges:
                    a, b = rng.choice(all_edges)
                    c, d = rng.choice(all_non_edges)
                    
                    if a in (c, d) or b in (c, d):
                        continue
                    
                    new_adj = adj_np.copy()
                    new_deg = deg_np.copy()
                    
                    new_adj[a][b] = 0
                    new_adj[b][a] = 0
                    new_adj[c][d] = 1
                    new_adj[d][c] = 1
                    
                    new_deg[a] -= 1
                    new_deg[b] -= 1
                    new_deg[c] += 1
                    new_deg[d] += 1
                    
                    ns = _score_from_adj_fast(new_adj, new_deg)
                    
                    if ns < score:
                        adj_np = new_adj
                        deg_np = new_deg
                        score = ns
                        try_improve(_edges_from_adj(adj_np.astype(bool)), score)
                        no_improve_count = 0
        
        return adj_np, deg_np, score

    # --- Main search ---
    # Start with seed
    adj = np.zeros((n, n), dtype=np.uint8)
    deg = np.zeros(n, dtype=np.int32)
    for u, v in best_edges:
        adj[u][v] = adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    score = _score_from_adj_fast(adj, deg)
    adj, deg, score = local_search(adj, deg, score)

    # Try diverse starting points
    # Pattern: balanced degree sequences
    for _ in range(30):
        if time.time() >= deadline - 5:
            break
        
        # Random edge set with 65 edges
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        
        adj = np.zeros((n, n), dtype=np.uint8)
        deg = np.zeros(n, dtype=np.int32)
        for u, v in edges:
            adj[u][v] = adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
        
        score = _score_from_adj_fast(adj, deg)
        adj, deg, score = local_search(adj, deg, score)

    # Try structured starts: K_{8,8} + random intra edges
    for _ in range(20):
        if time.time() >= deadline - 5:
            break
        
        adj = np.zeros((n, n), dtype=np.uint8)
        deg = np.zeros(n, dtype=np.int32)
        
        # K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = 1
                deg[u] += 1
                deg[v] += 1
        
        # Add random intra edges to reach 65
        intra_pairs = [(u, v) for u in range(8) for v in range(u + 1, 8)] + \
                      [(u, v) for u in range(8, 16) for v in range(u + 1, 16)]
        rng.shuffle(intra_pairs)
        
        for u, v in intra_pairs:
            if sum(adj) >= 2 * target:
                break
            adj[u][v] = adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
        
        score = _score_from_adj_fast(adj, deg)
        adj, deg, score = local_search(adj, deg, score)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = 24

    return {'n': n, 'edges': best_edges}
