import time
import random
import sys
import numpy as np
from itertools import combinations

def construct():
    n = 16
    target = 65
    deadline = time.time() + 110.0  # safety margin under 150s wall-clock
    rng = random.Random(42)

    # Precompute all possible triangles (triples)
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)

    # Precompute all edges as (u, v) with u < v
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}

    # Precompute for each edge (u,v), the list of vertices w such that (u,v,w) is a triangle
    # i.e., w is connected to both u and v
    # Actually, we need: for a given edge (u,v), which other edges (u,w) and (v,w) complete the triangle
    # Let's precompute: for each edge index, the list of (other_edge_idx, vertex) that form triangles
    # This is complex; instead, we'll use a fast scoring function

    def _score_from_adj(adj, deg):
        """Compute max triangle degree sum. adj is 16x16 bool, deg is list of 16 ints."""
        best = 0
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best

    def _score_from_edges(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return _score_from_adj(adj, deg)

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
        # K_{8,8} + one intra edge -> 65 edges, score 24
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges

    best_edges = _seed_edges(n)
    best_score = _score_from_edges(best_edges)

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # --- Fast local search using numpy for scoring ---
    def local_search_fast(adj, deg, score):
        """Local search using edge swaps. Returns improved adj, deg, score."""
        try_improve(_edges_from_adj(adj), score)
        
        # Convert to numpy for faster operations
        adj_np = np.array(adj, dtype=np.uint8)
        deg_np = np.array(deg, dtype=np.int32)
        
        max_iter = 300
        no_improve_count = 0
        max_no_improve = 20
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Find worst triangle
            worst_tri = None
            worst_s = score
            # Use numpy to find triangles
            # A triangle (u,v,w) exists if adj[u,v] & adj[v,w] & adj[u,w]
            # We'll iterate through precomputed triangles
            for (u, v, w) in triangles:
                if adj_np[u][v] and adj_np[v][w] and adj_np[u][w]:
                    s = deg_np[u] + deg_np[v] + deg_np[w]
                    if s > worst_s:
                        worst_s = s
                        worst_tri = (u, v, w)
            
            if worst_tri is None:
                break
            
            u, v, w = worst_tri
            improved = False
            
            # Try to break the worst triangle by swapping one of its edges
            for a, b in [(u, v), (v, w), (u, w)]:
                # Find candidates: non-edges (c,d) where c,d not in {a,b}
                # and (c,d) is not already an edge
                best_new_score = score
                best_swap = None
                
                # Generate candidate non-edges
                candidates = []
                for c in range(n):
                    if c == a or c == b:
                        continue
                    for d in range(c + 1, n):
                        if d == a or d == b:
                            continue
                        if not adj_np[c][d]:
                            candidates.append((c, d))
                
                # Shuffle and try a subset
                rng.shuffle(candidates)
                for c, d in candidates[:10]:
                    # Simulate swap: remove (a,b), add (c,d)
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
                    
                    # Check if this creates any triangle with sum > worst_s
                    # Only check triangles involving the new edge (c,d)
                    ok = True
                    for x in range(n):
                        if x == c or x == d:
                            continue
                        if new_adj[c][x] and new_adj[d][x]:
                            s_new = new_deg[c] + new_deg[d] + new_deg[x]
                            if s_new > worst_s:
                                ok = False
                                break
                    
                    if not ok:
                        continue
                    
                    # Compute full score
                    ns = 0
                    for (tu, tv, tw) in triangles:
                        if new_adj[tu][tv] and new_adj[tv][tw] and new_adj[tu][tw]:
                            s = new_deg[tu] + new_deg[tv] + new_deg[tw]
                            if s > ns:
                                ns = s
                    
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
                    
                    ns = 0
                    for (tu, tv, tw) in triangles:
                        if new_adj[tu][tv] and new_adj[tv][tw] and new_adj[tu][tw]:
                            s = new_deg[tu] + new_deg[tv] + new_deg[tw]
                            if s > ns:
                                ns = s
                    
                    if ns < score:
                        adj_np = new_adj
                        deg_np = new_deg
                        score = ns
                        try_improve(_edges_from_adj(adj_np.astype(bool)), score)
                        no_improve_count = 0
        
        # Convert back to Python lists
        adj_final = adj_np.astype(bool)
        deg_final = deg_np.tolist()
        return adj_final, deg_final, score

    # --- Erdős–Gallai check ---
    def is_graphical(deg_seq):
        d = sorted(deg_seq, reverse=True)
        if sum(d) != 2 * target:
            return False
        if any(x < 0 or x > n - 1 for x in d):
            return False
        for k in range(1, n + 1):
            lhs = sum(d[:k])
            rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
            if lhs > rhs:
                return False
        return True

    # --- Main search loop ---
    start = time.time()

    # First, try the seed
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    for u, v in best_edges:
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    score = _score_from_adj(adj, deg)
    adj, deg, score = local_search_fast(adj, deg, score)

    # Generate diverse degree sequences to try
    # Start with balanced sequences around average degree 130/16 ≈ 8.125
    avg_deg = 2 * target / n  # 8.125

    # Try various degree distributions
    deg_patterns = []

    # Pattern 1: all 8s and 9s
    # sum = 8*a + 9*b = 130, a+b=16 -> 8a+9b=130, a=16-b -> 8(16-b)+9b=130 -> 128+b=130 -> b=2
    # So 14 vertices of degree 8, 2 vertices of degree 9
    base = [8] * 16
    for i in range(2):
        base[i] = 9
    deg_patterns.append(base[:])

    # Pattern 2: 15 of 8, 1 of 10
    deg_patterns.append([8] * 15 + [10])

    # Pattern 3: 2 of 7, 12 of 8, 2 of 10
    deg_patterns.append([7] * 2 + [8] * 12 + [10] * 2)

    # Pattern 4: 1 of 7, 14 of 8, 1 of 11
    deg_patterns.append([7] + [8] * 14 + [11])

    # Add some random patterns
    for _ in range(20):
        # Random degree sequence summing to 130
        deg_seq = [8] * 16
        remaining = 130 - sum(deg_seq)
        while remaining != 0:
            i = rng.randint(0, 15)
            if remaining > 0 and deg_seq[i] < 15:
                deg_seq[i] += 1
                remaining -= 1
            elif remaining < 0 and deg_seq[i] > 0:
                deg_seq[i] -= 1
                remaining += 1
            elif remaining > 0:
                break
            elif remaining < 0:
                break
        if sum(deg_seq) == 130 and is_graphical(deg_seq):
            deg_patterns.append(deg_seq)

    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in deg_patterns:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    # Try each pattern
    for deg_seq in unique_patterns:
        if time.time() >= deadline - 5:
            break
        if not is_graphical(deg_seq):
            continue

        # Try multiple realizations
        for attempt in range(3):
            if time.time() >= deadline - 5:
                break
            # Simple random realization: try to build graph with given degrees
            # This is complex, so we'll use a heuristic
            adj = [[False] * n for _ in range(n)]
            deg_current = [0] * n
            remaining = deg_seq[:]
            
            # Sort vertices by remaining degree
            for _ in range(100):  # Max iterations
                if time.time() >= deadline - 5:
                    break
                # Find vertex with max remaining degree
                max_idx = max(range(n), key=lambda i: remaining[i])
                if remaining[max_idx] <= 0:
                    break
                
                # Connect to vertices with positive remaining degree
                candidates = [i for i in range(n) if i != max_idx and remaining[i] > 0 and not adj[max_idx][i]]
                if len(candidates) < remaining[max_idx]:
                    break
                
                # Choose randomly
                chosen = rng.sample(candidates, remaining[max_idx])
                for v in chosen:
                    adj[max_idx][v] = adj[v][max_idx] = True
                    deg_current[max_idx] += 1
                    deg_current[v] += 1
                    remaining[max_idx] -= 1
                    remaining[v] -= 1
            
            if sum(remaining) == 0 and len(_edges_from_adj(adj)) == target:
                score = _score_from_adj(adj, deg_current)
                adj, deg, score = local_search_fast(adj, deg, score)

    # Random restarts with random edge sets
    while time.time() < deadline - 5:
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        if not _valid(edges, n, target):
            continue
        
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        score = _score_from_adj(adj, deg)
        adj, deg, score = local_search_fast(adj, deg, score)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = _score_from_edges(best_edges)

    return {'n': n, 'edges': best_edges}
