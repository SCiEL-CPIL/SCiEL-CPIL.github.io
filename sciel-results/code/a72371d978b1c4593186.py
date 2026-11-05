"""Polarity graph C4-free core + optimized extra edge + local search."""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate all 31 points in PG(2,5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)
                        vec = [(v * inv) % 5 for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build polarity graph: connect points if they are orthogonal
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    base_edges = list(edges)
    base_adj = adj.copy()
    
    def count_c4s(adj_matrix):
        """Count 4-cycles using A^2 matrix."""
        A2 = adj_matrix.astype(np.int32) @ adj_matrix.astype(np.int32)
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                common = A2[i, j]
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 4
    
    base_c4 = count_c4s(base_adj)
    
    # Find all non-edges in the base graph
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if base_adj[i, j] == 0:
                non_edges.append([i, j])
    
    def count_new_c4s(adj_matrix, u, v):
        """Count new C4s created by adding edge (u,v)."""
        count = 0
        for x in range(n):
            if x == u or x == v:
                continue
            if adj_matrix[v, x] == 1:
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if adj_matrix[u, y] == 1 and adj_matrix[x, y] == 1:
                        count += 1
        return count
    
    # Find the edge that creates the fewest new C4s
    best_edge = None
    best_new_c4s = float('inf')
    
    for edge in non_edges:
        u, v = edge
        new_c4s = count_new_c4s(base_adj, u, v)
        if new_c4s < best_new_c4s:
            best_new_c4s = new_c4s
            best_edge = edge
    
    # Add the best edge
    final_edges = base_edges + [best_edge]
    final_adj = base_adj.copy()
    u, v = best_edge
    final_adj[u, v] = 1
    final_adj[v, u] = 1
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    edge_set = set()
    for e in final_edges:
        key = tuple(sorted(e))
        assert key not in edge_set, f"Duplicate edge {e}"
        edge_set.add(key)
    
    for e in final_edges:
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30
    
    total_c4 = count_c4s(final_adj)
    
    # Local search: try to find a better configuration
    if time.time() > deadline - 25:
        return final_edges
    
    current_adj = final_adj.copy()
    current_edges = final_edges[:]
    current_c4 = total_c4
    
    # Precompute all possible non-edges for faster sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    current_edge_set = set(tuple(sorted(e)) for e in current_edges)
    
    import random
    rng = random.Random(42)
    
    # Vectorized delta computation for speed
    def compute_delta(adj_matrix, remove_u, remove_v, add_u, add_v):
        """Compute change in C4 count when removing edge (remove_u, remove_v) and adding edge (add_u, add_v)."""
        # C4s removed by removing edge (remove_u, remove_v)
        removed = 0
        for x in range(n):
            if x == remove_u or x == remove_v:
                continue
            if adj_matrix[remove_v, x] == 1:
                for y in range(n):
                    if y == remove_u or y == remove_v or y == x:
                        continue
                    if adj_matrix[remove_u, y] == 1 and adj_matrix[x, y] == 1:
                        removed += 1
        
        # C4s added by adding edge (add_u, add_v)
        added = 0
        for x in range(n):
            if x == add_u or x == add_v:
                continue
            if adj_matrix[add_v, x] == 1:
                for y in range(n):
                    if y == add_u or y == add_v or y == x:
                        continue
                    if adj_matrix[add_u, y] == 1 and adj_matrix[x, y] == 1:
                        added += 1
        
        return added - removed
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try multiple random swaps
        for attempt in range(30):
            # Pick a random edge to remove
            idx = rng.randint(0, len(current_edges) - 1)
            e1 = current_edges[idx]
            u1, v1 = e1
            
            # Find a random non-edge to add
            found = False
            for _ in range(50):
                u2, v2 = rng.choice(all_pairs)
                if (u2, v2) in current_edge_set:
                    continue
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                
                delta = compute_delta(current_adj, u1, v1, u2, v2)
                if delta < 0:
                    # Accept the swap
                    current_adj[u1, v1] = 0
                    current_adj[v1, u1] = 0
                    current_adj[u2, v2] = 1
                    current_adj[v2, u2] = 1
                    current_edges[idx] = [min(u2, v2), max(u2, v2)]
                    current_edge_set.remove((u1, v1))
                    current_edge_set.add((u2, v2))
                    current_c4 += delta
                    improved = True
                    found = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Build final result
    final_result = []
    for i in range(n):
        for j in range(i+1, n):
            if current_adj[i, j] == 1:
                final_result.append([i, j])
    
    assert len(final_result) == 91, f"Expected 91 edges, got {len(final_result)}"
    
    return final_result
