"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    start_time = time.time()
    DEADLINE = 115.0
    n = 31
    target_edges = 91
    
    # Polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                edges.add((i, j))
    
    edges = list(edges)
    
    # Build adjacency matrix for fast common-neighbor counting
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def count_c4(adj):
        """Count 4-cycles using matrix multiplication."""
        # Number of C4 = (sum of all entries of A^2)^2 - ... 
        # Actually: each C4 is counted 8 times in (A^2 * A^2) diagonal sum
        A2 = adj @ adj
        # Sum of squares of A2 entries / 8 gives C4 count
        # Each 4-cycle contributes to 8 pairs of opposite vertices
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                total += A2[i, j] * A2[i, j]
        return total // 8
    
    if len(edges) < target_edges:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = set(edges)
        candidates = [p for p in all_pairs if p not in existing]
        edge_set = set(edges)
        
        for _ in range(target_edges - len(edges)):
            if time.time() - start_time > DEADLINE - 10:
                break
            best, best_cost = None, float('inf')
            for u, v in candidates:
                common = int(np.sum(adj[u] & adj[v]))
                if common < best_cost:
                    best_cost, best = common, (u, v)
            if best:
                edges.append(best)
                edge_set.add(best)
                candidates.remove(best)
                adj[best[0], best[1]] = 1
                adj[best[1], best[0]] = 1
    
    elif len(edges) > target_edges:
        # Remove edges that contribute most to C4
        for _ in range(len(edges) - target_edges):
            if time.time() - start_time > DEADLINE - 10:
                break
            best, best_cost = None, float('inf')
            for u, v in edges:
                common = int(np.sum(adj[u] & adj[v]))
                if common > best_cost:
                    best_cost, best = common, (u, v)
            if best:
                edges.remove(best)
                adj[best[0], best[1]] = 0
                adj[best[1], best[0]] = 0
    
    # Local search: try swapping edges to reduce C4 count
    current_c4 = count_c4(adj)
    edge_set = set(edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [p for p in all_pairs if p not in edge_set]
    
    improved = True
    while improved and time.time() - start_time < DEADLINE - 15:
        improved = False
        best_swap = None
        best_new_c4 = current_c4
        
        # Sample a subset of swaps for efficiency
        import random
        random.seed(42)
        sample_edges = random.sample(edges, min(len(edges), 30))
        sample_non_edges = random.sample(non_edges, min(len(non_edges), 30))
        
        for e1 in sample_edges:
            if time.time() - start_time > DEADLINE - 15:
                break
            u1, v1 = e1
            for e2 in sample_non_edges:
                u2, v2 = e2
                # Check if swap is valid (no duplicate edges)
                if (u1, v1) in edge_set and (u2, v2) not in edge_set:
                    # Temporarily swap
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_c4 = count_c4(adj)
                    
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = (e1, e2)
        
        if best_swap:
            e1, e2 = best_swap
            # Apply swap
            adj[e1[0], e1[1]] = 0
            adj[e1[1], e1[0]] = 0
            adj[e2[0], e2[1]] = 1
            adj[e2[1], e2[0]] = 1
            
            edges.remove(e1)
            edges.append(e2)
            edge_set.remove(e1)
            edge_set.add(e2)
            
            non_edges.remove(e2)
            non_edges.append(e1)
            
            current_c4 = best_new_c4
            improved = True
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
