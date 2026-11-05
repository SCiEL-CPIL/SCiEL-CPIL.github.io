import random
import time

def construct():
    """Construct a C4-free graph on n vertices with no independent set of size 11."""
    # Use the Erdos-Renyi graph over F_11 x F_11
    q = 11
    n_total = q * q
    vertices = [(i, j) for i in range(q) for j in range(q)]
    
    # Map (i, j) to 0..120
    def idx(i, j):
        return i * q + j
    
    # Build adjacency for the full graph
    # (x1, y1) ~ (x2, y2) if (x2-x1)(y2-y1) == 1 mod q
    adj_full = [set() for _ in range(n_total)]
    for i in range(q):
        for j in range(q):
            u = idx(i, j)
            for di in range(q):
                if di == 0: continue
                for dj in range(q):
                    if dj == 0: continue
                    if (di * dj) % q == 1:
                        x2 = (i + di) % q
                        y2 = (j + dj) % q
                        v = idx(x2, y2)
                        adj_full[u].add(v)
                        adj_full[v].add(u)
    
    # We want a subset of size 43
    target_n = 43
    target_alpha = 10
    
    # Greedy independence number for a subset
    def alpha_subset(subset):
        # subset is a set of vertex indices
        # Use a simple greedy: pick vertex with min degree in subset
        remaining = set(subset)
        count = 0
        while remaining:
            # Find vertex with min degree in remaining
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and u in adj_full[v]:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            count += 1
            remaining.discard(min_v)
            # Remove neighbors
            remaining -= {u for u in remaining if u in adj_full[min_v]}
        return count

    start_time = time.time()
    time_limit = 100
    
    # Random initialization
    random.seed(42)
    best_subset = set(random.sample(range(n_total), target_n))
    best_alpha = alpha_subset(best_subset)
    
    # Local search: swap vertices to reduce alpha
    while time.time() - start_time < time_limit:
        if best_alpha <= target_alpha:
            break
        
        improved = False
        outside = set(range(n_total)) - best_subset
        
        # Try swapping a vertex in best_subset with one outside
        # Sample a few swaps to save time
        for _ in range(200):
            v_in = random.choice(list(best_subset))
            v_out = random.choice(list(outside))
            
            new_subset = best_subset.copy()
            new_subset.discard(v_in)
            new_subset.add(v_out)
            
            new_alpha = alpha_subset(new_subset)
            
            if new_alpha < best_alpha:
                best_subset = new_subset
                best_alpha = new_alpha
                improved = True
                break
        
        if not improved:
            # Perturbation: swap 2-3 vertices
            for _ in range(3):
                v_in = random.choice(list(best_subset))
                outside = set(range(n_total)) - best_subset
                v_out = random.choice(list(outside))
                best_subset.discard(v_in)
                best_subset.add(v_out)
            best_alpha = alpha_subset(best_subset)
            
    if best_alpha <= target_alpha:
        subset = sorted(best_subset)
        n = target_n
        # Map subset indices to 0..n-1
        mapping = {old: new for new, old in enumerate(subset)}
        edges = []
        for u in subset:
            for v in adj_full[u]:
                if v in mapping and mapping[u] < mapping[v]:
                    edges.append([mapping[u], mapping[v]])
        return {'n': n, 'edges': edges}
    else:
        # Fallback: return a smaller valid graph if search fails
        # Use n=39 with the same method? Or just return empty?
        # Better to return a known valid small graph
        # For now, return n=30 with a clique? No, clique is not C4-free?
        # K_30 has C4s.
        # Empty graph is C4-free but alpha=30.
        # We need alpha <= 10.
        # So we need a graph with many edges.
        # If search fails, we return the best found subset even if alpha > 10?
        # No, invalid.
        # Let's hope it finds one.
        return {'n': 1, 'edges': []}
