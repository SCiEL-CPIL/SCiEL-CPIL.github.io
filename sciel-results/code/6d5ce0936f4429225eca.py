"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        if not edges:
            return float('inf')
        
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            if u == v:
                continue
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        
        # Find all triangles
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    # Start with the baseline construction
    base_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    base_score = compute_score(base_edges)
    
    best_edges = base_edges[:]
    best_score = base_score
    
    # Local search: try to improve by swapping edges
    def local_search(edges, max_iters=10000):
        """Try to improve score via edge swaps."""
        current = edges[:]
        current_score = compute_score(current)
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in current:
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try all possible edge swaps: remove one edge, add another
            # We'll sample a subset for speed
            n_edges = len(current)
            for i in range(n_edges):
                if time.time() > deadline - 5.0:
                    break
                u1, v1 = current[i]
                # Try to replace edge (u1,v1) with some other non-edge
                # Sample a few candidates
                for _ in range(5):  # Try 5 random non-edges per edge removal
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if adj[u2][v2]:
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    
                    # Create new edge list
                    new_edges = current[:i] + current[i+1:] + [[u2, v2]]
                    new_score = compute_score(new_edges)
                    
                    if new_score < current_score:
                        current = new_edges
                        current_score = new_score
                        improved = True
                        # Rebuild adj and deg
                        adj = [[False] * n for _ in range(n)]
                        deg = [0] * n
                        for u, v in current:
                            adj[u][v] = True
                            adj[v][u] = True
                            deg[u] += 1
                            deg[v] += 1
                        break  # Restart from this new state
        
        return current, current_score
    
    # Run local search with multiple restarts
    best_score_local = best_score
    best_edges_local = best_edges[:]
    
    # Try several random perturbations of the base
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Start from base with small random perturbation
        perturbed = base_edges[:]
        # Remove 2-3 random edges and add 2-3 new random edges
        num_swap = random.randint(2, 4)
        for _ in range(num_swap):
            idx = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[idx]
            # Find a random non-edge
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                # Check if edge exists
                exists = False
                for a, b in perturbed:
                    if (a == u2 and b == v2) or (a == v2 and b == u2):
                        exists = True
                        break
                if not exists:
                    break
            perturbed[idx] = [u2, v2]
        
        if not is_valid(perturbed):
            continue
        
        pert_score = compute_score(perturbed)
        
        # Local search from perturbation
        local_edges, local_score = local_search(perturbed, max_iters=5000)
        
        if local_score < best_score_local:
            best_score_local = local_score
            best_edges_local = local_edges[:]
        
        # Also check if perturbation itself is better
        if pert_score < best_score_local:
            best_score_local = pert_score
            best_edges_local = perturbed[:]
    
    # Also run local search from the base
    base_local_edges, base_local_score = local_search(base_edges, max_iters=10000)
    if base_local_score < best_score_local:
        best_score_local = base_local_score
        best_edges_local = base_local_edges[:]
    
    return {'n': n, 'edges': best_edges_local}
# EVOLVE-BLOCK-END
