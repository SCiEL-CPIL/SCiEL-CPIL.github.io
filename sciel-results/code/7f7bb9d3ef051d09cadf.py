"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_signs = None
    best_radius = float('inf')
    
    # Build adjacency structure for fast connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    def is_connected(signs):
        """Check if the 2-lift is connected using BFS on the lift graph."""
        # Lift has 2n vertices: (v, 0) and (v, 1)
        visited = [False] * (2 * n)
        queue = deque()
        queue.append((0, 0))
        visited[0] = True
        count = 1
        
        while queue and count < 2 * n:
            v, side = queue.popleft()
            for (u, eidx) in adj[v]:
                # Determine the side of u in the lift
                if signs[eidx] == 1:
                    new_side = side
                else:
                    new_side = 1 - side
                idx = u * 2 + new_side
                if not visited[idx]:
                    visited[idx] = True
                    count += 1
                    queue.append((u, new_side))
        
        return count == 2 * n
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    
    def evaluate(signs):
        nonlocal best_signs, best_radius
        if not is_connected(signs):
            return None
        try:
            radius = spectral_radius(signs)
            # Verify with check_lift
            _, _ = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            return radius
        except Exception:
            return None
    
    def local_search(current, max_iters=500):
        nonlocal best_signs, best_radius
        cur = current.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Try single flips
            improved = False
            for i in range(num_edges):
                trial = cur.copy()
                trial[i] = -trial[i]
                trial_radius = evaluate(trial)
                if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                    cur = trial
                    cur_radius = trial_radius
                    improved = True
                    break
            
            # Try double flips if no single flip improves
            if not improved:
                for i in range(num_edges):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i+1, num_edges):
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        trial_radius = evaluate(trial)
                        if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = trial_radius
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Generate structured initial guesses
    def make_structured_signs(pattern_top, pattern_bottom, pattern_match):
        signs = np.ones(num_edges, dtype=int)
        edge_idx = 0
        for i in range(m):
            signs[edge_idx] = pattern_top[i % len(pattern_top)]
            edge_idx += 1
        for i in range(m):
            signs[edge_idx] = pattern_bottom[i % len(pattern_bottom)]
            edge_idx += 1
        for i in range(m):
            signs[edge_idx] = pattern_match[i % len(pattern_match)]
            edge_idx += 1
        return signs
    
    # Try various structured patterns
    patterns = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
        [-1, 1, -1, 1],
        [-1, -1, 1, 1],
    ]
    
    for pt in patterns:
        for pb in patterns:
            for pm in patterns:
                if time.time() > deadline - 10:
                    break
                signs = make_structured_signs(pt, pb, pm)
                evaluate(signs)
    
    # Random sampling
    rng = np.random.default_rng(42)
    
    num_random = 2000 if m <= 5 else 500
    for _ in range(num_random):
        if time.time() > deadline - 15:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
    
    # Local search from best
    if best_signs is not None:
        local_search(best_signs, max_iters=300)
    
    # More random restarts with local search
    num_restarts = 50 if m <= 5 else 20
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
        if best_signs is not None:
            local_search(best_signs, max_iters=100)
    
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
