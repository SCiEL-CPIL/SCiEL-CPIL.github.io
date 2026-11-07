"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    start_time = time.time()
    # Use 55s deadline to be safe against overhead
    deadline = start_time + 55

    # Precompute edge endpoints for fast access
    eu = np.array([e[0] for e in edges], dtype=int)
    ev = np.array([e[1] for e in edges], dtype=int)

    # Classify edges: top cycle, bottom cycle, matching
    # This helps in generating structured patterns
    cycle1_idx = []
    cycle2_idx = []
    match_idx = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_idx.append(idx)
        elif u >= m and v >= m:
            cycle2_idx.append(idx)
        else:
            match_idx.append(idx)

    def fast_connected(signs):
        """Return True if the 2-lift is connected (BFS on 2n vertices)."""
        n_lift = 2 * n
        # Use a list of lists for adjacency, but optimize for small n
        adj = [[] for _ in range(n_lift)]
        for i in range(num_edges):
            sgn = signs[i]
            u, v = eu[i], ev[i]
            if sgn == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        visited = bytearray(n_lift)
        visited[0] = 1
        q = deque([0])
        cnt = 1
        while q:
            x = q.popleft()
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = 1
                    cnt += 1
                    q.append(y)
        return cnt == n_lift

    best_signs = None
    best_radius = float('inf')
    
    # Store top candidates to diversify local search starts
    candidates = []

    def evaluate(signs, update_best=True):
        nonlocal best_signs, best_radius
        if not fast_connected(signs):
            return None
        try:
            radius, details = check_lift(graph, signs)
            if update_best:
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
            # Keep top 10 candidates for diverse restarts
            if len(candidates) < 10 or radius < candidates[-1][0]:
                candidates.append((radius, signs.copy()))
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 10:
                    candidates.pop()
            return radius
        except Exception:
            return None

    def make_pattern(c1, c2, mt):
        signs = np.ones(num_edges, dtype=int)
        for k, idx in enumerate(cycle1_idx):
            signs[idx] = c1[k % len(c1)]
        for k, idx in enumerate(cycle2_idx):
            signs[idx] = c2[k % len(c2)]
        for k, idx in enumerate(match_idx):
            signs[idx] = mt[k % len(mt)]
        return signs

    def local_search_steepest(start_signs, max_iters=500):
        """
        Local search using steepest descent: find the best single flip in the neighborhood.
        If no single flip improves, try double flips.
        """
        nonlocal best_signs, best_radius
        cur = start_signs.copy()
        cur_radius = evaluate(cur, update_best=False)
        if cur_radius is None:
            return

        # Update global best if this start is good
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = cur.copy()

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            best_trial = None
            best_trial_radius = cur_radius

            # 1. Single flips (Steepest Descent)
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                trial = cur.copy()
                trial[i] = -trial[i]
                tr = evaluate(trial, update_best=False)
                if tr is not None and tr < best_trial_radius - 1e-12:
                    best_trial_radius = tr
                    best_trial = trial

            if best_trial is not None:
                cur = best_trial
                cur_radius = best_trial_radius
                improved = True
                if cur_radius < best_radius:
                    best_radius = cur_radius
                    best_signs = cur.copy()
                continue

            # 2. Double flips if no single flip worked
            # Limit double flips to save time, only check a subset or all if small
            # For m=9, E=27, E^2/2 ~ 351 checks. This is fast enough.
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline - 2:
                        break
                    trial = cur.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    tr = evaluate(trial, update_best=False)
                    if tr is not None and tr < best_trial_radius - 1e-12:
                        best_trial_radius = tr
                        best_trial = trial
                        break
                if best_trial is not None:
                    break
            
            if best_trial is not None:
                cur = best_trial
                cur_radius = best_trial_radius
                improved = True
                if cur_radius < best_radius:
                    best_radius = cur_radius
                    best_signs = cur.copy()
            else:
                # No improvement found in neighborhood, stop
                break

    # Phase 1: Structured patterns
    base_patterns = [
        [1],
        [-1],
        [1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
    ]
    for pt in base_patterns:
        for pb in base_patterns:
            for pm in base_patterns:
                if time.time() > deadline - 20:
                    break
                signs = make_pattern(pt, pb, pm)
                evaluate(signs)

    # Phase 2: Random sampling
    rng = np.random.default_rng(42)
    # Adjust number of random samples based on time remaining and graph size
    # Smaller graphs allow more samples
    num_random = 2000 if m <= 5 else 500
    for _ in range(num_random):
        if time.time() > deadline - 25:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)

    # Phase 3: Local search from top candidates
    # We have up to 10 candidates in `candidates` list
    # Sort them by radius (already sorted in evaluate, but re-sort to be sure)
    candidates.sort(key=lambda x: x[0])
    
    # Allocate time for local search
    # Use remaining time minus a small buffer
    # Run local search on the best candidate first, then others if time permits
    
    # Estimate time per local search iteration. 
    # For m=9, one iter is ~27 single + ~350 double checks = ~377 evals.
    # Each eval is fast_connected (O(V+E)) + check_lift (O(V^3) or similar).
    # check_lift is the bottleneck.
    
    # Let's run local search on the top 3 candidates if time allows.
    top_candidates = candidates[:3]
    
    for rank, (rad, sig) in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        # If the candidate is already the global best, we might have already searched it?
        # No, we haven't run local search yet.
        local_search_steepest(sig, max_iters=100)

    # Phase 4: Random restarts + local search if time remains
    # If we still have time, do random restarts
    while time.time() < deadline - 3:
        signs = rng.choice([-1, 1], size=num_edges)
        rad = evaluate(signs)
        if rad is not None:
            # Only do local search if this is a promising candidate
            # Or just do a short local search
            local_search_steepest(signs, max_iters=50)
            
    if best_signs is None:
        return np.ones(num_edges, dtype=int)

    return best_signs
# EVOLVE-BLOCK-END
