import numpy as np
import time
import random

def construct():
    n = 40
    # Safety margin: evaluator kills at 150s, we aim for 115s
    start_time = time.time()
    deadline = start_time + 115.0

    # Initialize adjacency matrix with 8 disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1

    m = 80
    best_m = m
    best_A = A.copy()

    # Precompute indices for upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    total_non_edges = len(i_idx)

    def compute_P5(mat):
        # A^5 = A^4 * A = (A^2)^2 * A
        A2 = mat @ mat
        A4 = A2 @ A2
        A5 = A4 @ mat
        return A5

    P5 = compute_P5(A)

    def get_valid_non_edges(adj, p5):
        # Check non-edges in upper triangle
        # adj[i_idx, j_idx] == 0 AND p5[i_idx, j_idx] == 0
        valid_mask = (adj[i_idx, j_idx] == 0) & (p5[i_idx, j_idx] == 0)
        valid_indices = np.where(valid_mask)[0]
        # Convert to list of tuples
        return [(int(i_idx[k]), int(j_idx[k])) for k in valid_indices]

    def get_all_edges(adj):
        edge_mask = adj[i_idx, j_idx] == 1
        edge_indices = np.where(edge_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in edge_indices]

    def add_edge(adj, p5, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        # Update P5: P5' = P5 + (A^4 * e + e * A^4) * A + e * A^5 + A^5 * e
        # Wait, A' = A + E where E has 1s at u,v and v,u
        # A'^5 = (A+E)^5. This is complex to update incrementally.
        # Given n=40, full recompute is fast enough (~1-2ms).
        # Let's just recompute.
        return compute_P5(adj)

    def remove_edge(adj, p5, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        return compute_P5(adj)

    # ILS Loop
    current_T = 0.5
    T_min = 0.01
    cooling = 0.995

    while time.time() < deadline:
        # Phase 1: Perturbation (Kick)
        # Randomly remove k edges
        # k is chosen based on current T to allow larger kicks when T is high
        # But we need to be careful not to degrade too much.
        # Let's try removing 1-5 edges.
        current_edges = get_all_edges(A)
        if not current_edges:
            break

        num_removals = random.randint(1, 5)
        if len(current_edges) < num_removals:
            num_removals = len(current_edges)
        
        removed = random.sample(current_edges, num_removals)
        for u, v in removed:
            A[u, v] = 0
            A[v, u] = 0
            m -= 1
        P5 = compute_P5(A)

        # Phase 2: Local Search (Greedy Add)
        # Try to add valid edges until no more can be added
        # We use a loop because adding one edge might make another valid (unlikely but possible? 
        # Actually, adding an edge can only create more paths, so it can only make MORE edges invalid.
        # So a single pass is sufficient if we pick from the current set of valid edges.
        # However, if we pick randomly, we might miss some.
        # Better: Repeatedly find ALL valid edges, pick one, add it, recompute.
        # Since adding edges reduces the set of valid edges, this terminates.
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            valid_edges = get_valid_non_edges(A, P5)
            if not valid_edges:
                improved = False
                break
            
            # Pick a random valid edge to add
            # Or pick the one that is "most central"? Random is fine for diversity.
            u, v = valid_edges[random.randint(0, len(valid_edges) - 1)]
            A[u, v] = 1
            A[v, u] = 1
            m += 1
            P5 = compute_P5(A)

        # Phase 3: Acceptance
        if m > best_m:
            best_m = m
            best_A = A.copy()
            # Reset temperature or keep it? Keep it.
        else:
            # Simulated Annealing Acceptance
            delta = m - best_m # negative
            if delta < 0:
                prob = np.exp(delta / current_T)
                if random.random() < prob:
                    # Accept, but don't update best
                    pass
                else:
                    # Reject: revert to best
                    A = best_A.copy()
                    m = best_m
                    P5 = compute_P5(A)
            
        # Cooling
        current_T *= cooling
        if current_T < T_min:
            # Reheat
            current_T = 0.5
            # Optional: perturb best solution again?
            # The loop will perturb A (which is currently best or a neighbor)
            # If we rejected, A is best. If we accepted, A is worse than best.
            # If we reheat, we should probably start from best to avoid drifting too far?
            # Or keep current A to continue exploration?
            # Standard SA keeps current. ILS usually restarts from best.
            # Let's restart from best to ensure we don't get stuck in a deep hole.
            A = best_A.copy()
            m = best_m
            P5 = compute_P5(A)

    # Convert best_A to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                result.append([u, v])
                
    return result
