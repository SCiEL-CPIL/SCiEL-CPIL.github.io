"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def simulated_annealing(labels, T_initial, T_final, steps, cooling_factor):
        """
        Simulated annealing for Max-Cut.
        labels: int array of 0/1
        T_initial: initial temperature
        T_final: final temperature
        steps: number of SA steps
        cooling_factor: not used directly, we use geometric cooling
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        # Precompute degrees for potential speedup? No, we need incremental updates.
        # For SA, we pick a random vertex and flip it.
        # Delta in cut = 2 * (sum of edges to opposite side) - 2 * (sum of edges to same side)
        # Wait, if we flip v:
        # Edges from v to same side become crossing (+1 each)
        # Edges from v to opposite side become non-crossing (-1 each)
        # So delta = 2 * (deg_same - deg_opp)
        
        # We maintain an array of 'sum_same' and 'sum_opp' for each vertex.
        sum_same = np.zeros(n, dtype=np.int32)
        sum_opp = np.zeros(n, dtype=np.int32)
        
        # Initialize sum_same and sum_opp
        for v in range(n):
            same_mask = labels == labels[v]
            opp_mask = labels != labels[v]
            sum_same[v] = int(adj[v, same_mask].sum())
            sum_opp[v] = int(adj[v, opp_mask].sum())
        
        T = T_initial
        # Geometric cooling: T_{k+1} = T_k * cooling_factor
        # We want to reach T_final after 'steps' iterations.
        # T_final = T_initial * cooling_factor^steps
        # cooling_factor = (T_final / T_initial)^(1/steps)
        if T_initial > 0 and T_final > 0:
            cf = (T_final / T_initial) ** (1.0 / steps)
        else:
            cf = 0.999

        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick random vertex
            v = random.randint(0, n - 1)
            
            # Calculate delta
            # If v flips:
            # Its 'same' neighbors become 'opp', so they lose 1 from sum_same, gain 1 from sum_opp? No.
            # Let's trace carefully.
            # Current state:
            # v is in group G.
            # Neighbors in G: contribute to sum_same[v].
            # Neighbors in !G: contribute to sum_opp[v].
            
            # After flip:
            # v is in !G.
            # Neighbors in G (now !G for v? No, they are in G, v is in !G).
            # So neighbors in G become 'opp' for v.
            # Neighbors in !G become 'same' for v.
            
            # Delta in cut for v:
            # Old contribution to cut from v's edges: sum_opp[v] (edges to opp side cross)
            # New contribution to cut from v's edges: sum_same[v] (edges to same side now cross)
            # Wait, the cut size is half the sum of crossing edges.
            # Let's just compute the change in the total cut size.
            # Let C be the current cut size.
            # Edges (v, u) where u in same side: currently NOT in cut. After flip, IN cut. Change: +1 per edge.
            # Edges (v, u) where u in opp side: currently IN cut. After flip, NOT in cut. Change: -1 per edge.
            # Total change in cut size = sum_same[v] - sum_opp[v]
            
            delta = sum_same[v] - sum_opp[v]
            
            # Acceptance
            accept = False
            if delta > 0:
                accept = True
            else:
                # Boltzmann factor
                if T > 0:
                    ratio = np.exp(delta / T)
                    if random.random() < ratio:
                        accept = True
            
            if accept:
                # Flip v
                old_label = labels[v]
                new_label = 1 - old_label
                labels[v] = new_label
                
                # Update cut
                current_cut += delta
                
                # Update sum_same and sum_opp for v
                sum_same[v], sum_opp[v] = sum_opp[v], sum_same[v]
                
                # Update sum_same and sum_opp for all neighbors of v
                # For any neighbor u of v:
                # If u was in same side as v (old), now u is in opp side relative to v (new).
                # So for u:
                #   u's sum_same decreases by 1 (v was same, now opp)
                #   u's sum_opp increases by 1
                # If u was in opp side relative to v (old), now u is in same side relative to v (new).
                # So for u:
                #   u's sum_same increases by 1
                #   u's sum_opp decreases by 1
                
                # Find neighbors of v
                neighbors = np.where(adj[v] == 1)[0]
                for u in neighbors:
                    if labels[u] == new_label:
                        # u is in same side as new v. So u was in opp side relative to old v.
                        # For u: v moves from opp to same.
                        sum_same[u] += 1
                        sum_opp[u] -= 1
                    else:
                        # u is in opp side relative to new v. So u was in same side relative to old v.
                        # For u: v moves from same to opp.
                        sum_same[u] -= 1
                        sum_opp[u] += 1
                
                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_labels = labels.copy()
            
            # Cool down
            T *= cf
            
        return best_local_labels, best_local_cut

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy: Multiple SA restarts with different initializations and temperature schedules.
    # Allocate time budget to SA runs.
    
    # Initial run with alternating start
    if time.time() < deadline:
        labels = np.arange(n) % 2
        # Quick local search to warm up
        # Use a simple 1-flip local search for a few passes
        for _ in range(5):
            changed = False
            for v in range(n):
                same = int(adj[v][labels == labels[v]].sum())
                other = int(adj[v][labels != labels[v]].sum())
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        try_labels(labels)

    # SA Restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:  # Leave 10s for final polish
            break
        
        # Initialization
        if r % 3 == 0:
            labels = np.random.randint(0, 2, size=n)
        elif r % 3 == 1:
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            # Spectral init (cheap version)
            degree = adj.sum(axis=1).astype(np.float64)
            D = np.diag(degree)
            L = D - adj.astype(np.float64)
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                k = min(5, n - 1)
                if k >= 2:
                    Y = eigvecs[:, 1:k+1]
                    norms = np.linalg.norm(Y, axis=1, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    Y = Y / norms
                    u = np.random.randn(k)
                    u = u / np.linalg.norm(u)
                    proj = Y @ u
                    labels = (proj > 0).astype(np.int8)
                else:
                    labels = np.random.randint(0, 2, size=n)
            except:
                labels = np.random.randint(0, 2, size=n)
        
        # SA Parameters
        # T_initial should be high enough to allow exploration.
        # Max possible delta is roughly n (if all edges cross).
        # Let's set T_initial to something like n / 10 or similar.
        T_initial = n / 5.0
        T_final = 0.1
        steps = 2000  # Adjust based on time
        
        # Estimate time per step? Hard to do. Just run and check deadline.
        sa_labels, sa_cut = simulated_annealing(labels, T_initial, T_final, steps, 0.999)
        try_labels(sa_labels)
        
        # Local search polish on the best found in SA
        if time.time() < deadline - 5.0:
            polished = sa_labels.copy()
            # 1-flip local search
            for _ in range(10):
                changed = False
                for v in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    same = int(adj[v][polished == polished[v]].sum())
                    other = int(adj[v][polished != polished[v]].sum())
                    if same > other:
                        polished[v] = 1 - polished[v]
                        changed = True
                if not changed:
                    break
            try_labels(polished)

    # Final polish on best found
    if best_labels is not None and time.time() < deadline - 1.0:
        final_labels = best_labels.copy()
        # 1-flip local search
        for _ in range(20):
            if time.time() > deadline - 1.0:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline - 1.0:
                    break
                same = int(adj[v][final_labels == final_labels[v]].sum())
                other = int(adj[v][final_labels != final_labels[v]].sum())
                if same > other:
                    final_labels[v] = 1 - final_labels[v]
                    changed = True
            if not changed:
                break
        try_labels(final_labels)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
