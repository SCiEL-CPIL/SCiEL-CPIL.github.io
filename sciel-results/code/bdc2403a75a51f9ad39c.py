import numpy as np
import time

def partition(adjacency):
    """
    Solves the Maximum Cut problem using Vectorized Simulated Annealing 
    with multiple restarts and a final local search polish.
    """
    A = np.array(adjacency, dtype=np.float64)
    n = len(A)
    
    # Ensure symmetry and zero diagonal for safety, though input should be valid
    # A = (A + A.T) / 2
    # np.fill_diagonal(A, 0)

    def cut_value(s):
        # s is in {-1, 1}
        # Cut size = 0.25 * sum(A_ij * (1 - s_i s_j))
        # = 0.25 * (sum(A) - sum(A_ij s_i s_j))
        # sum(A) is constant. Minimize s^T A s.
        # We return the actual cut size for comparison.
        val = s @ A @ s
        # Total edges = sum(A)/2
        # Cut = (TotalEdges * 2 - val) / 4 ? 
        # Let's derive: 
        # s_i s_j = 1 if same, -1 if different.
        # sum(A_ij s_i s_j) = sum_same - sum_diff.
        # sum_same + sum_diff = 2 * E (since A is symmetric, each edge counted twice in matrix sum? No, A_ij is 1 if edge. sum(A) = 2E).
        # Wait, s^T A s = sum_i sum_j s_i A_ij s_j.
        # If i!=j, A_ij=1 if edge.
        # Term is 1 if same, -1 if different.
        # Let S be sum over edges of (1 if same, -1 if diff).
        # Cut size C is number of edges with diff.
        # Same edges = E - C.
        # S = (E - C) * 1 + C * (-1) = E - 2C.
        # So E - 2C = s^T A s / 2 ? No.
        # s^T A s = sum_{i,j} A_ij s_i s_j.
        # For each edge (i,j), it appears as A_ij s_i s_j + A_ji s_j s_i = 2 s_i s_j.
        # So s^T A s = 2 * (sum_edges s_i s_j) = 2 * ( (E-C) - C ) = 2(E - 2C) = 2E - 4C.
        # Therefore, 4C = 2E - s^T A s.
        # C = (2E - s^T A s) / 4.
        total_edges = np.sum(A) / 2.0
        return (2 * total_edges - (s @ A @ s)) / 4.0

    def local_search(s, max_iters=1000):
        """Greedy local search to polish the solution."""
        s = s.copy()
        for _ in range(max_iters):
            field = A @ s
            # Delta E for flipping i is -4 * s_i * field_i
            # We want to minimize s^T A s, so we want negative Delta E.
            # If Delta E < 0, flipping improves cut.
            delta_e = -4.0 * s * field
            # Find best improvement
            best_idx = np.argmin(delta_e)
            if delta_e[best_idx] >= 0:
                break
            s[best_idx] *= -1
        return s

    def sa_restart(time_budget):
        """Run Simulated Annealing for a specific time budget."""
        s = np.random.choice([-1, 1], size=n)
        current_score = cut_value(s)
        best_s = s.copy()
        best_score = current_score
        
        # Cooling schedule
        # Start high, end low.
        T_start = 10.0
        T_end = 0.01
        # We don't know exact steps, so we use a simple decay per step
        # Or better, decay based on elapsed time in this restart.
        
        start_time = time.time()
        step = 0
        # Estimate steps per second to set decay rate?
        # Simpler: fixed number of steps per second?
        # Let's just run until time_budget is exhausted.
        # Use a dynamic cooling factor.
        # T = T_start * exp(-alpha * t_elapsed)
        alpha = 2.0 # Decay rate
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            
            # Calculate field
            field = A @ s
            # Calculate delta energy for all flips
            delta_e = -4.0 * s * field
            
            # Acceptance probability
            # If delta_e < 0, always accept.
            # If delta_e > 0, accept with prob exp(-delta_e / T)
            T = max(T_end, T_start * np.exp(-alpha * elapsed))
            
            # Vectorized acceptance
            # r is random uniform
            r = np.random.random(n)
            # prob_accept = 1 if delta_e < 0 else exp(-delta_e/T)
            # We can compute log_prob = -delta_e / T
            # Accept if log(r) < log_prob
            # To avoid log(0), use r > 0
            # Or simply:
            # accept = (delta_e < 0) | (np.random.random(n) < np.exp(-delta_e / T))
            # This is safe.
            
            # Optimization: 
            # Only calculate exp for positive delta_e
            pos_mask = delta_e > 0
            neg_mask = delta_e < 0
            
            # Initialize accept array
            accept = np.zeros(n, dtype=bool)
            accept[neg_mask] = True
            
            if np.any(pos_mask):
                # Calculate probabilities for positive deltas
                # Clip to avoid overflow in exp if T is very small
                probs = np.exp(-delta_e[pos_mask] / T)
                # Accept if random < prob
                accept[pos_mask] = (np.random.random(np.sum(pos_mask)) < probs)
            
            if np.any(accept):
                s[accept] *= -1
                new_score = cut_value(s)
                if new_score > best_score:
                    best_score = new_score
                    best_s = s.copy()
            else:
                # No changes, just cool down (implicit in elapsed time)
                pass
            
            step += 1
            
            # Micro-optimization: if no changes for a long time, break early?
            # The time check handles it.

        # Final polish
        best_s = local_search(best_s)
        final_score = cut_value(best_s)
        
        return best_s, final_score

    # Main Loop
    start_total = time.time()
    best_global_s = None
    best_global_score = -1
    
    # Strategy: Multiple restarts.
    # Each restart gets a small time slice, e.g., 2 seconds.
    # As time runs out, make slices smaller to fit more restarts.
    # Or just loop with a fixed slice size.
    slice_time = 2.0
    
    while True:
        elapsed_total = time.time() - start_total
        remaining = 145.0 - elapsed_total # Safety margin
        if remaining <= 0:
            break
        
        current_slice = min(slice_time, remaining)
        if current_slice < 0.5:
             # If very little time left, do one last polish on the best found
             if best_global_s is not None:
                 best_global_s = local_search(best_global_s)
                 best_global_score = cut_value(best_global_s)
             break
        
        s, score = sa_restart(current_slice)
        
        if score > best_global_score:
            best_global_score = score
            best_global_s = s
            
        # Adapt slice time? 
        # If we found a very good solution, maybe spend more time polishing?
        # For now, keep it simple.
        
        # Optional: If we hit a score that is likely optimal (e.g. > 0.5 * total_edges * 1.05 heuristic?), 
        # we could stop, but Max Cut is hard. 0.5 is random. Good solutions are ~0.5-0.6.
        # We'll just run out of time.

    # Final local search on the best candidate
    if best_global_s is not None:
        best_global_s = local_search(best_global_s)
    
    # Convert spins {-1, 1} to labels {0, 1}
    # -1 -> 0, 1 -> 1
    labels = (best_global_s + 1) // 2
    labels = labels.astype(int)
    
    return labels
