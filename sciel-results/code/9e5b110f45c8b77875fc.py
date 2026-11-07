"""Regret-based heuristic for ACO orienteering."""
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Ensure float arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Start with a baseline: prize / distance (like seed)
    # But enhance with regret-based counterfactual value
    
    # For each node i, simulate a greedy tour and count how often each j
    # is selected as the next step, weighted by the prize collected.
    # This gives us a "probability" that j is the optimal next step from i.
    
    # Compute a regret matrix: regret[i,j] = prize[j] * P(j is optimal next from i)
    # P(j is optimal next from i) estimated by: in a greedy tour from i,
    # how often is j picked as the next unvisited node, normalized.
    
    # Efficient approach: For each i, do a quick greedy pass and track counts.
    # But n=50, so 50 greedy passes of up to ~5 steps each is fast.
    
    # Greedy tour from i: at each step, pick unvisited j maximizing prize[j]/(d[i,j]+d[j,0])
    # subject to length constraint. Count how often each j is picked.
    
    regret = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[i] = True
        current = i
        length_so_far = 0.0
        count = np.zeros(n, dtype=np.float64)
        
        # Greedy: repeatedly pick best next node
        for step in range(n - 1):
            best_j = -1
            best_score = -1.0
            for j in range(n):
                if visited[j]:
                    continue
                d_ij = distance_matrix[current, j]
                d_j0 = distance_matrix[j, 0]
                if length_so_far + d_ij + d_j0 > max_len:
                    continue
                # Score: prize per remaining budget usage
                score = prizes[j] / (d_ij + 1e-9)
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j == -1:
                break
            count[best_j] += 1.0
            length_so_far += distance_matrix[current, best_j]
            visited[best_j] = True
            current = best_j
        
        # Normalize counts to get probabilities
        total = count.sum()
        if total > 0:
            prob = count / total
            # Regret: prize[j] * prob[j] for each j
            regret[i, :] = prizes * prob
    
    # Combine with baseline prize/distance heuristic
    # Use a weighted combination to balance direct prize density with regret value
    
    # Baseline: prize / distance (seed heuristic)
    baseline = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                baseline[i, j] = prizes[j] / (distance_matrix[i, j] + 1e-9)
    
    # Combine: use regret as primary, baseline as secondary
    # Normalize both to [0,1] range for fair combination
    regret_max = regret.max()
    baseline_max = baseline.max()
    if regret_max > 0:
        regret_norm = regret / regret_max
    else:
        regret_norm = regret
    if baseline_max > 0:
        baseline_norm = baseline / baseline_max
    else:
        baseline_norm = baseline
    
    # Weighted combination: 0.7 regret + 0.3 baseline
    combined = 0.7 * regret_norm + 0.3 * baseline_norm
    
    # Ensure all entries are positive (task requirement: non-positive become 1e-9)
    combined[combined <= 0] = 1e-9
    
    # Also ensure diagonal is small (no self-loops)
    np.fill_diagonal(combined, 1e-9)
    
    return combined
