import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = adjacency.astype(int)
    degrees = A.sum(axis=1)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving 20s for overhead
    
    best_set = np.zeros(n, dtype=bool)
    best_size = 0
    
    # Helper to run greedy with a specific score order
    def greedy_decode(scores):
        order = np.argsort(-scores)
        picked = np.zeros(n, dtype=bool)
        for v in order:
            if picked[v]:
                continue
            # Check if any neighbor is picked
            # A[v] gives neighbors. If any neighbor is picked, skip.
            if np.any(picked[A[v] == 1]):
                continue
            picked[v] = True
        return picked

    # Helper to run local search improvement
    def improve_set(picked):
        current_set = picked.copy()
        current_size = np.sum(current_set)
        improved = True
        steps = 0
        max_steps = 500 # Safety limit
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            if time.time() > deadline:
                break
                
            # Try removing each vertex in the set
            for v in np.where(current_set)[0]:
                if time.time() > deadline:
                    break
                
                # Temporarily remove v
                current_set[v] = False
                
                # Find vertices that are not in set and not adjacent to any other vertex in set
                # neighbors_of_set = A @ current_set
                # We need u such that current_set[u] == 0 and (A @ current_set)[u] == 0
                # Note: current_set[v] is now False, so A @ current_set reflects neighbors of S \ {v}
                
                neighbors_count = A @ current_set
                candidates = np.where((~current_set) & (neighbors_count == 0))[0]
                
                if len(candidates) == 0:
                    current_set[v] = True
                    continue
                
                # Try to add a maximal independent set from candidates
                # Greedy on candidates: sort by degree to other candidates? Or just global degree?
                # Let's just greedily pick from candidates in order of lowest global degree
                cand_degrees = degrees[candidates]
                cand_order = candidates[np.argsort(cand_degrees)]
                
                new_additions = 0
                temp_set = current_set.copy()
                for u in cand_order:
                    if temp_set[u]:
                        continue
                    if np.any(temp_set[A[u] == 1]):
                        continue
                    temp_set[u] = True
                    new_additions += 1
                
                # We removed 1, added new_additions
                if new_additions > 1:
                    current_set = temp_set
                    current_size += (new_additions - 1)
                    improved = True
                    # Restart the loop to check for more improvements
                    break 
                else:
                    # Revert
                    current_set[v] = True
        
        return current_set, np.sum(current_set)

    # Run multiple starts with different initial heuristics
    strategies = []
    
    # 1. Lowest degree first
    strategies.append(degrees.copy())
    
    # 2. Random
    strategies.append(np.random.rand(n))
    
    # 3. High degree first (sometimes good to block hubs early? No, usually bad for MIS, but let's try)
    strategies.append(-degrees)
    
    # 4. Random permutation
    strategies.append(np.random.rand(n))
    
    for i, scores in enumerate(strategies):
        if time.time() > deadline:
            break
            
        # Add small noise to break ties
        scores_noisy = scores + np.random.normal(0, 0.01, n)
        
        # Greedy phase
        picked = greedy_decode(scores_noisy)
        size = np.sum(picked)
        
        # Local Search phase
        if size > best_size:
            improved_picked, improved_size = improve_set(picked)
            if improved_size > best_size:
                best_size = improved_size
                best_set = improved_picked
        
        # Also check if the greedy result itself is better (in case improve_set fails or times out)
        if size > best_size:
            best_size = size
            best_set = picked

    # If we didn't find anything, fallback to simple degree-based
    if best_size == 0:
        picked = greedy_decode(-degrees)
        best_set = picked
        best_size = np.sum(picked)

    # Construct final priority vector
    # Vertices in best_set get high score, others get low score
    # The fixed greedy decoder will pick all vertices in best_set (since they are independent and highest priority)
    # and skip all others (since they are neighbors of best_set or lower priority)
    
    final_scores = np.zeros(n)
    final_scores[best_set] = 1.0
    
    # To ensure deterministic tie-breaking or stability, add tiny epsilon
    # But since we want the greedy decoder to pick EXACTLY best_set, 
    # and best_set is independent, the order among them doesn't matter for correctness.
    # However, if there are ties in the "0.0" group, it doesn't matter as they won't be picked 
    # (assuming best_set is maximal).
    
    return final_scores
