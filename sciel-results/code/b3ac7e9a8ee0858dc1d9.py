import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    n_elements = 129  # 0 to 128
    
    def compute_counts(A_set):
        """Compute pair sum counts for a given set A"""
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = sorted(A_set)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, n):
                aj = A_list[j]
                s_ij = ai + aj
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        return counts
    
    def get_max_mult(counts):
        """Get maximum multiplicity in target range"""
        target_counts = counts[min_sum:max_sum + 1]
        return int(np.max(target_counts))
    
    def get_min_mult(counts):
        """Get minimum multiplicity in target range"""
        target_counts = counts[min_sum:max_sum + 1]
        return int(np.min(target_counts))
    
    def is_valid(counts):
        """Check if all target sums are covered"""
        target_counts = counts[min_sum:max_sum + 1]
        return np.all(target_counts > 0)
    
    # Multiplicative weights / load balancing approach
    # Maintain weights for each element 0-128
    # High weight = more likely to be in A
    
    best_score = float('inf')
    best_set = None
    
    # Try different set sizes
    for k in range(16, 35):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts with different initial weights
        num_restarts = 8 if k < 25 else 4
        time_per_restart = (deadline - time.time() - 5) / (num_restarts * (35 - k))
        if time_per_restart < 0.05:
            continue
        
        for r in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize weights: start with uniform or slightly biased
            weights = np.ones(n_elements)
            
            # Initialize A: pick k elements with highest weights (with some randomness)
            # Add some randomness to avoid getting stuck
            noise = np.random.dirichlet(np.ones(n_elements))
            weighted = weights * (1 + 0.1 * noise)
            
            # Select top k elements
            A_indices = np.random.choice(n_elements, size=k, replace=False, p=weighted / weighted.sum())
            A_set = set(A_indices.tolist())
            
            # Iterative reweighting
            max_iter = 500
            T = 1.0
            cooling = 0.995
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                counts = compute_counts(A_set)
                
                if not is_valid(counts):
                    # Invalid: need to fix coverage
                    # Find uncovered sums
                    uncovered = [s for s in target_sums if counts[s] == 0]
                    
                    # For each uncovered sum, try to add an element that helps
                    for s in uncovered[:5]:  # Limit to 5 per iteration
                        # Find pairs (a, b) where a+b=s and at least one is not in A
                        best_elem = None
                        best_gain = 0
                        for a in range(max(0, s - 128), min(128, s) + 1):
                            b = s - a
                            if b < 0 or b > 128:
                                continue
                            # Check if adding a or b would help
                            if a not in A_set and b in A_set:
                                gain = 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_elem = a
                            elif b not in A_set and a in A_set:
                                gain = 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_elem = b
                            elif a not in A_set and b not in A_set:
                                # Adding both would help, but we can only add one
                                # Prefer the one that helps more other sums
                                gain = 0.5
                                if gain > best_gain:
                                    best_gain = gain
                                    best_elem = a
                        
                        if best_elem is not None:
                            # Remove a random element to keep size k
                            if len(A_set) >= k:
                                remove_elem = random.choice(list(A_set))
                                A_set.discard(remove_elem)
                            A_set.add(best_elem)
                    
                    continue
                
                # Valid solution: try to improve max multiplicity
                max_mult = get_max_mult(counts)
                
                if max_mult < best_score:
                    best_score = max_mult
                    best_set = A_set.copy()
                    if best_score <= 2:
                        break  # Can't do better than 2 for reasonable k
                
                # Find over-represented sums
                over_sums = [s for s in target_sums if counts[s] >= max_mult]
                
                # Penalize elements that contribute to over-represented sums
                penalty = np.zeros(n_elements)
                for s in over_sums:
                    for a in range(max(0, s - 128), min(128, s) + 1):
                        b = s - a
                        if b < 0 or b > 128:
                            continue
                        if a in A_set and b in A_set:
                            penalty[a] += 1
                            penalty[b] += 1
                        elif a == b and a in A_set:
                            penalty[a] += 1
                
                # Boost elements that help under-represented sums
                under_sums = [s for s in target_sums if counts[s] <= max_mult - 2]
                boost = np.zeros(n_elements)
                for s in under_sums:
                    for a in range(max(0, s - 128), min(128, s) + 1):
                        b = s - a
                        if b < 0 or b > 128:
                            continue
                        if a not in A_set and b in A_set:
                            boost[a] += 1
                        elif b not in A_set and a in A_set:
                            boost[b] += 1
                
                # Update weights
                weights = weights * (1 - 0.1 * penalty) * (1 + 0.1 * boost)
                weights = np.maximum(weights, 0.01)
                
                # Occasionally swap an element based on weights
                if random.random() < 0.3:
                    # Pick an element to remove (lower weight preferred)
                    remove_candidates = [x for x in A_set if penalty[x] > 0]
                    if not remove_candidates:
                        remove_candidates = list(A_set)
                    remove_elem = random.choice(remove_candidates)
                    
                    # Pick an element to add (higher weight preferred)
                    add_candidates = [x for x in range(n_elements) if x not in A_set and boost[x] > 0]
                    if not add_candidates:
                        add_candidates = [x for x in range(n_elements) if x not in A_set]
                    
                    # Weighted selection
                    add_weights = np.array([weights[x] for x in add_candidates])
                    if np.sum(add_weights) > 0:
                        add_elem = random.choice(add_candidates, p=add_weights / add_weights.sum())
                    else:
                        add_elem = random.choice(add_candidates)
                    
                    A_set.discard(remove_elem)
                    A_set.add(add_elem)
                
                T *= cooling
        
        # If we found a very good solution, stop early
        if best_score <= 2:
            break
    
    # If no valid solution found, fallback
    if best_set is None:
        # Try a simple structured approach
        # Use all even numbers from 0 to 128
        A = list(range(0, 129, 2))
        # Check if valid
        counts = compute_counts(set(A))
        if is_valid(counts):
            return A
        else:
            # Fallback to seed-like
            return list(range(64, 129))
    
    return list(best_set)
