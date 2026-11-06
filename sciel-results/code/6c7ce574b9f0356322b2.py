import time
import random
import math

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    A is assumed to be a list of distinct positive integers.
    """
    n = len(A)
    sum_set = set()
    prod_set = set()
    
    # Using index to ensure i <= j for pairs
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
            
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Candidate pool: Small numbers are preferred for collisions
    # We go up to 2000 to allow some flexibility
    max_pool_val = 2000
    # Pre-compute the pool. We use a slightly biased distribution for random restarts
    # but the greedy search will explore the whole range.
    base_pool = list(range(1, max_pool_val + 1))
    
    best_set = None
    best_score = float('inf')
    
    # Strategy: Stochastic Beam Search for construction + Local Search polish
    # We will run several restarts of the beam search.
    
    num_restarts = 20
    beam_width = 5
    n_elements = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize the beam with the first element.
        # To add diversity, pick the first element randomly from a small range or 1.
        # 1 is almost always good for products (identity), but for sums it just shifts.
        # Let's start with 1 for most restarts, and a small random number for others.
        if restart % 5 == 0:
            first_val = 1
        else:
            first_val = random.randint(1, 50)
            
        # Beam: list of (score, set_list, sum_set, prod_set)
        # Score is max(len(sum_set), len(prod_set))
        beam = []
        current_set = [first_val]
        current_sums = {2 * first_val}
        current_prods = {first_val * first_val}
        current_score = max(len(current_sums), len(current_prods))
        beam.append((current_score, current_set, current_sums, current_prods))
        
        # Grow the set to size 20
        for step in range(1, n_elements):
            if time.time() > deadline - 5.0:
                break
                
            next_beam = []
            # For each state in the current beam, find the best next additions
            for score, A, sums, prods in beam:
                # We need to find the best next value to add to A.
                # To avoid checking all 2000 values for every beam state (expensive),
                # we sample a subset or use a heuristic.
                # Heuristic: Values that create many collisions with existing A are good.
                
                # Let's sample a random subset of candidates to keep it fast,
                # plus always check some small values.
                candidates = set()
                # Small values are often good
                for v in range(1, 50):
                    if v not in A:
                        candidates.add(v)
                # Random samples
                for _ in range(200):
                    v = random.randint(1, max_pool_val)
                    if v not in A:
                        candidates.add(v)
                
                best_local_score = float('inf')
                best_local_state = None
                
                for v in candidates:
                    new_A = A + [v]
                    new_sums = set(sums)
                    new_prods = set(prods)
                    
                    # Update sums and prods with v
                    for a in A:
                        new_sums.add(a + v)
                        new_prods.add(a * v)
                    new_sums.add(2 * v)
                    new_prods.add(v * v)
                    
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_state = (new_score, new_A, new_sums, new_prods)
                    
                    # Early termination if we find a perfect collision? (Unlikely)
                    # Or if we have enough candidates?
                    
                if best_local_state:
                    next_beam.append(best_local_state)
            
            # Prune the next beam to keep only the best 'beam_width' states
            # Sort by score
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            
            if not beam:
                break
        
        # After construction, we have a few candidate sets in the beam.
        # Take the best one and polish it with local search.
        if beam:
            best_candidate_score, best_candidate_set, _, _ = beam[0]
            
            # Local Search: Hill Climbing
            # We try to swap elements to improve the score.
            # To be fast, we only evaluate if the swap is promising.
            current_set = best_candidate_set[:]
            current_score = compute_metrics(current_set) # Recalculate to be safe
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
            
            # Polish loop
            improved = True
            polish_deadline = min(deadline - 1.0, time.time() + 2.0) # Limit polish time per restart
            
            while improved and time.time() < polish_deadline:
                improved = False
                for i in range(n_elements):
                    old_val = current_set[i]
                    # Try swapping with a random new value
                    # To be efficient, we don't try ALL values, just a sample
                    # and some neighbors.
                    candidates = set()
                    # Neighbors of old_val
                    candidates.update(range(max(1, old_val - 10), min(max_pool_val + 1, old_val + 11)))
                    # Random small values
                    for _ in range(20):
                        candidates.add(random.randint(1, 100))
                    # Random large values
                    for _ in range(20):
                        candidates.add(random.randint(100, max_pool_val))
                    
                    for new_val in candidates:
                        if new_val in current_set:
                            continue
                        
                        # Fast check: Calculate marginal impact
                        # This is expensive to do perfectly fast without precomputed sets,
                        # so we just compute the full metric but only for a limited number of candidates.
                        # Actually, let's just compute the full metric for the top few candidates.
                        # For now, we compute for all candidates in this small set.
                        
                        current_set[i] = new_val
                        new_score = compute_metrics(current_set)
                        
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_set = current_set[:]
                            improved = True
                            # Restart the loop for this i? Or break and go to next i?
                            # Usually, breaking to next i is fine, or re-checking i is better.
                            # Let's break and let the outer 'improved' loop handle it.
                            break
                        else:
                            current_set[i] = old_val
                
                # If we improved, we should probably re-scan all i again in the next iteration
                # but we already looped through all i in this iteration.
                # The 'improved' flag triggers another full pass.

    # Final check: ensure best_set is valid
    if best_set is None:
        # Fallback
        best_set = list(range(1, 21))
        best_score = compute_metrics(best_set)
        
    return sorted(best_set)
