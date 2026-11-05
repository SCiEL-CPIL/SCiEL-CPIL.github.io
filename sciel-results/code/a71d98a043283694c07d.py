import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 511
    
    # Precompute conflicts: conflicts[a] contains b if gcd(a,b) > 1
    # We need to check if a_new is coprime with all others.
    # It is faster to check gcd directly if we don't store the list, 
    # but storing a set of conflicts allows O(1) check if we had a count.
    # Given |S| is small, iterating S and checking gcd is fine.
    # But precomputing coprime matrix is also fine.
    
    # Let's stick to the coprime matrix for O(1) check per pair.
    coprime = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if math.gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
                
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)

    def greedy_init(randomized=False):
        candidates = list(range(1, N + 1))
        if randomized:
            # Bias towards larger numbers to get good score quickly, 
            # but shuffle to allow diversity?
            # Simple shuffle might pick small numbers first, leading to bad sets.
            # Better: Sort by weight descending, then shuffle with a "temperature" for order?
            # Or just shuffle completely? 
            # If we shuffle completely, we might pick 1, 2, ... which is bad.
            # We want to pick large numbers.
            # Let's sort by value descending, but randomly swap adjacent elements with prob 0.1?
            # Or just use the standard greedy for the first few, then random?
            # Let's try: Sort by value descending. Then, with prob 0.5, swap i and i+1?
            # Actually, just shuffling the *order of consideration* among candidates with similar weights?
            # Let's try a simple approach: 
            # 1. Standard greedy (descending value)
            # 2. Randomized: Shuffle candidates, but weight the selection?
            # Let's just do standard greedy for the first init, 
            # and for subsequent inits, do a "noisy" greedy:
            # Iterate 1..N, but randomly skip some? No.
            
            # Let's try: Randomly permute 1..N, but keep the top 50% sorted by value?
            # Simpler: Just use standard greedy. The SA will explore.
            # To get diversity, we can perturb the best solution found by SA.
            
            pass # We will implement perturbation instead of random greedy init for now.
            
        chosen = []
        # Sort by value descending to maximize score in greedy phase
        for a in range(N, 0, -1):
            if all(coprime[a][b] for b in chosen):
                chosen.append(a)
        return chosen

    best_set = greedy_init()
    best_score = get_score(best_set)
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    all_candidates = list(range(1, N + 1))
    
    # Simulated Annealing
    def run_sa(init_set, max_time=10.0):
        current_set = list(init_set)
        current_score = get_score(current_set)
        local_best_set = list(current_set)
        local_best_score = current_score
        
        T = 0.1
        cooling = 0.999
        min_T = 0.0001
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            if T < min_T:
                # Restart cooling
                T = 0.05
                # Perturb current best?
                # Let's just restart from current_set
                pass
            
            # Pick a random element to remove
            if not current_set:
                break
            idx = random.randint(0, len(current_set) - 1)
            a_old = current_set[idx]
            
            # Pick a random candidate to add
            # Optimization: Pick from candidates not in set
            # Random choice from 1..N and check if in set is O(1) avg if set is small?
            # Set lookup is O(1).
            a_new = random.choice(all_candidates)
            
            if a_new in current_set:
                continue
            
            # Check validity
            # a_new must be coprime with all a in current_set except a_old
            valid = True
            for i, a in enumerate(current_set):
                if i == idx:
                    continue
                if not coprime[a_new][a]:
                    valid = False
                    break
            
            if valid:
                # Calculate delta
                delta = (1.0 / (512 - a_new)) - (1.0 / (512 - a_old))
                
                # Acceptance
                accept = False
                if delta > 0:
                    accept = True
                else:
                    # Avoid exp overflow for very negative delta
                    if delta > -10 * T: 
                        if random.random() < math.exp(delta / T):
                            accept = True
                
                if accept:
                    # Perform swap
                    current_set[idx] = a_new
                    current_score += delta
                    
                    if current_score > local_best_score:
                        local_best_score = current_score
                        local_best_set = list(current_set)
                    
                    T *= cooling
            else:
                # Invalid move, don't change T? Or cool slightly?
                # Usually T cools regardless of move validity in some implementations,
                # but here we only cool on valid attempts to keep temperature relevant to the search space.
                # Let's cool on every valid candidate selection (even if invalid, it was a "step" in the search space exploration? No, invalid is just a failed step).
                # Let's cool only on valid swaps.
                pass

        return local_best_set, local_best_score

    # Main Loop: Multi-Start SA
    current_init = list(best_set)
    
    while time.time() < deadline:
        # Run SA for a slice of time
        # Use a shorter slice to allow multiple restarts
        slice_time = min(10.0, deadline - time.time())
        if slice_time <= 0:
            break
            
        new_set, new_score = run_sa(current_init, max_time=slice_time)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            # Restart from the new best
            current_init = list(best_set)
        else:
            # If no improvement, perturb the current best to escape local optimum
            # Create a perturbed version of best_set
            perturbed = list(best_set)
            # Remove 2-3 random elements
            for _ in range(random.randint(2, 3)):
                if perturbed:
                    idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            # Add some random valid elements?
            # Just use this as the next init
            current_init = perturbed

    return best_set
# EVOLVE-BLOCK-END
