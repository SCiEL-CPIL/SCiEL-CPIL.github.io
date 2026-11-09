import itertools
import random
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0   # Safety margin under the 90s kill
    N = 4096

    # Helper to check if v can be added to chosen set with existing sums
    def check_add(v, chosen, sums):
        """
        Checks if v can be added to the set 'chosen' without creating duplicate triple sums.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        new_sums = set()
        # Triple (v, v, v)
        new_sums.add(3 * v)
        
        # Triples (v, v, u) and (v, u, u) for u in chosen
        for u in chosen:
            new_sums.add(2 * v + u)
            new_sums.add(v + 2 * u)
            
        # Triples (v, u, w) for u, w in chosen with u <= w
        for i in range(len(chosen)):
            u = chosen[i]
            for j in range(i, len(chosen)):
                w = chosen[j]
                new_sums.add(v + u + w)
                
        # Check for conflicts with existing sums
        if new_sums & sums:
            return False, None
        return True, new_sums

    # Full validation for safety
    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # Greedy construction from a specific order of candidates
    def greedy_build(order, limit):
        chosen = []
        sums = set()
        for v in order:
            if time.time() > limit:
                break
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        return chosen

    # Compute collision density for candidate v given current set
    def collision_density(v, chosen, sums):
        """
        Estimate how many triple-sum collisions would occur if v were added.
        Returns the number of new sums that conflict with existing sums.
        """
        new_sums = set()
        new_sums.add(3 * v)
        for u in chosen:
            new_sums.add(2 * v + u)
            new_sums.add(v + 2 * u)
        for i in range(len(chosen)):
            u = chosen[i]
            for j in range(i, len(chosen)):
                w = chosen[j]
                new_sums.add(v + u + w)
        return len(new_sums & sums)

    # Simulated annealing to improve the set
    def simulated_annealing(chosen, limit, max_iter=500):
        """
        Uses simulated annealing to escape local optima.
        Moves: add, remove, or swap elements.
        Accepts non-improving moves with probability based on temperature.
        """
        current = list(chosen)
        current_sums = set()
        for x in itertools.combinations_with_replacement(current, 3):
            current_sums.add(sum(x))
        
        current_score = len(current)
        best = list(current)
        best_sums = set(current_sums)
        best_score = current_score
        
        T = 10.0  # Initial temperature
        cooling = 0.995
        
        for iteration in range(max_iter):
            if time.time() > limit:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [v for v in range(1, N + 1) if v not in set(current)]
                if not candidates:
                    continue
                v = random.choice(candidates)
                
                # Check if valid
                ok, ns = check_add(v, current, current_sums)
                if not ok:
                    # Compute collision density for scoring
                    col = collision_density(v, current, current_sums)
                    new_score = current_score  # Size unchanged if we don't add
                    delta = new_score - current_score - col * 0.1
                    if delta > 0 or random.random() < np.exp(delta / T):
                        # Accept but don't actually add (it's invalid)
                        pass
                    continue
                else:
                    new_current = current + [v]
                    new_sums = current_sums | ns
                    new_score = len(new_current)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current = new_current
                        current_sums = new_sums
                        current_score = new_score
                        if current_score > best_score:
                            best = list(current)
                            best_sums = set(current_sums)
                            best_score = current_score
                            
            elif move_type < 0.7:
                # Remove a random element
                if not current:
                    continue
                idx = random.randrange(len(current))
                v = current[idx]
                new_current = current[:idx] + current[idx+1:]
                
                # Recompute sums
                new_sums = set()
                for x in itertools.combinations_with_replacement(new_current, 3):
                    new_sums.add(sum(x))
                
                new_score = len(new_current)
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current = new_current
                    current_sums = new_sums
                    current_score = new_score
                    if current_score > best_score:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_score = current_score
                        
            else:
                # Swap: remove one, add another
                if not current:
                    continue
                idx = random.randrange(len(current))
                v_remove = current[idx]
                new_current = current[:idx] + current[idx+1:]
                
                candidates = [v for v in range(1, N + 1) if v not in set(new_current)]
                if not candidates:
                    continue
                v_add = random.choice(candidates)
                
                # Check if v_add can be added to new_current
                ok, ns = check_add(v_add, new_current, set())
                # Need to recompute sums for new_current first
                new_current_sums = set()
                for x in itertools.combinations_with_replacement(new_current, 3):
                    new_current_sums.add(sum(x))
                
                ok, ns = check_add(v_add, new_current, new_current_sums)
                if not ok:
                    continue
                    
                final_current = new_current + [v_add]
                final_sums = new_current_sums | ns
                final_score = len(final_current)
                delta = final_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current = final_current
                    current_sums = final_sums
                    current_score = final_score
                    if current_score > best_score:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_score = current_score
            
            T *= cooling
            if T < 0.01:
                break
        
        return best

    best = []
    
    # Strategy 1: Powers of 3 core + Greedy
    pow3 = [3 ** i for i in range(0, 8) if 3 ** i <= N]
    if is_valid(pow3):
        # Extend powers of 3 greedily
        order = list(range(1, N + 1))
        # Ensure pow3 elements are prioritized? 
        # Greedy build will pick them if they are in order.
        # But we want to start with them.
        chosen = list(pow3)
        sums = set()
        for x in itertools.combinations_with_replacement(chosen, 3):
            sums.add(sum(x))
            
        for v in range(1, N + 1):
            if time.time() > DEADLINE - 10:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # Strategy 2: Randomized Greedy Restarts
    random.seed(42)
    
    # We will run multiple trials with different random orderings
    num_trials = 50
    trial_time = (DEADLINE - 5) / num_trials
    
    for t in range(num_trials):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate a random order
        # Mix of small numbers and random numbers
        # Option A: Pure random permutation
        # Option B: Sorted, but with random swaps
        # Option C: Random weights
        
        # Let's try a "Randomized Ascending" approach:
        # Pick the next candidate from the smallest K available, with some probability picking a larger one.
        
        # Simpler: Random permutation of 1..N
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Bias towards smaller numbers?
        # Sort by (random() * 1000 + v) ?
        
        # Let's just use a shuffled list.
        # To improve, we can try "Greedy with Randomized Tie-Breaking"
        # Or "Greedy from Random Subsets"
        
        # Try 1: Pure Random
        res = greedy_build(order, DEADLINE - 5)
        if is_valid(res) and len(res) > len(best):
            best = res
            
        # Try 2: Random but biased to small
        # Create a list where small numbers appear more often?
        # No, greedy takes first valid.
        # Let's try a different order: Sort by v * random()
        order2 = list(range(1, N + 1))
        keys = [v * random.uniform(0.5, 2.0) for v in order2]
        order2 = [v for v, k in sorted(zip(order2, keys))]
        
        if time.time() < DEADLINE - 5:
            res2 = greedy_build(order2, DEADLINE - 5)
            if is_valid(res2) and len(res2) > len(best):
                best = res2

    # Strategy 3: Simulated Annealing on Best
    if time.time() < DEADLINE - 10 and best:
        sa_result = simulated_annealing(best, DEADLINE - 5, max_iter=300)
        if is_valid(sa_result) and len(sa_result) > len(best):
            best = sa_result

    # Strategy 4: Local Search on Best
    # Try to add more to the best set found
    if time.time() < DEADLINE - 2:
        chosen = list(best)
        sums = set()
        for x in itertools.combinations_with_replacement(chosen, 3):
            sums.add(sum(x))
        
        # Try to add any missing number
        for v in range(1, N + 1):
            if time.time() > DEADLINE - 1:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # Final sanity check
    best = sorted(set(best))
    return best
