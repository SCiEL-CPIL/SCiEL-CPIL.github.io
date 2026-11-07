import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Precompute all potential triples (a, b, c) with a < b < c
    # such that a+b, a+c, b+c are within 1..N
    # A triple is a violation if all 6 values are in the set.
    # We represent the set as a bitmask.
    
    # List of triples. Each triple is a tuple of 6 indices (1-based for ease of masking)
    # (a, b, c, a+b, a+c, b+c)
    all_triples = []
    # Map from element index to list of triple indices involving it
    triples_involving = {i: [] for i in range(1, N + 1)}

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            for c in range(b + 1, N + 1):
                s_ac = a + c
                s_bc = b + c
                if s_ac > N or s_bc > N:
                    break
                # Valid candidate triple
                t_idx = len(all_triples)
                all_triples.append((a, b, c, s_ab, s_ac, s_bc))
                for elem in (a, b, c, s_ab, s_ac, s_bc):
                    triples_involving[elem].append(t_idx)

    def get_violation_count(mask):
        """Calculate the number of violated triples for a given mask."""
        count = 0
        for t in all_triples:
            a, b, c, s1, s2, s3 = t
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1 and \
               (mask >> s1) & 1 and (mask >> s2) & 1 and (mask >> s3) & 1:
                count += 1
        return count

    def delta_violation(mask, k):
        """Calculate change in violation count if we flip bit k."""
        # If bit k is 1, we are removing it. New state: 0.
        # If bit k is 0, we are adding it. New state: 1.
        # Current contribution of triples involving k to total violation count:
        # A triple contributes 1 if all 6 bits are 1.
        # If we flip k, the contribution changes only if the other 5 bits are 1.
        
        delta = 0
        for t_idx in triples_involving[k]:
            a, b, c, s1, s2, s3 = all_triples[t_idx]
            # Check if the other 5 bits are 1
            other_ones = 0
            for elem in (a, b, c, s1, s2, s3):
                if elem == k:
                    continue
                if (mask >> elem) & 1:
                    other_ones += 1
            
            if other_ones == 5:
                # If k is currently 1, this triple is currently violated (contributes 1).
                # After flip, it will not be violated (contributes 0). Delta = -1.
                # If k is currently 0, this triple is currently NOT violated (contributes 0).
                # After flip, it will be violated (contributes 1). Delta = +1.
                if (mask >> k) & 1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    best_size = 0
    best_mask = 0

    # Run multiple restarts of Simulated Annealing
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize mask
        # Start with a random subset or all elements?
        # Starting with all elements has high violation count.
        # Starting with random might be better for SA.
        # Let's try starting with a random permutation and adding greedily to get a valid start,
        # or just random mask.
        # Random mask is fine for SA.
        mask = 0
        # Maybe start with a dense random set to have many elements
        # Let's try a random set of density 0.5
        for i in range(1, N + 1):
            if random.random() < 0.5:
                mask |= (1 << i)
        
        current_violations = get_violation_count(mask)
        current_size = bin(mask).count("1")
        
        # SA Parameters
        T_start = 10.0
        T_end = 0.1
        steps = 5000
        
        # Objective: Minimize Cost = violations * 1000 - size
        # We want to minimize this.
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick random element to flip
            k = random.randint(1, N)
            
            d_viol = delta_violation(mask, k)
            # d_size is +1 if adding (0->1), -1 if removing (1->0)
            if (mask >> k) & 1:
                d_size = -1
            else:
                d_size = +1
            
            # Calculate change in Cost
            # Cost = 1000 * violations - size
            d_cost = 1000 * d_viol - d_size
            
            # Temperature schedule
            # Linear cooling
            T = T_start - (T_start - T_end) * (step / steps)
            
            if d_cost < 0:
                # Always accept improving moves
                mask ^= (1 << k)
                current_violations += d_viol
                current_size += d_size
            else:
                # Accept with probability exp(-d_cost / T)
                # If d_cost is large, probability is small.
                # Use random log trick to avoid overflow
                if random.random() < math.exp(-d_cost / T):
                    mask ^= (1 << k)
                    current_violations += d_viol
                    current_size += d_size
            
            # Track best valid solution found so far
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_mask = mask

    # Final Polish: Greedy addition on the best found mask
    # We can try to add any missing element that doesn't create a violation
    # This is a local search step
    
    # We need a fast check for "can add"
    # We can use the delta_violation function.
    # If we add x, delta_violation tells us change in violations.
    # If delta is 0 and x was not in set, we can add it.
    # But delta_violation assumes we flip.
    # If x is 0, delta is change if we set to 1.
    # So if delta_violation(mask, x) == 0, we can add x without increasing violations.
    
    # Iterate through missing elements
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if (best_mask >> x) & 1:
            continue
        # Check if adding x keeps violations at 0
        # Note: best_mask might have violations > 0 if we didn't find a valid one?
        # But we track best_size only when violations == 0.
        # So best_mask is always valid (violations == 0).
        d_v = delta_violation(best_mask, x)
        if d_v == 0:
            best_mask |= (1 << x)
            best_size += 1

    # Convert mask to list
    result = [int(i) for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result
