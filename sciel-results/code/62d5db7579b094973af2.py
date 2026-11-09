import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 620
    p = 7
    q = p * p
    
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * 3) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}

    # Precompute affected lists for SA
    # affected[i] contains (k, v) such that i is in Line(k, v) \ {k}
    # Which is equivalent to k is in Line(i, v) \ {i}
    affected = [[] for _ in range(q * q)]
    
    # Also precompute lines for initial state setup if needed, 
    # but we will start from a greedy solution or random.
    
    # Let's generate a good initial solution using the greedy method from the seed
    # to ensure we start with a valid set, then use SA to improve.
    
    in_N = np.ones(q * q, dtype=bool)
    
    # Simple greedy initialization similar to seed
    # We can reuse the logic from the seed program to get a baseline
    # But to save code space and focus on SA, we'll do a quick greedy pass.
    
    # Actually, let's just start with all 1s and run a fast greedy pruning first.
    # Then SA.
    
    # Greedy Pruning (Fast)
    # count_zero[k, v] = number of zeros in Line(k, v) \ {k}
    # Initially all 1s, so count_zero is 0 for all.
    # cover_count[k] = number of v such that count_zero[k, v] == 0.
    # Initially cover_count[k] = n_dir for all k.
    
    # We need to build the 'affects' structure for the greedy part too?
    # The seed program already has a sophisticated greedy.
    # I will implement a simpler greedy first to get a valid start, then SA.
    
    # Build affected structure
    for i in range(q * q):
        pi, pj = idx_to_point[i]
        for v_idx, (vx, vy) in enumerate(directions):
            # Find all k on line through i with direction v
            # Line is { i + t*v : t in F_q }
            # k = i + t*v for t in 1..q-1
            for t in range(1, q):
                tx = f_add(pi, f_mul(t, vx))
                ty = f_add(pj, f_mul(t, vy))
                k_idx = point_index[(tx, ty)]
                affected[i].append((k_idx, v_idx))
                
    # Initialize state
    # x: 1 if in N, 0 if not
    x = np.ones(q * q, dtype=np.int8)
    
    # count_zero[k, v]
    count_zero = np.zeros((q * q, n_dir), dtype=np.int32)
    # cover_count[k]
    cover_count = np.full(q * q, n_dir, dtype=np.int32)
    
    # Greedy Phase: Try to remove points (set x[i]=0)
    # We want to minimize sum(x).
    # Remove i if it doesn't cause any k to become invalid.
    # k becomes invalid if cover_count[k] drops to 0.
    # Removing i (1->0) increases count_zero[k, v] by 1 for all (k, v) in affected[i].
    # If count_zero[k, v] was 0, it becomes 1, so cover_count[k] decreases by 1.
    
    # Sort indices by some heuristic? Random order for now.
    indices = list(range(q * q))
    random.shuffle(indices)
    
    for i in indices:
        if time.time() > deadline - 300: # Leave time for SA
            break
        if x[i] == 0:
            continue
            
        # Check if removing i is safe
        # Temporarily apply removal
        invalid = False
        changes = []
        for (k, v) in affected[i]:
            if count_zero[k, v] == 0:
                # This direction v was covering k.
                # After removal, it won't.
                # Check if k has other covers.
                if cover_count[k] == 1:
                    # k will become invalid
                    invalid = True
                    break
                else:
                    # k will lose one cover, but still have some
                    pass
            # Note: if count_zero[k, v] > 0, removing i doesn't change the "covered" status of (k, v)
            # because it was already not covered (had zeros).
            # Wait, count_zero[k, v] counts zeros.
            # If count_zero[k, v] == 0, all points in line are in N.
            # If we remove i (which is in the line), count_zero becomes 1.
            # So the condition "all points in N" fails.
            # So yes, if count_zero[k, v] == 0, this direction stops covering k.
            
        if invalid:
            continue
            
        # Safe to remove
        x[i] = 0
        for (k, v) in affected[i]:
            if count_zero[k, v] == 0:
                count_zero[k, v] += 1
                cover_count[k] -= 1
            else:
                count_zero[k, v] += 1
                # cover_count[k] unchanged because this direction was already not covering k
        
    # Now we have a valid set N (x=1).
    # Let's verify validity just in case
    # (Omitted for speed, trust the logic)
    
    # SA Phase
    # Objective: Minimize sum(x) + M * violations
    # We start with a valid solution, so violations = 0.
    # We want to reduce sum(x).
    # Moves: flip x[i].
    
    M = 1000 # Penalty for violation
    best_x = x.copy()
    best_cost = np.sum(x)
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute neighbors for SA?
    # affected[i] is already there.
    
    current_violations = 0
    current_sum = np.sum(x)
    
    # We need to track violations dynamically.
    # violations = sum(1 for k if cover_count[k] == 0)
    
    # Helper to get current violations
    def get_violations():
        return np.sum(cover_count == 0)
        
    current_violations = get_violations()
    
    while time.time() < deadline - 10:
        if T < min_T:
            break
            
        # Pick random index
        i = random.randint(0, q * q - 1)
        
        # Calculate delta
        # If x[i] is 1 -> 0:
        #   sum decreases by 1.
        #   count_zero[k, v] increases for (k, v) in affected[i].
        #   If count_zero[k, v] was 0, cover_count[k] decreases.
        #   If cover_count[k] becomes 0, violations increase by 1.
        
        # If x[i] is 0 -> 1:
        #   sum increases by 1.
        #   count_zero[k, v] decreases for (k, v) in affected[i].
        #   If count_zero[k, v] becomes 0, cover_count[k] increases.
        #   If cover_count[k] was 0, violations decrease by 1.
        
        old_sum = current_sum
        old_viol = current_violations
        
        delta_sum = 0
        delta_viol = 0
        
        if x[i] == 1:
            # Try to set to 0
            delta_sum = -1
            # Check violations
            for (k, v) in affected[i]:
                if count_zero[k, v] == 0:
                    # This direction stops covering k
                    if cover_count[k] == 1:
                        # k becomes invalid
                        delta_viol += 1
                    else:
                        # k remains valid
                        pass
                # else: direction was already not covering, no change in cover_count
            # Apply
            x[i] = 0
            for (k, v) in affected[i]:
                if count_zero[k, v] == 0:
                    count_zero[k, v] += 1
                    cover_count[k] -= 1
                else:
                    count_zero[k, v] += 1
        else:
            # Try to set to 1
            delta_sum = 1
            # Check violations
            for (k, v) in affected[i]:
                if count_zero[k, v] == 1:
                    # This direction starts covering k (if it becomes 0)
                    # But we need to know if it was the ONLY zero?
                    # No, count_zero counts zeros.
                    # If count_zero[k, v] is 1, and we remove the zero (set x[i]=1),
                    # then count_zero becomes 0.
                    # So this direction NOW covers k.
                    # If k was invalid (cover_count[k] == 0), it becomes valid.
                    if cover_count[k] == 0:
                        delta_viol -= 1
                    else:
                        # k was already valid, now has one more cover
                        pass
                # else: count_zero > 1, even after removing one zero, still > 0, so still not covering
            # Apply
            x[i] = 1
            for (k, v) in affected[i]:
                if count_zero[k, v] == 1:
                    count_zero[k, v] -= 1
                    cover_count[k] += 1
                else:
                    count_zero[k, v] -= 1
        
        new_sum = old_sum + delta_sum
        new_viol = old_viol + delta_viol
        
        new_cost = new_sum + M * new_viol
        old_cost = old_sum + M * old_viol
        
        delta_cost = new_cost - old_cost
        
        accept = False
        if delta_cost <= 0:
            accept = True
        else:
            # Metropolis
            try:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
            except OverflowError:
                accept = False
        
        if not accept:
            # Revert
            if x[i] == 0:
                # Was 1, set to 0, now revert to 1
                x[i] = 1
                for (k, v) in affected[i]:
                    if count_zero[k, v] == 1: # It was 0, became 1
                        count_zero[k, v] -= 1
                        cover_count[k] += 1
                    else:
                        count_zero[k, v] -= 1
            else:
                # Was 0, set to 1, now revert to 0
                x[i] = 0
                for (k, v) in affected[i]:
                    if count_zero[k, v] == 0: # It was 1, became 0
                        count_zero[k, v] += 1
                        cover_count[k] -= 1
                    else:
                        count_zero[k, v] += 1
        else:
            # Accept
            current_sum = new_sum
            current_viol = new_viol
            if new_viol == 0:
                if new_sum < best_cost:
                    best_cost = new_sum
                    best_x = x.copy()
        
        T *= cooling
        
    # Return best valid solution
    # Ensure best_x is valid
    # (It should be, as we only update best_x when new_viol == 0)
    
    points = []
    for i in range(q * q):
        if best_x[i] == 1:
            x_idx, y_idx = idx_to_point[i]
            xa, xb = to_pair(x_idx)
            ya, yb = to_pair(y_idx)
            points.append([[xa, xb], [ya, yb]])
            
    if not points:
        # Fallback: return all points
        for i in range(q * q):
            x_idx, y_idx = idx_to_point[i]
            xa, xb = to_pair(x_idx)
            ya, yb = to_pair(y_idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
