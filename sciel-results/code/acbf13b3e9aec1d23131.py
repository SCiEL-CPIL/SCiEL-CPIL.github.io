import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill

    # 1. Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for s in range(n):
            indices = list(range(s, n, d))
            if indices:
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Build impact map: for each index i, which APs contain it?
    impact = [[] for _ in _ in range(n)]
    impact = [[] for _ in range(n)]
    for ap_idx, indices in enumerate(ap_indices):
        for i in indices:
            impact[i].append(ap_idx)

    max_possible = n

    def compute_max_disc(ap_sums):
        """Compute maximum absolute sum from ap_sums array."""
        mx = 0
        for s in ap_sums:
            a = abs(s)
            if a > mx:
                mx = a
        return mx

    def compute_ap_sums(signs):
        """Compute all AP sums given a sign vector."""
        ap_sums = [0] * num_aps
        for ap_idx, indices in enumerate(ap_indices):
            s = 0
            for i in indices:
                s += signs[i]
            ap_sums[ap_idx] = s
        return ap_sums

    def optimize_from_start(signs, time_budget, rng, temperature_start=5.0, cooling_rate=0.9995, min_temp=0.05):
        """Run simulated annealing from a given starting point."""
        ap_sums = compute_ap_sums(signs)
        current_max = compute_max_disc(ap_sums)
        best_signs = signs[:]
        best_disc = current_max
        
        # Build histogram for fast max lookup
        hist = [0] * (max_possible + 1)
        for s in ap_sums:
            hist[abs(s)] += 1
        
        temperature = temperature_start
        no_improve = 0
        improve_threshold = 5000  # Reset no_improve counter periodically
        
        while time.time() - start_time < time_budget:
            # Choose move
            if rng.random() < 0.8:
                # Pick from worst APs
                worst_aps = [k for k in range(num_aps) if abs(ap_sums[k]) == current_max]
                if not worst_aps:
                    idx = rng.randint(0, n - 1)
                else:
                    target_ap = rng.choice(worst_aps)
                    target_indices = ap_indices[target_ap]
                    target_sum = ap_sums[target_ap]
                    
                    candidates = []
                    if target_sum > 0:
                        for i in target_indices:
                            if signs[i] == 1:
                                candidates.append(i)
                    elif target_sum < 0:
                        for i in target_indices:
                            if signs[i] == -1:
                                candidates.append(i)
                    else:
                        candidates = target_indices[:]
                    
                    if candidates:
                        idx = rng.choice(candidates)
                    else:
                        idx = rng.randint(0, n - 1)
            else:
                idx = rng.randint(0, n - 1)
            
            # Perform flip
            old_sign = signs[idx]
            new_sign = -old_sign
            delta_sum = -2 * old_sign
            
            affected_aps = impact[idx]
            old_sums = {}
            old_hist = hist[:]
            
            for k in affected_aps:
                old_s = ap_sums[k]
                old_sums[k] = old_s
                old_abs = abs(old_s)
                hist[old_abs] -= 1
                
                new_s = old_s + delta_sum
                ap_sums[k] = new_s
                new_abs = abs(new_s)
                hist[new_abs] += 1
            
            # Find new max
            new_max = 0
            for j in range(max_possible, -1, -1):
                if hist[j] > 0:
                    new_max = j
                    break
            
            delta_disc = new_max - current_max
            
            # Acceptance
            accepted = False
            if delta_disc < 0:
                accepted = True
            elif delta_disc == 0:
                accepted = True
            else:
                prob = math.exp(-delta_disc / temperature)
                if rng.random() < prob:
                    accepted = True
                
            if accepted:
                signs[idx] = new_sign
                current_max = new_max
                if current_max < best_disc:
                    best_disc = current_max
                    best_signs = signs[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                signs[idx] = old_sign
                hist = old_hist
                for k in affected_aps:
                    ap_sums[k] = old_sums[k]
                no_improve += 1
            
            # Cooling
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
            
            # Occasional restart if stuck
            if no_improve > improve_threshold:
                no_improve = 0
                temperature = max(temperature, temperature_start * 0.5)

        return best_signs, best_disc

    # 2. Multi-start optimization
    rng = random.Random(42)
    
    # Define diverse initializations
    def thue_morse(n):
        return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    def random_signs(n, seed):
        r = random.Random(seed)
        return [r.choice([-1, 1]) for _ in range(n)]
    
    def periodic_signs(n, period, offset=0):
        """Create a periodic pattern."""
        base = [1 if (i % period) < period // 2 else -1 for i in range(period)]
        return [base[(i + offset) % period] for i in range(n)]
    
    def structured_signs(n):
        """Use a structured pattern based on prime factorization."""
        signs = []
        for i in range(n):
            # Use the 2-adic valuation pattern
            v2 = 0
            x = i + 1
            while x % 2 == 0 and x > 0:
                v2 += 1
                x //= 2
            signs.append(1 if v2 % 2 == 0 else -1)
        return signs
    
    # Generate diverse starting points
    initializations = [
        thue_morse(n),
        random_signs(n, 1),
        random_signs(n, 2),
        random_signs(n, 3),
        periodic_signs(n, 4, 0),
        periodic_signs(n, 4, 1),
        periodic_signs(n, 8, 0),
        periodic_signs(n, 16, 0),
        structured_signs(n),
    ]
    
    # Shuffle and select a subset for time budget
    rng.shuffle(initializations)
    
    best_signs = None
    best_disc = float('inf')
    
    # Allocate time across starts
    num_starts = min(len(initializations), 8)
    time_per_start = (time_limit - 2.0) / num_starts  # Leave 2s buffer
    
    for start_idx in range(num_starts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 5.0:
            break
        
        # Use slightly less than full time per start to allow for next
        alloc = min(time_per_start, remaining - 2.0)
        if alloc < 3.0:
            break
            
        init_signs = initializations[start_idx][:]
        
        # Quick evaluation
        ap_sums = compute_ap_sums(init_signs)
        disc = compute_max_disc(ap_sums)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = init_signs[:]
        
        # Optimize from this start
        result_signs, result_disc = optimize_from_start(
            init_signs[:], 
            start_time + elapsed + alloc,
            rng
        )
        
        if result_disc < best_disc:
            best_disc = result_disc
            best_signs = result_signs[:]
        
        # Check if we should continue
        if time.time() - start_time > time_limit - 5.0:
            break

    # Final safety: ensure we have a valid result
    if best_signs is None:
        best_signs = thue_morse(n)
    
    return best_signs
