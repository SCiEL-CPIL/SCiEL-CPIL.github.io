import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 128
    target_lo = 128
    target_hi = 256
    targets = np.arange(target_lo, target_hi + 1)
    
    def compute_counts(A_set):
        counts = np.zeros(257, dtype=np.int32)
        A_list = sorted(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            for j in range(i, m):
                counts[ai + aj] += 1
        return counts

    def get_metrics(counts):
        seg = counts[target_lo:target_hi + 1]
        peak = int(np.max(seg))
        uncovered = int(np.sum(seg == 0))
        ss = int(np.sum(seg ** 2))
        return peak, uncovered, ss

    # Start with full set
    A = set(range(0, n + 1))
    counts = compute_counts(A)
    peak, unc, ss = get_metrics(counts)
    
    best_A = set(A)
    best_peak = peak
    best_ss = ss
    
    temperature = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        improved = False
        move_type = random.random()
        
        if move_type < 0.5 and len(A) > 25:
            # Try Removal
            x = random.choice(list(A))
            
            new_counts = counts.copy()
            for y in A:
                if y != x:
                    new_counts[x + y] -= 1
            new_counts[2 * x] -= 1
            
            new_peak, new_unc, new_ss = get_metrics(new_counts)
            
            if new_unc == 0:
                delta_peak = new_peak - peak
                delta_ss = new_ss - ss
                
                if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                    A.remove(x)
                    counts = new_counts
                    peak, unc, ss = new_peak, new_unc, new_ss
                    improved = True
                    if peak < best_peak or (peak == best_peak and ss < best_ss):
                        best_peak = peak
                        best_ss = ss
                        best_A = set(A)
                else:
                    delta = delta_peak * 10 + delta_ss
                    if delta > 0 and random.random() < np.exp(-delta / max(temperature, 0.01)):
                        A.remove(x)
                        counts = new_counts
                        peak, unc, ss = new_peak, new_unc, new_ss
                        improved = True
            else:
                # Invalid: Try Repair Swap
                unc_targets = [t for t in targets if new_counts[t] == 0]
                
                # Find y to add
                # y must cover unc_targets
                # y + z = t => z = t - y
                # z must be in A \ {x}
                
                best_y = None
                best_score = float('inf')
                
                # Sample candidates
                candidates = [i for i in range(n + 1) if i not in A]
                if candidates:
                    # Prioritize candidates that might cover unc_targets
                    # Just sample
                    sample_size = min(50, len(candidates))
                    for y in random.sample(candidates, sample_size):
                        # Check if y covers all unc_targets
                        valid = True
                        for t in unc_targets:
                            z = t - y
                            if z < 0 or z > n or z not in A or z == x:
                                valid = False
                                break
                            # Also if z == y, we need y in A (it is being added)
                            # But z must be in the *remaining* set A\{x} OR z==y
                            # If z==y, then 2y=t. The pair (y,y) is formed by adding y.
                            # So it's valid.
                            # But my check `z not in A` fails if z==y and y not in A.
                            # So:
                            if z == y:
                                continue
                            if z not in A:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Calculate new metrics for swap x->y
                        trial_counts = new_counts.copy()
                        A_trial = set(A)
                        A_trial.remove(x)
                        
                        for z in A_trial:
                            trial_counts[y + z] += 1
                        trial_counts[2 * y] += 1
                        
                        t_peak, t_unc, t_ss = get_metrics(trial_counts)
                        
                        if t_unc == 0:
                            # Score: we want to minimize peak, then ss
                            # Compare to current peak/ss
                            # If better, accept
                            if t_peak < peak or (t_peak == peak and t_ss < ss):
                                # Accept
                                A.remove(x)
                                A.add(y)
                                counts = trial_counts
                                peak, unc, ss = t_peak, t_unc, t_ss
                                improved = True
                                if peak < best_peak or (peak == best_peak and ss < best_ss):
                                    best_peak = peak
                                    best_ss = ss
                                    best_A = set(A)
                                break
                            
        elif move_type < 0.8:
            # Try Swap (x in A, y not in A)
            x = random.choice(list(A))
            candidates = [i for i in range(n + 1) if i not in A]
            if candidates:
                y = random.choice(candidates)
                
                # Remove x
                new_counts = counts.copy()
                for z in A:
                    if z != x:
                        new_counts[x + z] -= 1
                new_counts[2 * x] -= 1
                
                # Add y
                A_trial = set(A)
                A_trial.remove(x)
                for z in A_trial:
                    new_counts[y + z] += 1
                new_counts[2 * y] += 1
                
                new_peak, new_unc, new_ss = get_metrics(new_counts)
                
                if new_unc == 0:
                    delta_peak = new_peak - peak
                    delta_ss = new_ss - ss
                    
                    if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                        A.remove(x)
                        A.add(y)
                        counts = new_counts
                        peak, unc, ss = new_peak, new_unc, new_ss
                        improved = True
                        if peak < best_peak or (peak == best_peak and ss < best_ss):
                            best_peak = peak
                            best_ss = ss
                            best_A = set(A)
                    else:
                        delta = delta_peak * 10 + delta_ss
                        if delta > 0 and random.random() < np.exp(-delta / max(temperature, 0.01)):
                            A.remove(x)
                            A.add(y)
                            counts = new_counts
                            peak, unc, ss = new_peak, new_unc, new_ss
                            improved = True
                            
        if improved:
            temperature *= cooling
        else:
            temperature *= 0.99
            if temperature < 0.05:
                A = set(best_A)
                counts = compute_counts(A)
                peak, unc, ss = get_metrics(counts)
                temperature = 1.0

    return sorted(list(best_A))
