import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Precompute APs
    # We only consider APs with length >= 2.
    # Length 1 APs have sum +/- 1, so max is at least 1.
    # We minimize max_abs_sum for len >= 2.
    # Final score = max(1, min_max_abs_sum)
    
    ap_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Map position to APs
    pos_to_aps = [[] for _ in range(N)]
    for ap_id, terms in enumerate(ap_indices):
        for idx in terms:
            pos_to_aps[idx].append(ap_id)
            
    # Helper to compute initial sums
    def compute_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, terms in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[terms])
        return sums

    # Helper to compute hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))

    # Soft Surrogate SA
    # Objective: F = beta * log(sum(exp(|s_j| / beta)))
    # Note: Using |s_j| makes it symmetric and focuses on magnitude.
    
    best_signs = None
    best_hard_max = N # Worst case
    
    # Try a few different strategies/restarts
    # Strategy: Soft Surrogate Simulated Annealing
    
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums = compute_sums(signs)
        
        # Initial Hard Max
        current_hard_max = get_hard_max(sums)
        
        # Parameters for SA
        # Beta controls the sharpness of the log-sum-exp
        # T controls the acceptance of worse moves
        beta = 2.0
        T = 5.0
        
        # Initial Z = sum(exp(|s_j|/beta))
        # To avoid overflow, we can normalize or use log-sum-exp trick, 
        # but sums are small (max 256), exp(256/2) is huge.
        # We should use a stable log-sum-exp or scale.
        # Actually, for comparison, we only need Delta F.
        # Delta F = beta * (log(Z_new) - log(Z_old))
        # We can maintain log_Z = log(sum(exp(|s_j|/beta)))
        
        def calc_log_Z(sums, beta):
            # Stable log sum exp
            max_val = np.max(np.abs(sums)) / beta
            if max_val == 0:
                return np.log(num_aps)
            # shift by max_val
            shifted = np.abs(sums) / beta - max_val
            sum_exp = np.sum(np.exp(shifted))
            return max_val + np.log(sum_exp)
            
        log_Z = calc_log_Z(sums, beta)
        
        # SA Loop
        steps_per_restart = 0
        max_steps = 50000 # Adjust based on time
        
        while time.time() < deadline - 2:
            steps_per_restart += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Identify affected APs
            affected_aps = pos_to_aps[i]
            
            # Calculate change in log_Z
            # Z = sum_j exp(|S_j|/beta)
            # Z_new = Z - sum_{j in affected} exp(|S_j|/beta) + sum_{j in affected} exp(|S_j - 2*old_sign|/beta)
            
            # To be fast, we sum the terms
            old_terms_sum = 0.0
            new_terms_sum = 0.0
            
            # We need current S_j for affected APs
            # And new S_j = S_j - 2 * old_sign
            
            for ap_id in affected_aps:
                s_old = sums[ap_id]
                s_new = s_old - 2 * old_sign
                
                # Use exp(s/beta)
                # To avoid overflow in exp, we can factor out a common term?
                # Or just use math.exp if values are small.
                # Max sum is 256. beta=2. 256/2 = 128. exp(128) is ~1e55.
                # This is within float64 range (1e308). So direct exp is safe.
                
                old_terms_sum += math.exp(abs(s_old) / beta)
                new_terms_sum += math.exp(abs(s_new) / beta)
                
            # We need Z_old. We have log_Z_old.
            # Z_old = exp(log_Z_old)
            # This might be huge.
            # Instead of computing Z_old, we can compute the ratio Z_new/Z_old
            # Z_new = Z_old - old_terms_sum + new_terms_sum
            # Ratio = 1 - old_terms_sum/Z_old + new_terms_sum/Z_old
            
            # Let's compute Z_old explicitly?
            # If log_Z is large, Z is large.
            # Let's check the magnitude.
            # If we use beta=2, and sums are small, it's fine.
            
            # Alternative: Maintain Z directly?
            # Z might overflow.
            # Let's stick to log domain for the objective value F, 
            # but for the update, we need the relative change.
            
            # Let's try to maintain Z as a float.
            # If Z overflows, we can rescale.
            # But let's assume it fits.
            
            Z_old = math.exp(log_Z)
            Z_new = Z_old - old_terms_sum + new_terms_sum
            
            if Z_new <= 0:
                # Should not happen, but safeguard
                continue
                
            log_Z_new = math.log(Z_new)
            delta_F = beta * (log_Z_new - log_Z)
            
            # Metropolis Criterion
            # Accept if delta_F < 0 or with prob exp(-delta_F / T)
            accept = False
            if delta_F < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_F / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_F / T)
            
            if accept:
                # Apply move
                signs[i] = new_sign
                for ap_id in affected_aps:
                    sums[ap_id] -= 2 * old_sign
                log_Z = log_Z_new
                
                # Check hard max periodically to update best
                if steps_per_restart % 100 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            # Decrease T and beta slowly
            # beta should decrease to sharpen the approximation of max
            # T should decrease to freeze the solution
            if steps_per_restart % 1000 == 0:
                T *= 0.95
                beta *= 0.98
                if beta < 0.5:
                    beta = 0.5
                if T < 0.1:
                    T = 0.1
                    
            # Early stop if we hit 1 (optimal for len>=2 is 0 or 1? 
            # If max abs sum is 0, total max is 1.
            # If max abs sum is 1, total max is 1.
            # So if we reach 1, we can stop improving this restart.
            # But we might find 0?
            # For N=256, is max abs sum 0 possible?
            # Alternating 1,-1 gives 0 for d=1.
            # For d=2, 1,-1,1,-1... sums are 0.
            # It is theoretically possible to have max abs sum 0 for all APs?
            # This is a very strong condition.
            # If we reach 1, we are likely at the global minimum (since 0 is hard).
            # Actually, if max abs sum is 1, the score is max(1, 1) = 1.
            # If we can get max abs sum 0, score is max(1, 0) = 1.
            # So score 1 is the theoretical minimum.
            
            if best_hard_max <= 1:
                break
                
    # If no best_signs found (should not happen), return random
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
