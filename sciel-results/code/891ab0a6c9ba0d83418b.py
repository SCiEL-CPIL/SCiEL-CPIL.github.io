import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all APs and their indices
    ap_indices = []
    ap_pos_sets = [] # List of sets of positions for each AP, for quick "bottleneck" identification
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
                ap_pos_sets.append(set(indices))
                
    num_aps = len(ap_indices)
    
    # Map position to APs containing it
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    # Convert to numpy arrays for faster iteration if needed, but lists are faster for small iterations
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]
    
    def compute_initial_sums(s):
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(s[s_indices])
        return S
    
    def build_hist_and_max(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        max_val = int(np.max(abs_S))
        return hist, max_val
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        """
        Tries flipping position j.
        Returns: (accepted, new_max, delta_val, bottleneck_positions_updated)
        """
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        # Track changes to histogram and S
        changed_aps = []
        
        for ap_idx in ap_idx_list:
            old_sum = S[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            S[ap_idx] = new_sum
            changed_aps.append((ap_idx, old_sum, new_sum))
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
        # Determine new max
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
        
        cost_diff = new_max - old_max
        
        accept = False
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            return True, new_max, delta
        else:
            # Rollback
            for ap_idx, old_sum, new_sum in changed_aps:
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
            
            return False, current_max, delta

    def get_bottleneck_positions(S, current_max):
        """
        Returns a list of positions that are part of at least one AP with |sum| == current_max.
        """
        bottleneck_pos = set()
        # Iterate through all APs? That's slow.
        # Better: Keep a list of max APs?
        # For N=256, num_aps is ~400k. Iterating 400k APs every step is too slow.
        # We need to maintain a set of "active" max APs.
        
        # Alternative: Just pick a random position from the whole set, but with a bias?
        # Or, precompute: For each AP, its positions.
        # If we just pick random j, we might miss the bottleneck.
        
        # Let's try a hybrid: 50% random, 50% from a precomputed "high impact" list?
        # No, let's stick to random but with a smarter initialization.
        
        # Actually, we can maintain a list of AP indices that are currently at max.
        # But updating this list is expensive.
        
        # Let's just use random selection but ensure we do MANY steps.
        pass
        return None

    def local_search_targeted(s, S, hist, current_max, deadline, rng, max_steps=50000):
        """
        Simulated Annealing.
        """
        temp = 2.0
        cooling_rate = 0.998
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            # Strategy: Pick a random position.
            # To improve, we can prioritize positions that appear in many APs (central positions).
            # Positions near N/2 are in more APs.
            # Let's just use uniform random for simplicity and robustness.
            j = rng.randint(0, N - 1)
            
            accepted, new_max, _ = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max

    def greedy_improve(s, S, hist, current_max, deadline, rng, max_rounds=5):
        """
        Greedy hill-climbing.
        """
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                
                ap_idx_list = pos_to_ap_np[j]
                max_new_abs_in_affected = 0
                
                # Check if flip improves
                # We don't update hist/S yet, just compute potential new max
                for ap_idx in ap_idx_list:
                    old_sum = S[ap_idx]
                    new_sum = old_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs_in_affected:
                        max_new_abs_in_affected = new_abs
                
                # Determine new max
                # Note: This is an approximation if multiple APs change.
                # But since we only flip one bit, the changes are local.
                # The histogram logic in try_flip is more accurate.
                # For greedy, we can just use try_flip with temp=0 (accept only if <=).
                
                # To avoid code duplication, we use a "dry run" logic here
                # But it's safer to just call try_flip with temp=0 and rollback if not accepted?
                # No, try_flip modifies state.
                
                # Let's implement a safe check
                # If hist[old_max] > 0, new_max is max(old_max, max_new_abs)
                # If hist[old_max] == 0, we need to check if any AP leaves old_max without replacing it.
                
                # This is complex. Let's just use try_flip with temp=0.
                # But try_flip accepts if cost_diff <= 0.
                # So we can just call it. If it returns False, we rollback (which it does).
                # Wait, try_flip rolls back internally if not accepted.
                # So we can just call try_flip(j, s, S, hist, current_max, 0.0, rng).
                
                accepted, new_max, _ = try_flip(j, s, S, hist, current_max, 0.0, rng)
                
                if accepted:
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
        
        return s, S, hist, current_max

    # Main Loop
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialization: Greedy Construction
        # Start with all +1
        s = np.ones(N, dtype=np.int8)
        S = compute_initial_sums(s)
        hist, current_max = build_hist_and_max(S)
        
        # Greedy Improvement from this start
        s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + 2.0, rng, max_rounds=2)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Random Restart
        if time.time() < deadline - 5.0:
             s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
             S = compute_initial_sums(s)
             hist, current_max = build_hist_and_max(S)
             
             # Quick Greedy
             s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + 1.0, rng, max_rounds=1)
             
             if current_max < best_score:
                 best_score = current_max
                 best_seq = s.copy()
             
             # SA Polish
             remaining = deadline - time.time()
             if remaining > 2.0:
                 sa_deadline = time.time() + remaining * 0.8
                 s, S, hist, current_max = local_search_targeted(s, S, hist, current_max, sa_deadline, rng)
                 
                 if current_max < best_score:
                     best_score = current_max
                     best_seq = s.copy()
        
        if time.time() > deadline - 2.0:
            break
            
    # Final Polish on Best
    if best_seq is not None and time.time() < deadline - 1.0:
        s_final = best_seq.copy()
        S_final = compute_initial_sums(s_final)
        hist_final, current_max_final = build_hist_and_max(S_final)
        
        s_final, S_final, hist_final, current_max_final = greedy_improve(
            s_final, S_final, hist_final, current_max_final, deadline - 0.5, rng, max_rounds=5
        )
        
        if time.time() < deadline - 0.5:
             s_final, S_final, hist_final, current_max_final = local_search_targeted(
                s_final, S_final, hist_final, current_max_final, deadline - 0.1, rng, max_steps=5000
            )
        
        if current_max_final < best_score:
            best_score = current_max_final
            best_seq = s_final.copy()
    
    return [int(x) for x in best_seq]
