import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all APs (length >= 2)
    aps_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                aps_indices.append(terms)
    
    num_aps = len(aps_indices)
    
    # Build membership matrix: M[ap_idx, position] = 1 if position in AP
    M = np.zeros((num_aps, N), dtype=np.int8)
    for ap_idx, terms in enumerate(aps_indices):
        for pos in terms:
            M[ap_idx, pos] = 1
    
    # For each position, which APs contain it (as numpy array of indices)
    pos_to_aps = []
    for pos in range(N):
        pos_to_aps.append(np.where(M[:, pos] == 1)[0])
    
    # Precompute: for each position, the AP indices and the column of M
    # This allows fast delta computation
    
    def compute_ap_sums(signs):
        """Compute all AP sums from signs vector."""
        return M @ signs.astype(np.int16)
    
    def get_max_abs(ap_sums):
        """Get max absolute value of ap_sums."""
        return int(np.max(np.abs(ap_sums)))
    
    def greedy_init(rng):
        """Greedy initialization: place signs one by one, minimizing current max."""
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_sums = None
            
            for s in [1, -1]:
                # Compute new sums
                new_sums = ap_sums.copy()
                # Add s to all APs containing position i
                aps_i = pos_to_aps[i]
                if len(aps_i) > 0:
                    new_sums[aps_i] += s
                
                m = get_max_abs(new_sums)
                
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_sums = new_sums
            
            signs[i] = best_sign
            ap_sums = best_sums
        
        return signs, ap_sums, best_max
    
    def try_2flip(signs, ap_sums, current_max, rng, critical_aps=None):
        """Try a random 2-flip move. Returns (accepted, new_signs, new_ap_sums, new_max)."""
        i = rng.randint(0, N - 1)
        j = rng.randint(0, N - 1)
        
        if i == j:
            return False, signs, ap_sums, current_max
        
        # Determine which APs are affected
        aps_i = pos_to_aps[i]
        aps_j = pos_to_aps[j]
        
        # Find union of affected APs
        if critical_aps is not None:
            # Only consider critical APs
            crit_set = set(critical_aps)
            affected = [a for a in aps_i if a in crit_set]
            for a in aps_j:
                if a in crit_set and a not in affected:
                    affected.append(a)
            if len(affected) == 0:
                return False, signs, ap_sums, current_max
            affected_arr = np.array(affected, dtype=np.int32)
        else:
            # Union of aps_i and aps_j
            if len(aps_i) == 0:
                affected_arr = aps_j
            elif len(aps_j) == 0:
                affected_arr = aps_i
            else:
                # Use set union
                set_i = set(aps_i)
                set_j = set(aps_j)
                union = set_i | set_j
                affected_arr = np.array(list(union), dtype=np.int32)
        
        if len(affected_arr) == 0:
            return False, signs, ap_sums, current_max
        
        # Compute deltas
        di = -2 * signs[i]
        dj = -2 * signs[j]
        
        # For each affected AP, compute new sum
        old_sums = ap_sums[affected_arr].copy()
        
        # Determine if i is in each affected AP
        i_in_affected = np.array([1 if a in set(aps_i) else 0 for a in affected_arr], dtype=np.int8)
        j_in_affected = np.array([1 if a in set(aps_j) else 0 for a in affected_arr], dtype=np.int8)
        
        # New sums
        new_sums_affected = old_sums + i_in_affected * di + j_in_affected * dj
        
        # Check if this improves or maintains the max
        new_max_affected = get_max_abs(new_sums_affected)
        
        # The overall new max is max(new_max_affected, max of unaffected APs)
        # For efficiency, just check if new_max_affected <= current_max
        # If it does, the overall max is at most current_max (might be less if critical APs changed)
        
        if new_max_affected > current_max:
            return False, signs, ap_sums, current_max
        
        # Accept the move
        signs[i] = -signs[i]
        signs[j] = -signs[j]
        ap_sums[affected_arr] = new_sums_affected
        
        # Compute new overall max
        new_max = get_max_abs(ap_sums)
        
        return True, signs, ap_sums, new_max
    
    def local_search(signs, ap_sums, current_max, rng, time_limit, critical_aps=None):
        """Local search with 2-flip moves."""
        no_improve_count = 0
        max_no_improve = 5000 if critical_aps is not None else 2000
        
        while time.time() < time_limit:
            accepted, signs, ap_sums, new_max = try_2flip(
                signs, ap_sums, current_max, rng, critical_aps
            )
            
            if accepted:
                if new_max < current_max:
                    current_max = new_max
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
        
        return signs, ap_sums, current_max
    
    # Main search loop
    rng = random.Random(42)
    best_val = N + 1
    best_signs = None
    
    while time.time() < deadline:
        # Greedy init
        signs, ap_sums, current_max = greedy_init(rng)
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
        
        if best_val <= 2:
            break
        
        # Phase 1: Critical-AP local search (fast)
        # Identify critical APs (those with |sum| close to current_max)
        if time.time() < deadline - 10.0:
            abs_sums = np.abs(ap_sums)
            # APs with |sum| >= current_max - 1 are critical
            threshold = max(0, current_max - 1)
            critical_aps = np.where(abs_sums >= threshold)[0].tolist()
            
            # Limit critical APs for speed
            if len(critical_aps) > 5000:
                # Randomly sample
                critical_aps = rng.sample(critical_aps, 5000)
            
            time_limit_1 = min(deadline - 5.0, start_time + 25.0)
            signs, ap_sums, current_max = local_search(
                signs, ap_sums, current_max, rng, time_limit_1, critical_aps
            )
        
        # Phase 2: Full-AP local search (polish)
        if time.time() < deadline - 2.0:
            time_limit_2 = deadline - 2.0
            signs, ap_sums, current_max = local_search(
                signs, ap_sums, current_max, rng, time_limit_2, critical_aps=None
            )
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
        
        if best_val <= 2:
            break
    
    return best_signs.tolist()
