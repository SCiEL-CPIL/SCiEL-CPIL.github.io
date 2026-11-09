import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill

    # Precompute all arithmetic progressions
    # AP: (start, step) where start in [0, n-1], step >= 1
    # Elements: start, start+step, start+2*step, ... < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for step in range(1, n + 1):
        for start in range(0, n, step):
            idxs = list(range(start, n, step))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    def compute_max_discrepancy(signs, ap_sums):
        """Compute max |sum| over all APs given current ap_sums."""
        max_disc = 0
        for s in ap_sums:
            a = abs(s)
            if a > max_disc:
                max_disc = a
        return max_disc
    
    def initialize_ap_sums(signs):
        """Compute initial AP sums."""
        ap_sums = [0] * num_aps
        for ap_id, idxs in enumerate(aps):
            s = 0
            for idx in idxs:
                s += signs[idx]
            ap_sums[ap_id] = s
        return ap_sums
    
    def flip_and_update(signs, ap_sums, pos):
        """Flip sign at pos and update affected AP sums. Returns old sign."""
        old_sign = signs[pos]
        signs[pos] = -old_sign
        delta = -2 * old_sign  # change in sign value
        for ap_id in index_to_aps[pos]:
            ap_sums[ap_id] += delta
        return old_sign
    
    def thue_morse(n):
        """Generate Thue-Morse-like sequence: -1/+1 based on parity of set bits."""
        result = []
        for i in range(n):
            # Count set bits in i
            c = bin(i).count('1')
            result.append(1 if c % 2 == 0 else -1)
        return result
    
    best_signs = None
    best_max_disc = float('inf')
    
    # Initial solution: Thue-Morse pattern
    signs = thue_morse(n)
    ap_sums = initialize_ap_sums(signs)
    current_max = compute_max_discrepancy(signs, ap_sums)
    
    if current_max < best_max_disc:
        best_max_disc = current_max
        best_signs = signs[:]
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() - start_time < time_limit:
        # Pick a random position to flip
        pos = random.randint(0, n - 1)
        
        # Compute new max discrepancy if we flip
        old_sign = signs[pos]
        delta = -2 * old_sign
        
        # We need to check if flipping improves things
        # Quick check: look at affected APs
        affected = index_to_aps[pos]
        
        # Tentatively flip
        signs[pos] = -old_sign
        for ap_id in affected:
            ap_sums[ap_id] += delta
        
        new_max = compute_max_discrepancy(signs, ap_sums)
        
        if new_max <= current_max:
            # Accept (improvement or equal)
            current_max = new_max
            if current_max < best_max_disc:
                best_max_disc = current_max
                best_signs = signs[:]
        else:
            # Metropolis acceptance
            delta_E = new_max - current_max
            if random.random() < math.exp(-delta_E / max(T, T_min)):
                current_max = new_max
            else:
                # Reject: undo the flip
                signs[pos] = old_sign
                for ap_id in affected:
                    ap_sums[ap_id] -= delta
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # If we have time, try a few more restarts with random initializations
    while time.time() - start_time < time_limit - 5.0:
        signs = [random.choice([-1, 1]) for _ in range(n)]
        ap_sums = initialize_ap_sums(signs)
        current_max = compute_max_discrepancy(signs, ap_sums)
        
        if current_max < best_max_disc:
            best_max_disc = current_max
            best_signs = signs[:]
        
        # Brief annealing from this start
        T = 5.0
        cooling = 0.9999
        steps = 0
        while steps < 50000 and time.time() - start_time < time_limit - 2.0:
            pos = random.randint(0, n - 1)
            old_sign = signs[pos]
            delta = -2 * old_sign
            affected = index_to_aps[pos]
            
            signs[pos] = -old_sign
            for ap_id in affected:
                ap_sums[ap_id] += delta
            
            new_max = compute_max_discrepancy(signs, ap_sums)
            
            if new_max <= current_max:
                current_max = new_max
                if current_max < best_max_disc:
                    best_max_disc = current_max
                    best_signs = signs[:]
            else:
                delta_E = new_max - current_max
                if random.random() < math.exp(-delta_E / max(T, T_min)):
                    current_max = new_max
                else:
                    signs[pos] = old_sign
                    for ap_id in affected:
                        ap_sums[ap_id] -= delta
            
            T *= cooling
            if T < T_min:
                T = T_min
            steps += 1
    
    if best_signs is None:
        best_signs = thue_morse(n)
    
    return best_signs
