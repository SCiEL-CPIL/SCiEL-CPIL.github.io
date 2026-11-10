import numpy as np

def construct():
    n = 256
    
    # 1. Enumerate all Arithmetic Progressions
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for a in range(n):
            k = 1
            while a + (k - 1) * d < n:
                ap = list(range(a, a + k * d, d))
                ap_idx = len(aps)
                aps.append(ap)
                for idx in ap:
                    index_to_aps[idx].append(ap_idx)
                k += 1
                
    num_aps = len(aps)
    
    # 2. Initialize signs
    # Using Thue-Morse sequence as a good initial guess for low discrepancy
    def thue_morse(n):
        t = [1]
        for _ in range(n.bit_length()):
            t += [-x for x in t]
        return t[:n]
        
    signs = np.array(thue_morse(n), dtype=int)
    
    # Calculate initial sums
    ap_sums = np.zeros(num_aps, dtype=int)
    for i, ap in enumerate(aps):
        ap_sums[i] = np.sum(signs[ap])
        
    current_max = np.max(np.abs(ap_sums))
    
    # 3. Local Search
    # We will try to reduce the maximum absolute sum
    # Loop until no improvement or max steps
    max_iterations = 5000
    no_improve_count = 0
    limit_no_improve = 50
    
    for _ in range(max_iterations):
        if current_max <= 1:
            break
            
        # Find the AP with the maximum absolute sum
        abs_sums = np.abs(ap_sums)
        worst_idx = np.argmax(abs_sums)
        worst_ap = aps[worst_idx]
        worst_sum = ap_sums[worst_idx]
        
        found_improvement = False
        
        # Determine which signs to flip to reduce the absolute sum
        # If worst_sum > 0, flip +1 to -1
        # If worst_sum < 0, flip -1 to +1
        target_sign = 1 if worst_sum > 0 else -1
        
        # Collect candidates: indices in worst_ap with sign == target_sign
        candidates = [i for i in worst_ap if signs[i] == target_sign]
        
        # Shuffle candidates to avoid bias? Or just iterate.
        # Let's iterate.
        for i in candidates:
            s_i = signs[i]
            # Flip i
            signs[i] = -s_i
            
            # Update sums for all APs containing i
            affected_aps = index_to_aps[i]
            ap_sums[affected_aps] -= 2 * s_i
            
            # Check new max
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                found_improvement = True
                # We can break here or continue to try to improve further?
                # Greedy: take the first improvement.
                break
            else:
                # Revert
                ap_sums[affected_aps] += 2 * s_i
                signs[i] = s_i
        
        if found_improvement:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > limit_no_improve:
                # Try a random restart or perturbation?
                # For now, break.
                # A simple perturbation: flip a random sign and see if it helps later?
                # Let's just break to save time.
                break
                
    return signs.tolist()
