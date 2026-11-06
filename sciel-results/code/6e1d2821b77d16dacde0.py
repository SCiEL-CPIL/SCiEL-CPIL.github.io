import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute all 4-APs as tuples of 4 indices
    # We need to know, for any move, how many 4-APs are completed or broken.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # Let's store for each index i, the list of 4-APs it is part of.
    # ap_list[i] = list of (j, k, l) such that {i,j,k,l} is a 4-AP.
    # Actually, to calculate delta energy efficiently:
    # When adding x:
    #   New violations = number of APs containing x where the other 3 are present.
    # When removing x:
    #   Removed violations = number of APs containing x where the other 3 are present (before removal).
    
    # Let's precompute neighbors for each index in the hypergraph of 4-APs.
    # For each i, store the list of other triples.
    # conflicts[i] = list of (j, k, l)
    
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            i, j, k, l = a-1, b-1, c-1, e-1
            
            # Store in all 4 lists
            conflicts[i].append((j, k, l))
            conflicts[j].append((i, k, l))
            conflicts[k].append((i, j, l))
            conflicts[l].append((i, j, k))

    # Helper to check if a set of 3 indices is present in bits
    # We will use bit operations for speed.
    # But checking 3 bits individually might be faster than creating masks on the fly?
    # Let's precompute masks for the triples to make it a single bitwise AND.
    # masks[i] = list of masks for the triples in conflicts[i]
    
    triple_masks = [[] for _ in range(N)]
    for i in range(N):
        for j, k, l in conflicts[i]:
            mask = (1 << j) | (1 << k) | (1 << l)
            triple_masks[i].append(mask)

    def count_violations(bits, x):
        """Count how many 4-APs involving x are fully present in bits."""
        count = 0
        for mask in triple_masks[x]:
            if (bits & mask) == mask:
                count += 1
        return count

    # Simulated Annealing
    # State: bits
    # Energy: -size (we want to minimize -size, i.e., maximize size)
    # Or simply: Objective = Size. We want to maximize.
    # Standard SA minimizes Energy. Let E = -Size.
    # Move: Add x (Size +1, E -1) or Remove x (Size -1, E +1).
    # Constraint: We don't strictly require validity at each step, but we penalize violations?
    # The task requires the FINAL result to be valid.
    # If we allow invalid intermediate states, we must ensure the final state is valid.
    # Strategy: 
    # 1. Run SA to maximize a score: Score = Size - Penalty * Violations.
    # 2. At the end, if Violations > 0, repair by removing elements from violating APs.
    # 3. This allows SA to explore "dense" regions that might be slightly invalid, 
    #    then clean them up. This often finds larger valid sets than strict validity enforcement.
    
    # Let's define the cost function for SA:
    # Cost = -Size + Lambda * Violations
    # We want to minimize Cost.
    # Lambda should be large enough to discourage violations but small enough to allow exploration.
    # Or, simpler: 
    # We perform moves:
    # 1. Pick random x.
    # 2. If x in S: Remove x.
    #    Delta Size = -1.
    #    Delta Violations = - (number of APs containing x that were fully present before removal).
    #    New Cost - Old Cost = 1 + Lambda * (-Delta_Viol)
    # 3. If x not in S: Add x.
    #    Delta Size = +1.
    #    Delta Violations = + (number of APs containing x where other 3 are present).
    #    New Cost - Old Cost = -1 + Lambda * Delta_Viol
    
    # We need to track the current number of violations for fast delta calculation?
    # No, we just calculate the local change.
    
    # Initialization
    # Start with a random valid set? Or empty?
    # Let's start with a random set and repair it to be valid, then run SA.
    # Or start empty and add.
    
    # Let's try starting with a random set of density 0.4, repair it, then SA.
    
    def repair(bits):
        """Remove elements to make bits valid."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(N):
                if time.time() > deadline:
                    break
                if not (bits & (1 << i)):
                    continue
                # Check if i is in any violation
                # We need to know if i is part of a complete 4-AP.
                # This is expensive to check from scratch.
                # Better: iterate all APs?
                # Let's just iterate all APs and remove the last element if present.
                # This is O(N^2) per pass, which might be slow.
                # Alternative: For each i, check its conflicts.
                for mask in triple_masks[i]:
                    if (bits & mask) == mask:
                        # i is part of a violation. Remove i.
                        bits &= ~(1 << i)
                        changed = True
                        break
        return bits

    # Better repair:
    # Iterate through all APs. If all 4 present, remove one.
    # To be faster, we can precompute the list of APs.
    all_aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_aps.append((a-1, b-1, c-1, e-1))

    def repair_fast(bits):
        """Remove elements to make bits valid."""
        # We'll just loop through APs. If all present, remove the last one.
        # To avoid O(N^2) full scans repeatedly, we can just do a few passes.
        for _ in range(10): # Max 10 passes
            if time.time() > deadline:
                break
            changed = False
            for i, j, k, l in all_aps:
                mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                if (bits & mask) == mask:
                    bits &= ~(1 << l)
                    changed = True
            if not changed:
                break
        return bits

    # Simulated Annealing Loop
    # We will run multiple restarts.
    
    best_bits = 0
    best_size = 0
    
    # Lambda: Penalty for violations.
    # If Lambda is too high, we stay valid always (like local search).
    # If Lambda is low, we allow violations.
    # Let's try Lambda = 1.0. Cost = -Size + Violations.
    # Move add x: dCost = -1 + dViol.
    # Move rem x: dCost = +1 - dViol.
    
    Lambda = 1.0
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Init
        if r == 0:
            # Start with ternary-based set, repair it
            init_bits = 0
            for num in range(1, N + 1):
                x = num
                valid = True
                while x > 0:
                    if x % 3 == 0:
                        valid = False
                        break
                    x //= 3
                if valid:
                    init_bits |= (1 << (num - 1))
            bits = repair_fast(init_bits)
        else:
            # Random start
            bits = 0
            for i in range(N):
                if random.random() < 0.3:
                    bits |= (1 << i)
            bits = repair_fast(bits)
        
        # SA Parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        current_bits = bits
        current_size = bin(current_bits).count('1')
        # Calculate initial violations
        current_violations = 0
        for i, j, k, l in all_aps:
            mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
            if (current_bits & mask) == mask:
                current_violations += 1
        
        current_cost = -current_size + Lambda * current_violations
        
        best_in_sa_bits = current_bits
        best_in_sa_size = current_size
        best_in_sa_cost = current_cost
        
        while time.time() < deadline and T > min_T:
            # Pick random move
            x = random.randint(0, N - 1)
            
            if current_bits & (1 << x):
                # Remove x
                # Delta Size = -1
                # Delta Violations = - (number of APs containing x that are fully present)
                # Note: "fully present" includes x. So we check if other 3 are present.
                v_count = 0
                for mask in triple_masks[x]:
                    if (current_bits & mask) == mask:
                        v_count += 1
                
                delta_size = -1
                delta_viol = -v_count
                delta_cost = -delta_size + Lambda * delta_viol
                
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                    current_bits &= ~(1 << x)
                    current_size += delta_size
                    current_violations += delta_viol
                    current_cost += delta_cost
                    
                    # Update best if valid
                    if current_violations == 0:
                        if current_size > best_in_sa_size:
                            best_in_sa_size = current_size
                            best_in_sa_bits = current_bits
                            best_in_sa_cost = current_cost
            else:
                # Add x
                # Delta Size = +1
                # Delta Violations = + (number of APs containing x where other 3 are present)
                v_count = 0
                for mask in triple_masks[x]:
                    if (current_bits & mask) == mask:
                        v_count += 1
                
                delta_size = 1
                delta_viol = v_count
                delta_cost = -delta_size + Lambda * delta_viol
                
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                    current_bits |= (1 << x)
                    current_size += delta_size
                    current_violations += delta_viol
                    current_cost += delta_cost
                    
                    # Update best if valid
                    if current_violations == 0:
                        if current_size > best_in_sa_size:
                            best_in_sa_size = current_size
                            best_in_sa_bits = current_bits
                            best_in_sa_cost = current_cost
            
            T *= cooling
        
        # After SA, repair the best found state if it has violations
        # Actually, we tracked best_in_sa which was only updated if valid.
        # So best_in_sa_bits should be valid.
        
        if best_in_sa_size > best_size:
            best_size = best_in_sa_size
            best_bits = best_in_sa_bits

    # Final sanity check and repair
    final_bits = repair_fast(best_bits)
    final_size = bin(final_bits).count('1')
    
    result = [i + 1 for i in range(N) if (final_bits >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
