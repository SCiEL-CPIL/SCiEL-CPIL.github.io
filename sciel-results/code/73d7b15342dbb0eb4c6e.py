import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def compute_sums_and_conflicts(S):
        n = len(S)
        # Bitset DP
        # sums[k] has bit s set if sum s is possible with k elements
        sums = [0] * (n + 1)
        sums[0] = 1  # Sum 0 with 0 elements
        
        for x in S:
            # Update backwards
            for k in range(n, 0, -1):
                sums[k] |= (sums[k-1] << x)
                
        # Count conflicts
        # Conflict if sums[k] & sums[j] != 0 for k != j
        conflicts = 0
        for k in range(n + 1):
            for j in range(k + 1, n + 1):
                conflicts += (sums[k] & sums[j]).bit_count()
                
        return sums, conflicts

    best_size = 0
    best_set = []
    rng = random.Random(42)
    
    all_vals = list(range(1, 257))
    
    # Try different starting sizes
    # Larger start sizes might get stuck in high conflict areas
    # Smaller start sizes might grow slowly
    # Let's try a range of start sizes
    
    for start_size in [20, 25, 30, 35, 40]:
        for trial in range(100): # 500 trials total
            if time.time() > deadline - 5:
                break
                
            # Initialize S
            S = rng.sample(all_vals, start_size)
            sums, conflicts = compute_sums_and_conflicts(S)
            current_size = start_size
            
            # Local Search
            while time.time() < deadline - 2:
                if conflicts == 0:
                    # Try to grow
                    # Sample candidates from unused
                    unused = [v for v in all_vals if v not in S]
                    if not unused:
                        break
                    
                    # Try a few random additions
                    added = False
                    # Shuffle unused to pick random samples
                    rng.shuffle(unused)
                    candidates = unused[:10] # Check 10 candidates
                    
                    for y in candidates:
                        new_S = S + [y]
                        _, new_conflicts = compute_sums_and_conflicts(new_S)
                        if new_conflicts == 0:
                            S = new_S
                            current_size += 1
                            conflicts = 0
                            added = True
                            break
                    
                    if not added:
                        # No easy addition found. 
                        # We could try harder, but let's break to restart with different seed
                        # Or try to remove and re-add? 
                        # For now, break to try another restart.
                        break
                        
                else:
                    # Try to reduce conflicts by removing an element
                    # Sample candidates for removal
                    candidates = rng.sample(S, min(10, len(S)))
                    best_candidate = None
                    best_conflicts = conflicts + 1 # Must be better than current?
                    # Actually, we just want the minimum conflict count among candidates
                    min_conf = float('inf')
                    
                    for x in candidates:
                        # Remove x
                        idx = S.index(x)
                        new_S = S[:idx] + S[idx+1:]
                        _, new_conflicts = compute_sums_and_conflicts(new_S)
                        
                        if new_conflicts < min_conf:
                            min_conf = new_conflicts
                            best_candidate = x
                            
                    if best_candidate is not None:
                        # Remove best_candidate
                        idx = S.index(best_candidate)
                        S = S[:idx] + S[idx+1:]
                        current_size -= 1
                        sums, conflicts = compute_sums_and_conflicts(S)
                    else:
                        # Should not happen
                        break
                        
            # End of local search for this trial
            
            if current_size > best_size:
                best_size = current_size
                best_set = S

    return list(best_set)
