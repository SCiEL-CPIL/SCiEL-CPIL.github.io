"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin from 60s limit
    
    # Precompute all arithmetic progressions (start, step, length)
    # We only care about progressions that could potentially have large sums
    # For efficiency, we'll generate them on the fly or precompute key ones
    
    # Start with a reasonable initial bound
    # Thue-Morse has low discrepancy, so let's start there and find its max
    # Actually, let's just start with a bound and try to satisfy it
    
    # Initial sequence: Thue-Morse
    def _thue_morse(n):
        seq = [1] * n
        for i in range(n):
            c = 0
            x = i
            while x:
                c += x & 1
                x >>= 1
            seq[i] = 1 if c % 2 == 0 else -1
        return seq
    
    seq = _thue_morse(n)
    
    # Compute current max discrepancy
    def _compute_max_discrepancy(seq):
        n = len(seq)
        max_abs = 0
        arr = np.array(seq, dtype=np.int32)
        for d in range(1, n):
            for r in range(d):
                idxs = np.arange(r, n, d)
                if len(idxs) == 0:
                    continue
                vals = arr[idxs]
                cumsum = np.cumsum(vals)
                local_max = max(int(np.max(cumsum)), -int(np.min(cumsum)))
                if local_max > max_abs:
                    max_abs = local_max
        return max_abs
    
    current_max = _compute_max_discrepancy(seq)
    best_seq = list(seq)
    best_cost = current_max
    
    # Now try to improve using CSP-style backtracking with forward checking
    # We'll try to find a sequence with max discrepancy < current_max
    
    # Strategy: Try to reduce the bound by 1, then 2, etc.
    # For each bound B, do a backtracking search with forward checking
    
    def _check_partial(seq, B):
        """
        Check if partial assignment violates bound B.
        Returns True if feasible so far, False if violated.
        For efficiency, we only check progressions that are "active" (have unassigned elements)
        or are fully assigned and exceed B.
        """
        n = len(seq)
        # For each step d, check each residue class
        # This is still expensive, so we'll use a simplified check
        # Only check fully assigned progressions and those with few unassigned elements
        
        # Actually, for CSP, we need to check if any progression CANNOT be kept within B
        # given the current partial assignment.
        
        # Simplified: Check all fully assigned progressions
        # And for partially assigned, check if the current sum + max possible remaining > B
        # or current sum - max possible remaining < -B
        
        for d in range(1, n):
            for r in range(d):
                idxs = np.arange(r, n, d)
                if len(idxs) == 0:
                    continue
                
                # Check if this progression is fully assigned
                # If not, we can't fully verify, but we can bound
                assigned_count = 0
                current_sum = 0
                unassigned_count = 0
                for idx in idxs:
                    if seq[idx] != 0:
                        assigned_count += 1
                        current_sum += seq[idx]
                    else:
                        unassigned_count += 1
                
                if assigned_count == 0:
                    continue
                
                # If fully assigned, check if sum exceeds B
                if unassigned_count == 0:
                    if abs(current_sum) > B:
                        return False
                else:
                    # Partially assigned: check if it's impossible to keep within B
                    # The remaining elements can contribute at most unassigned_count in absolute value
                    # So the final sum is in [current_sum - unassigned_count, current_sum + unassigned_count]
                    # We need this interval to intersect [-B, B]
                    min_possible = current_sum - unassigned_count
                    max_possible = current_sum + unassigned_count
                    if min_possible > B or max_possible < -B:
                        return False
        
        return True
    
    # Backtracking search with forward checking
    def _backtracking_search(seq, B, pos, deadline):
        """
        Try to assign positions 0..n-1 such that max discrepancy <= B.
        Returns True if found, and modifies seq in place.
        """
        if pos == n:
            # Fully assigned, verify
            return _check_partial(seq, B)
        
        if time.time() > deadline:
            return False
        
        # Try both signs
        for sign in [1, -1]:
            seq[pos] = sign
            
            # Forward checking: verify feasibility
            if _check_partial(seq, B):
                if _backtracking_search(seq, B, pos + 1, deadline):
                    return True
            
            seq[pos] = 0
        
        return False
    
    # Try to improve the bound
    # Start from current_max - 1 and go down
    # But backtracking might be too slow for n=256
    
    # Alternative: Use local search guided by the CSP constraints
    # Flip positions that are in progressions currently exceeding the target bound
    
    target_bound = current_max - 1
    if target_bound < 1:
        target_bound = 1
    
    # Use a greedy/local search approach that respects the bound
    # Flip positions to reduce the maximum discrepancy
    
    rng = random.Random(42)
    T = 5.0
    decay = 0.999
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Find progressions that are currently at the maximum
        # Flip a position that is part of such a progression
        # This is heuristic
        
        # Pick a random position
        pos = rng.randint(0, n - 1)
        
        # Flip
        seq[pos] = -seq[pos]
        
        # Compute new cost
        new_cost = _compute_max_discrepancy(seq)
        
        if new_cost <= best_cost:
            best_cost = new_cost
            best_seq = list(seq)
        elif new_cost <= current_max:
            # Accept if within current bound
            current_max = new_cost
        else:
            # Accept with probability
            delta = new_cost - current_max
            prob = np.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < prob:
                current_max = new_cost
            else:
                seq[pos] = -seq[pos]
        
        T *= decay
        if T < 0.1:
            T = 0.1
    
    return best_seq
# EVOLVE-BLOCK-END
