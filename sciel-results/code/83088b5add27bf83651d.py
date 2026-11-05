import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    targets = range(target_min, target_max + 1)
    
    # State:
    # A: set of elements
    # counts: array of length 257, counts[s] = number of pairs (a,b) in A with a<=b and a+b=s
    
    def evaluate(A):
        """
        A: set of ints
        Returns: (peak_multiplicity, is_valid)
        """
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A)
        for i, a in enumerate(A_list):
            for b in A_list[i:]: # b >= a
                s = a + b
                if s <= 256:
                    counts[s] += 1
        
        valid = True
        peak = 0
        for s in targets:
            if counts[s] == 0:
                valid = False
                break
            if counts[s] > peak:
                peak = counts[s]
        
        return peak, valid

    # Initial random construction
    # We need enough elements to cover the range.
    # Minimum size?
    # Let's start with ~40 elements.
    
    best_A = None
    best_peak = float('inf')
    
    # Try a few random inits
    for init_iter in range(5):
        # Random subset
        # To ensure coverage, maybe start with a structured set?
        # Or just random and repair?
        # Let's try random uniform.
        size = 40
        A = set(random.sample(range(n+1), size))
        
        # Repair phase: ensure all targets are covered
        # If a target sum is 0, add an element that helps?
        # This is complex. Instead, let's use a greedy init.
        
        # Greedy Init:
        A = set()
        # Start with some spread
        # Add elements that maximize coverage of currently 0-count sums?
        # Or just add random elements until valid?
        
        # Let's use a heuristic init:
        # Pick elements from [0, 128] randomly, but biased towards spreading sums.
        # Actually, a known good structure is roughly uniform.
        # Let's just do random and if invalid, do a "repair" by adding random elements.
        
        # But we can't add elements arbitrarily if we want to minimize peak.
        # We must swap.
        
        # Let's initialize with a "dense" random set.
        A = set(random.sample(range(n+1), 50))
        
        # Check validity
        peak, valid = evaluate(A)
        
        if not valid:
            # Try to fix by adding elements? No, we want fixed size or variable?
            # The problem says "subset A of [0,n]". It doesn't fix the size.
            # But adding elements increases peak multiplicity generally.
            # However, if we have holes, we MUST add elements.
            # So we allow size to grow if needed to become valid.
            
            # Simple repair: while not valid, add a random element that covers a 0-sum.
            while not valid and time.time() < deadline:
                counts = np.zeros(257, dtype=int)
                A_list = sorted(A)
                for i, a in enumerate(A_list):
                    for b in A_list[i:]:
                        s = a + b
                        if s <= 256:
                            counts[s] += 1
                
                zero_sums = [s for s in targets if counts[s] == 0]
                if not zero_sums:
                    break
                    
                # Pick a random zero sum
                s_target = random.choice(zero_sums)
                # Find a pair (x, y) such that x+y=s_target, x<=y
                # We want to add x or y to A.
                # Let's pick a random x in [0, 128] such that y = s_target - x is in [0, 128] and x<=y.
                # x in [max(0, s_target-128), min(128, s_target//2)]
                lo = max(0, s_target - 128)
                hi = min(128, s_target // 2)
                if lo <= hi:
                    x = random.randint(lo, hi)
                    A.add(x)
                    peak, valid = evaluate(A)
                else:
                    # Should not happen for valid targets
                    break
        
        if not valid:
            continue
            
        if peak < best_peak:
            best_peak = peak
            best_A = set(A)
            
        # Local Search
        current_A = set(A)
        current_peak, _ = evaluate(current_A)
        current_counts = np.zeros(257, dtype=int)
        A_list = sorted(current_A)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                s = a + b
                if s <= 256:
                    current_counts[s] += 1
                    
        no_improve_count = 0
        
        while time.time() < deadline:
            if no_improve_count > 500:
                break
                
            # Find peak sum
            # We only care about [128, 256]
            target_counts = current_counts[128:257]
            max_val = np.max(target_counts)
            
            if max_val < current_peak:
                current_peak = max_val
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(current_A)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Pick a random sum with max multiplicity
            # Or pick the global max?
            # Let's pick the global max to reduce the peak.
            peak_sum_idx = 128 + np.argmax(target_counts)
            
            # Try swaps
            # Pick a random element a in A to remove
            a_to_remove = random.choice(list(current_A))
            
            # Pick a random element z not in A to add
            # We want z to be "different" from a to change the sum distribution.
            z_to_add = random.randint(0, n)
            if z_to_add in current_A:
                continue
                
            # Simulate swap: A' = A - {a} + {z}
            # 1. Remove a from counts
            # Pairs involving a: (a, b) with b in A, b >= a.
            # Note: if a==b, it's (a,a).
            
            # We need to be careful with the counts update.
            # Let's just recalculate counts for the new set if the set is small?
            # |A| ~ 50. Recalc is 50*50/2 = 1250 ops.
            # This is very fast.
            # So we can just propose a swap, recalc, and check.
            
            new_A = current_A.copy()
            new_A.remove(a_to_remove)
            new_A.add(z_to_add)
            
            new_peak, new_valid = evaluate(new_A)
            
            if new_valid:
                if new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    no_improve_count = 0
                    # Update best
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(current_A)
                elif new_peak == current_peak and random.random() < 0.1:
                    # Accept sideways moves to escape local minima
                    current_A = new_A
                    # current_peak stays same
                else:
                    pass
            else:
                pass

    # Return the best A found
    # Ensure it's a list
    return list(best_A) if best_A else list(range(64, 129))
