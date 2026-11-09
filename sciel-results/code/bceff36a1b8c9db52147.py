import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s margin for safety
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # Returns list of counts for sums N_LOW..N_HIGH
        reps = [0] * RANGE_SIZE
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                aj = A_list[j]
                s = ai + aj
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def is_covered(reps):
        return all(r > 0 for r in reps)

    def get_peak(reps):
        return max(reps)

    # Phase 1: Initialize a covering set
    # Start with 0 and 128 to anchor the range
    A = set([0, 128])
    
    # Greedy add to ensure coverage
    # We want to keep A small initially to have room to optimize, but must cover.
    # Adding elements increases reps. 
    # Strategy: Add elements that cover the most missing sums.
    
    # To speed up init, let's just add random elements until covered.
    # Or better: 
    # Check coverage. If missing, pick a missing sum S.
    # Find y such that y + z = S for z in A.
    # Add y.
    
    while True:
        if time.time() > deadline:
            break
        reps = calc_reps(A)
        if is_covered(reps):
            break
        
        # Find a missing sum
        missing_indices = [i for i, r in enumerate(reps) if r == 0]
        if not missing_indices:
            break
            
        # Pick a random missing sum
        idx = random.choice(missing_indices)
        S = N_LOW + idx
        
        # Find candidates y
        candidates = []
        for z in A:
            y = S - z
            if 0 <= y <= MAX_VAL and y not in A:
                # Check if adding y helps cover this S (it does by definition)
                candidates.append(y)
        
        if candidates:
            # Pick candidate that minimizes increase in peak?
            # Just pick one randomly or first one.
            y = random.choice(candidates)
            A.add(y)
        else:
            # If no candidate found (shouldn't happen if A is non-empty and S in range 
            # and we can extend A? 
            # If A={0, 128}, S=128 is covered by 0+128.
            # If S=129, 129-0=129 (out of range? 129>128). 129-128=1. 
            # So y=1 is valid.
            # If we are stuck, it means no y in [0, 128]\A works with existing A.
            # This implies A is "isolated" for this S.
            # We should just pick a random y in [0, 128] not in A and add it, hoping it connects.
            # But better: add a random y that might bridge gaps.
            all_vals = list(range(MAX_VAL + 1))
            non_A = [v for v in all_vals if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break # A is full
    
    # Phase 2: Optimization
    best_A = set(A)
    best_reps = calc_reps(best_A)
    best_peak = get_peak(best_reps)
    
    # Local Search Loop
    # 1. Try to remove elements (if coverage maintained)
    # 2. Try to swap elements (if coverage maintained and peak reduced)
    
    while time.time() < deadline:
        improved = False
        
        # Try Removals
        # Removing an element can only decrease or maintain peak, 
        # but might break coverage.
        for a in list(best_A):
            if time.time() > deadline:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                # Accept if peak is same or lower (it will be)
                # Actually, removal always lowers avg multiplicity, so peak likely lower or same.
                # If peak is same, we still accept to reduce size? 
                # The metric is "largest representation multiplicity". 
                # If peak doesn't change, size reduction doesn't help the score directly, 
                # but might allow future swaps to be more effective? 
                # Let's accept if peak < best_peak OR (peak == best_peak and len(new_A) < len(best_A)).
                if new_peak < best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    improved = True
                elif new_peak == best_peak and len(new_A) < len(best_A):
                    best_A = new_A
                    best_reps = new_reps
                    improved = True
        
        if time.time() > deadline:
            break
            
        # Try Swaps
        # For each a in A, for each b not in A:
        # Check if A' = A - {a} + {b} is covered.
        # Check if peak(A') < best_peak.
        
        # To optimize, we can pre-filter b.
        # But simple loop is okay for small A.
        
        A_list = list(best_A)
        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
        
        # Shuffle to avoid bias
        random.shuffle(A_list)
        random.shuffle(non_A)
        
        for a in A_list:
            if time.time() > deadline:
                break
            for b in non_A:
                if time.time() > deadline:
                    break
                new_A = best_A - {a}
                new_A.add(b)
                
                # Quick check: if b is very far from a, might not help much?
                # Just calc.
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_peak:
                        best_A = new_A
                        best_reps = new_reps
                        best_peak = new_peak
                        improved = True
                        # Restart swap loops with new A
                        A_list = list(best_A)
                        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
                        break # Restart inner loop
            if improved:
                break
        
        if not improved:
            # Try random perturbations if stuck?
            # Or just break if we've been stuck for a while?
            # Let's try a few random swaps even if they don't improve peak, 
            # to escape local minima?
            # No, we only care about minimizing peak. 
            # If we can't improve peak, we are likely at a local minimum for peak.
            # We can try to minimize "sum of squares" or similar to flatten?
            # But the objective is strictly max multiplicity.
            # If we can't reduce max, we are done.
            break
            
    return list(best_A)

# Run once to verify? No, just define.
# The evaluator calls construct().

if __name__ == "__main__":
    # Simple test
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
