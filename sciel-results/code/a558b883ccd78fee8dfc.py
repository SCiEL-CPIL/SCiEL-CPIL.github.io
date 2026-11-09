import time
import random
import math
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

    def get_sum_squares(reps):
        return sum(r * r for r in reps)

    # Phase 1: Initialize a covering set
    # Start with 0 and 128 to anchor the range
    A = set([0, 128])
    
    # Greedy add to ensure coverage
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
                candidates.append(y)
        
        if candidates:
            y = random.choice(candidates)
            A.add(y)
        else:
            non_A = [v for v in range(MAX_VAL + 1) if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break # A is full
        
    if not is_covered(calc_reps(A)):
        # Fallback
        return list(range(64, 129))

    # Phase 2: Optimization
    # We use a hybrid approach:
    # 1. Simulated Annealing to escape local minima, accepting worse peaks occasionally.
    # 2. Greedy Removal and Swap to polish the result.
    # Objective: Minimize peak. Tie-breaker: Minimize sum of squares (flatter distribution).
    
    best_A = set(A)
    best_reps = calc_reps(best_A)
    best_peak = get_peak(best_reps)
    best_sq = get_sum_squares(best_reps)
    
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    current_sq = get_sum_squares(current_reps)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    # Precompute potential swaps? No, dynamic is fine for small A.
    # But we can precompute the list of all values to pick from.
    all_vals = list(range(MAX_VAL + 1))
    
    while time.time() < deadline and T > T_min:
        # Pick a random element to swap out
        if not current_A:
            break
            
        a = random.choice(list(current_A))
        
        # Pick a random element to swap in
        # To avoid picking an element already in A
        # We can pick from all_vals and check, or maintain a list of non-A
        # Since A is small, checking is fast.
        while True:
            b = random.choice(all_vals)
            if b not in current_A:
                break
        
        # Perform swap
        new_A = set(current_A)
        new_A.remove(a)
        new_A.add(b)
        
        new_reps = calc_reps(new_A)
        
        # Check validity
        if not is_covered(new_reps):
            T *= cooling
            continue
            
        new_peak = get_peak(new_reps)
        new_sq = get_sum_squares(new_reps)
        
        # Calculate energy difference
        # Primary: Peak. Secondary: Sum of Squares.
        # We define a composite score for SA acceptance probability?
        # Or just use Peak for acceptance, and update best if (Peak, Sq) is lexicographically smaller.
        
        delta_peak = new_peak - current_peak
        
        accept = False
        if delta_peak < 0:
            accept = True
        elif delta_peak == 0:
            # Accept if sum of squares is lower, or with some probability to explore
            if new_sq < current_sq:
                accept = True
            else:
                # Accept with probability 0.5 to allow random walk in flat regions
                accept = random.random() < 0.5
        else:
            # delta_peak > 0
            # Accept with probability exp(-delta/T)
            if T > 0:
                accept = random.random() < math.exp(-delta_peak / T)
        
        if accept:
            current_A = new_A
            current_reps = new_reps
            current_peak = new_peak
            current_sq = new_sq
            
            # Update best
            if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                best_peak = current_peak
                best_sq = current_sq
                best_A = set(current_A)
                best_reps = current_reps
        
        T *= cooling
        
    # Phase 3: Local Search Polish
    # 1. Greedy Removals
    # 2. Greedy Swaps (deterministic improvement)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try Removals
        for a in list(best_A):
            if time.time() > deadline:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                new_sq = get_sum_squares(new_reps)
                # Accept if peak is lower, or (peak same and sq lower)
                if new_peak < best_peak or (new_peak == best_peak and new_sq < best_sq):
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    best_sq = new_sq
                    improved = True
        
        if time.time() > deadline:
            break
            
        # Try Swaps
        A_list = list(best_A)
        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
        
        # Shuffle to avoid bias? Deterministic order might be okay, but random helps break symmetry.
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
                
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    new_sq = get_sum_squares(new_reps)
                    
                    if new_peak < best_peak or (new_peak == best_peak and new_sq < best_sq):
                        best_A = new_A
                        best_reps = new_reps
                        best_peak = new_peak
                        best_sq = new_sq
                        improved = True
                        # Restart loops with new A
                        A_list = list(best_A)
                        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
                        break # Restart inner loop
            if improved:
                break
        
        if not improved:
            # If no improvement found in this pass, we might be stuck.
            # We could try a few random perturbations, but we already did SA.
            # Let's break to save time, or do one more SA cycle if time permits?
            # Given the 60s limit, and that SA usually finds the global or near-global,
            # and local search polishes it, breaking is safe.
            break

    return list(best_A)

if __name__ == "__main__":
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
