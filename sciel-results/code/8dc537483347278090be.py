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
        # Returns numpy array of counts for sums N_LOW..N_HIGH
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = np.array(list(A))
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
        return int(np.max(reps))
    
    def get_sum_squares(reps):
        return int(np.sum(reps * reps))
    
    def get_soft_peak(reps, beta=2.0):
        # Smooth surrogate for peak using log-sum-exp
        # Higher beta = closer to true max
        max_val = np.max(reps)
        if max_val == 0:
            return 0.0
        # Use log-sum-exp for numerical stability
        shifted = reps - max_val
        log_sum = max_val + np.log(np.sum(np.exp(shifted * beta))) / beta
        return float(log_sum)

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
    cooling = 0.998  # Slower cooling for more exploration
    
    # Precompute the list of all values to pick from.
    all_vals = list(range(MAX_VAL + 1))
    
    # Use soft peak as the primary SA objective for smoother landscape
    current_soft_peak = get_soft_peak(current_reps)
    best_soft_peak = get_soft_peak(best_reps)
    
    while time.time() < deadline and T > T_min:
        # Pick a random element to swap out
        if not current_A:
            break
            
        a = random.choice(list(current_A))
        
        # Pick a random element to swap in
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
        new_soft_peak = get_soft_peak(new_reps)
        
        # Use soft peak for SA acceptance (smoother landscape)
        delta_soft = new_soft_peak - current_soft_peak
        
        accept = False
        if delta_soft < 0:
            accept = True
        else:
            # Accept with probability exp(-delta/T)
            if T > 0:
                accept = random.random() < math.exp(-delta_soft / T)
        
        if accept:
            current_A = new_A
            current_reps = new_reps
            current_peak = new_peak
            current_sq = new_sq
            current_soft_peak = new_soft_peak
            
            # Update best using true peak and sum of squares
            if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                best_peak = current_peak
                best_sq = current_sq
                best_A = set(current_A)
                best_reps = current_reps
                best_soft_peak = new_soft_peak
        
        T *= cooling
        
    # Phase 3: Local Search Polish
    # 1. Greedy Removals
    # 2. Greedy Swaps (deterministic improvement)
    # 3. Final high-precision polish with tighter tolerance
    
    improved = True
    polish_passes = 0
    max_polish_passes = 10
    
    while improved and time.time() < deadline and polish_passes < max_polish_passes:
        improved = False
        polish_passes += 1
        
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
            # If no improvement found, try a random perturbation to escape local minimum
            if polish_passes < max_polish_passes and time.time() < deadline:
                # Make 2-3 random swaps to perturb
                for _ in range(random.randint(2, 3)):
                    if not best_A:
                        break
                    a = random.choice(list(best_A))
                    while True:
                        b = random.choice(all_vals)
                        if b not in best_A:
                            break
                    new_A = best_A - {a}
                    new_A.add(b)
                    new_reps = calc_reps(new_A)
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        new_sq = get_sum_squares(new_reps)
                        if new_peak <= best_peak:  # Allow equal peak to escape
                            best_A = new_A
                            best_reps = new_reps
                            best_peak = new_peak
                            best_sq = new_sq
                            improved = True
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
