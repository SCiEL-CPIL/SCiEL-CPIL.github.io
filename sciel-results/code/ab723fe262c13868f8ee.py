import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        """Fast calculation of representation counts for sums in [N_LOW, N_HIGH]"""
        A_list = list(A)
        n = len(A_list)
        if n == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Use numpy for speed
        i_idx, j_idx = np.triu_indices(n, k=0)
        sums = np.array(A_list)[i_idx] + np.array(A_list)[j_idx]
        
        # Filter and bincount
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if not np.any(mask):
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        filtered_sums = sums[mask] - N_LOW
        reps = np.bincount(filtered_sums, minlength=RANGE_SIZE)
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))

    # Phase 1: Greedy Initialization
    A = {0}
    reps = calc_reps(A)
    
    max_init_steps = 200
    step = 0
    while not is_covered(reps) and step < max_init_steps and time.time() < deadline - 5:
        zero_reps_indices = np.where(reps == 0)[0]
        if len(zero_reps_indices) == 0:
            break
        target_offset = random.choice(zero_reps_indices)
        target_sum = target_offset + N_LOW
        
        candidates = []
        for a in A:
            x = target_sum - a
            if 0 <= x <= MAX_VAL and x not in A:
                candidates.append(x)
        
        if not candidates:
            A.add(random.randint(0, MAX_VAL))
            reps = calc_reps(A)
        else:
            x = random.choice(candidates)
            A.add(x)
            reps = calc_reps(A)
        step += 1
        
    if not is_covered(reps):
        while not is_covered(reps) and time.time() < deadline - 5:
            A.add(random.randint(0, MAX_VAL))
            reps = calc_reps(A)
    
    best_A = set(A)
    best_reps = reps
    best_peak = get_peak(best_reps)
    
    # Phase 2: Local Search
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try removals
        for a in list(best_A):
            if time.time() > deadline - 2:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                if new_peak <= best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    improved = True
                    break
        
        if improved:
            no_improvement_count = 0
            continue
        
        # Try swaps
        if len(best_A) > 0:
            a_to_remove = random.choice(list(best_A))
            non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
            if non_A:
                b_to_add = random.choice(non_A)
                new_A = best_A - {a_to_remove}
                new_A.add(b_to_add)
                
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_peak:
                        best_A = new_A
                        best_reps = new_reps
                        best_peak = new_peak
                        improved = True
                        no_improvement_count = 0
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
                if non_A:
                    best_A.add(random.choice(non_A))
                    best_reps = calc_reps(best_A)
                    best_peak = get_peak(best_reps)
                no_improvement_count = 0

    # Final cleanup
    for a in list(best_A):
        if time.time() > deadline - 1:
            break
        new_A = best_A - {a}
        if not new_A:
            continue
        new_reps = calc_reps(new_A)
        if is_covered(new_reps):
            new_peak = get_peak(new_reps)
            if new_peak <= best_peak:
                best_A = new_A
                best_reps = new_reps
                best_peak = new_peak

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
