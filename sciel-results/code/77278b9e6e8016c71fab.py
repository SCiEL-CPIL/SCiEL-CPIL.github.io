import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for safety
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    
    # Precompute sum indices for fast lookup
    # sums[i][j] = index into reps array for A[i]+A[j], or -1 if out of range
    # But A changes, so we compute on the fly efficiently.
    
    def calc_reps(A):
        """Returns numpy array of representation counts for sums N_LOW..N_HIGH"""
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    def get_sum_squares(reps):
        return int(np.sum(reps * reps))
    
    def get_score(reps):
        """Primary: peak. Secondary: sum of squares (lower is better)."""
        peak = int(np.max(reps))
        ss = int(np.sum(reps * reps))
        return (peak, ss)
    
    # Phase 1: Construct a valid covering set
    # Strategy: Start with a structured set that's likely to cover, then optimize.
    # Use a "balanced" initial set: include 0, 128, and a spread of middle values.
    
    A = set()
    A.add(0)
    A.add(128)
    
    # Add elements to ensure coverage. Use a greedy approach.
    # For each missing sum, add an element that helps.
    
    for _ in range(100):  # Max iterations
        if time.time() > deadline - 40:
            break
        reps = calc_reps(A)
        if is_covered(reps):
            break
        
        missing = np.where(reps == 0)[0]
        if len(missing) == 0:
            break
        
        # Pick a random missing sum
        idx = random.choice(missing)
        S = N_LOW + idx
        
        # Find a y such that there exists z in A with y+z = S and 0<=y<=128
        candidates = []
        for z in A:
            y = S - z
            if 0 <= y <= MAX_VAL and y not in A:
                candidates.append(y)
        
        if candidates:
            # Choose the candidate that minimizes the increase in peak
            best_y = candidates[0]
            best_peak_increase = float('inf')
            for y in candidates:
                test_A = A | {y}
                test_reps = calc_reps(test_A)
                peak = get_peak(test_reps)
                if peak < best_peak_increase:
                    best_peak_increase = peak
                    best_y = y
            A.add(best_y)
        else:
            # No valid y from existing A. Add a random element.
            non_A = [v for v in range(MAX_VAL + 1) if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break
    
    # Ensure we have a covering set. If not, force-add more elements.
    reps = calc_reps(A)
    while not is_covered(reps) and time.time() < deadline - 35:
        missing = np.where(reps == 0)[0]
        idx = random.choice(missing)
        S = N_LOW + idx
        candidates = []
        for z in A:
            y = S - z
            if 0 <= y <= MAX_VAL and y not in A:
                candidates.append(y)
        if candidates:
            A.add(random.choice(candidates))
        else:
            non_A = [v for v in range(MAX_VAL + 1) if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break
        reps = calc_reps(A)
    
    # Phase 2: Local search optimization
    best_A = set(A)
    best_reps = calc_reps(best_A)
    best_score = get_score(best_reps)
    best_peak = best_score[0]
    
    # Local search with simulated annealing
    T = 2.0  # Temperature
    T_min = 0.1
    cooling = 0.995
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try removals
        for a in list(best_A):
            if time.time() > deadline - 5:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if not is_covered(new_reps):
                continue
            new_score = get_score(new_reps)
            if new_score < best_score:
                best_A = new_A
                best_reps = new_reps
                best_score = new_score
                best_peak = new_score[0]
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try swaps
        A_list = list(best_A)
        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
        random.shuffle(A_list)
        random.shuffle(non_A)
        
        for a in A_list:
            if time.time() > deadline - 5:
                break
            for b in non_A:
                if time.time() > deadline - 5:
                    break
                new_A = best_A - {a}
                new_A.add(b)
                new_reps = calc_reps(new_A)
                if not is_covered(new_reps):
                    continue
                new_score = get_score(new_reps)
                delta_score = new_score[0] - best_score[0]
                if delta_score == 0:
                    delta_score = (new_score[1] - best_score[1]) / 10000.0
                
                if new_score < best_score or random.random() < np.exp(-delta_score / max(T, 0.01)):
                    best_A = new_A
                    best_reps = new_reps
                    best_score = new_score
                    best_peak = new_score[0]
                    improved = True
                    no_improve_count = 0
                    break
        
        if improved:
            continue
        
        # No improvement this iteration
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturb: add a random element
            non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
            if non_A:
                best_A.add(random.choice(non_A))
                best_reps = calc_reps(best_A)
                if not is_covered(best_reps):
                    # Revert
                    best_A.discard(random.choice(list(best_A)))
                    best_reps = calc_reps(best_A)
            no_improve_count = 0
        
        T = max(T * cooling, T_min)
    
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
