import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        n = len(A_arr)
        if n == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Use triu indices for unordered pairs with i<=j
        idx = np.triu_indices(n)
        upper_sums = A_arr[idx[0]] + A_arr[idx[1]]
        
        mask = (upper_sums >= N_LOW) & (upper_sums <= N_HIGH)
        if not np.any(mask):
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        valid_sums = upper_sums[mask]
        indices = (valid_sums - N_LOW).astype(np.int32)
        counts = np.bincount(indices, minlength=RANGE_SIZE)
        return counts[:RANGE_SIZE].astype(np.int32)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def greedy_construction():
        """Build A element by element, always choosing the best next element."""
        A = set()
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Start with a small seed set to ensure basic coverage
        seed_elements = [0, 128, 64, 1, 127, 2, 126]
        for x in seed_elements:
            A.add(x)
        
        reps = calc_reps(A)
        
        # Now greedily add elements that minimize peak while improving coverage
        max_elements = 60
        for _ in range(max_elements):
            if time.time() > deadline:
                break
            
            current_peak = get_peak(reps)
            missing_count = int(np.sum(reps == 0))
            
            if missing_count == 0 and len(A) > 20:
                break
            
            best_elem = None
            best_score = None
            
            # Try all candidate elements not in A
            for x in range(0, MAX_VAL + 1):
                if x in A:
                    continue
                
                new_A = A | {x}
                new_reps = calc_reps(new_A)
                new_peak = get_peak(new_reps)
                new_missing = int(np.sum(new_reps == 0))
                
                # Score: minimize peak, then minimize missing
                score = (new_peak, new_missing)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_elem = x
            
            if best_elem is not None and best_score is not None:
                A.add(best_elem)
                reps = calc_reps(A)
            else:
                break
        
        return A, reps

    def local_search(A, reps):
        """Perform local search to improve peak multiplicity."""
        current_A = set(A)
        current_reps = reps
        current_peak = get_peak(current_reps)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing elements
            for a in list(current_A):
                if time.time() > deadline:
                    break
                new_A = current_A - {a}
                if not new_A:
                    continue
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
                        improved = True
                    elif new_peak == current_peak:
                        # Keep if same peak but fewer elements
                        if len(new_A) < len(current_A):
                            current_A = new_A
                            current_reps = new_reps
                            improved = True
            
            # Try swapping elements
            for a in list(current_A):
                if time.time() > deadline:
                    break
                for b in range(0, MAX_VAL + 1):
                    if b in current_A:
                        continue
                    new_A = current_A - {a}
                    new_A.add(b)
                    new_reps = calc_reps(new_A)
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        if new_peak < current_peak:
                            current_A = new_A
                            current_reps = new_reps
                            current_peak = new_peak
                            improved = True
                            break
        
        return current_A, current_reps, current_peak

    best_A = None
    best_peak = float('inf')
    
    # Multi-start with greedy construction
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary the seed elements slightly for diversity
        A, reps = greedy_construction()
        
        if not is_covered(reps):
            continue
        
        # Local search to improve
        A, reps, peak = local_search(A, reps)
        
        if is_covered(reps) and peak < best_peak:
            best_peak = peak
            best_A = list(A)
    
    if best_A is None:
        # Fallback: use greedy construction result
        A, reps = greedy_construction()
        if is_covered(reps):
            return list(A)
        return list(range(64, 129))
    
    return best_A

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
