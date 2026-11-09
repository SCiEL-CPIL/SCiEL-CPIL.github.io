"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 128
    # A is a subset of [0, 128]
    # We need every integer in [128, 256] to be a sum of two elements of A (unordered, equal allowed)
    
    def compute_multiplicities(A_set):
        """Compute the number of representations for each sum in [128, 256]."""
        A_list = sorted(A_set)
        mult = {}
        for s in range(128, 257):
            count = 0
            for a in A_list:
                if a > s:
                    break
                b = s - a
                if b < a:
                    continue
                if b > n:
                    continue
                if b in A_set:
                    count += 1
            mult[s] = count
        return mult
    
    def peak_multiplicity(A_set):
        mult = compute_multiplicities(A_set)
        return max(mult.values())
    
    def is_valid(A_set):
        """Check that every integer in [128, 256] has at least one representation."""
        mult = compute_multiplicities(A_set)
        for s in range(128, 257):
            if mult[s] == 0:
                return False
        return True
    
    # Initial solution: two-cluster design
    # A = {0..k} ∪ {128-k..128}
    best_k = 43
    A_init = set(range(0, best_k + 1)) | set(range(128 - best_k, 129))
    
    # Verify validity
    if not is_valid(A_init):
        # Try different k values
        for k in range(40, 50):
            A_test = set(range(0, k + 1)) | set(range(128 - k, 129))
            if is_valid(A_test):
                A_init = A_test
                break
    
    current_A = set(A_init)
    current_peak = peak_multiplicity(current_A)
    best_A = set(current_A)
    best_peak = current_peak
    
    # Local search: try adding/removing elements
    all_elements = list(range(0, 129))
    random.shuffle(all_elements)
    
    def try_move(A_set, elem, add):
        """Try adding or removing an element, return new set if valid and improves."""
        new_A = set(A_set)
        if add:
            new_A.add(elem)
        else:
            if elem not in new_A:
                return None
            new_A.remove(elem)
        
        if not is_valid(new_A):
            return None
        new_peak = peak_multiplicity(new_A)
        return new_A, new_peak
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    step = 0
    while T > T_min and time.time() < deadline:
        # Choose a random move: add or remove a random element
        elem = random.choice(all_elements)
        if elem in current_A:
            add = False
        else:
            add = True
        
        new_A, new_peak = try_move(current_A, elem, add)
        
        if new_A is not None:
            delta = new_peak - current_peak
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(current_A)
        
        T *= cooling
        step += 1
        
        # Periodic local search polish
        if step % 100 == 0 and time.time() < deadline - 10:
            # Try all single-element moves
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for elem in all_elements:
                    if elem in current_A:
                        new_A, new_peak = try_move(current_A, elem, False)
                    else:
                        new_A, new_peak = try_move(current_A, elem, True)
                    if new_A is not None and new_peak < current_peak:
                        current_A = new_A
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(current_A)
                        improved = True
                        break
    
    return list(best_A)
# EVOLVE-BLOCK-END

