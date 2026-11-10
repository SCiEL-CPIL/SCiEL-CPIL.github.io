import numpy as np
import time
import random

def construct():
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    
    start_time = time.time()
    DEADLINE = 82.0
    
    def calc_mult(A_list):
        """A_list must be sorted. Returns multiplicity array."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    def is_valid(M):
        return bool(np.all(M >= 1))
    
    # Start with [64, 128]
    A = list(range(64, 129))
    M = calc_mult(A)
    best_max = int(np.max(M))
    best_A = list(A)
    
    # Phase 1: Greedy removal
    for _ in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        if not is_valid(M):
            break
        current_max = int(np.max(M))
        improved = False
        # Try removing each element, prefer those that reduce max
        for idx in range(len(A)):
            new_A = A[:idx] + A[idx+1:]
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_max = int(np.max(new_M))
                if new_max < current_max:
                    A = new_A
                    M = new_M
                    current_max = new_max
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_A = list(A)
                    break
        if not improved:
            # Try random removals
            for _ in range(10):
                if time.time() - start_time > DEADLINE:
                    break
                idx = random.randint(0, len(A)-1)
                new_A = A[:idx] + A[idx+1:]
                if not new_A:
                    continue
                new_M = calc_mult(new_A)
                if is_valid(new_M) and int(np.max(new_M)) <= current_max:
                    A = new_A
                    M = new_M
                    if int(np.max(M)) < best_max:
                        best_max = int(np.max(M))
                        best_A = list(A)
                    improved = True
                    break
    
    # Phase 2: Local search with swaps
    for iteration in range(20000):
        if time.time() - start_time > DEADLINE:
            break
        current_max = int(np.max(M))
        if len(A) == 0:
            break
        
        # Remove random element, add random available element
        idx_remove = random.randint(0, len(A) - 1)
        elem_remove = A[idx_remove]
        A_set = set(A)
        available = [x for x in range(129) if x not in A_set]
        if not available:
            break
        elem_add = random.choice(available)
        
        new_A = A[:idx_remove] + A[idx_remove+1:]
        # Insert elem_add in sorted position
        pos = 0
        while pos < len(new_A) and new_A[pos] < elem_add:
            pos += 1
        new_A = new_A[:pos] + [elem_add] + new_A[pos:]
        
        new_M = calc_mult(new_A)
        if is_valid(new_M):
            new_max = int(np.max(new_M))
            if new_max < current_max:
                A = new_A
                M = new_M
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = list(A)
            elif new_max == current_max and random.random() < 0.05:
                A = new_A
                M = new_M
    
    # Phase 3: Final greedy removal
    for _ in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        if not is_valid(M):
            break
        current_max = int(np.max(M))
        improved = False
        for idx in range(len(A)):
            new_A = A[:idx] + A[idx+1:]
            if not new_A:
                continue
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_max = int(np.max(new_M))
                if new_max <= current_max:
                    A = new_A
                    M = new_M
                    if new_max < best_max:
                        best_max = new_max
                        best_A = list(A)
                    improved = True
                    break
        if not improved:
            break
    
    if time.time() - start_time > DEADLINE:
        return best_A
    
    return best_A

construct()
