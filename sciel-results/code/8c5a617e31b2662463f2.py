import numpy as np
import time
import random

def construct():
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N = 128
    
    start_time = time.time()
    DEADLINE = 85.0
    
    def calc_mult(A_set):
        """A_set is a set of integers. Returns multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        # Create indicator vector of length N+1 (indices 0..N)
        x = np.zeros(N + 1, dtype=np.int32)
        for a in A_set:
            x[a] = 1
            
        # Convolution x * x gives counts of ordered pairs
        # Length of x is N+1=129. Convolution length is 2*N+1 = 257.
        # Indices 0..256.
        conv = np.convolve(x, x, mode='full')
        
        # We care about sums in [MIN_SUM, MAX_SUM] which are indices 128..256 in conv.
        # conv[k] is count of ordered pairs (a,b) in A x A such that a+b=k.
        
        # Calculate unordered multiplicity M[k]
        # M[k] = (conv[k] + eq_count) / 2
        # eq_count is 1 if k is even and k/2 is in A, else 0.
        
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        for i in range(NUM_SUMS):
            k = MIN_SUM + i
            eq = 1 if (k % 2 == 0 and (k // 2) in A_set) else 0
            M[i] = (conv[k] + eq) // 2
            
        return M

    def is_valid(M):
        return bool(np.all(M >= 1))

    # Initialize A
    # Start with full set [0, 128] to ensure validity, then prune.
    A = set(range(N + 1))
    M = calc_mult(A)
    best_max = int(np.max(M))
    best_A = set(A)
    
    # Phase 1: Greedy removal to reduce peak
    # Since calc_mult is fast, we can do more iterations.
    while time.time() - start_time < DEADLINE - 10:
        if not is_valid(M):
            break
        current_max = int(np.max(M))
        if current_max < best_max:
            best_max = current_max
            best_A = set(A)
            
        improved = False
        # Try removing each element
        for a in list(A):
            new_A = A - {a}
            if not new_A: continue
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_max = int(np.max(new_M))
                if new_max < current_max:
                    A = new_A
                    M = new_M
                    current_max = new_max
                    improved = True
                    break
        
        if not improved:
            # Try random swaps to escape local minima
            if len(A) < N + 1:
                A_list = list(A)
                available = [x for x in range(N + 1) if x not in A]
                if available:
                    for _ in range(50):
                        if time.time() - start_time > DEADLINE - 10:
                            break
                        a_remove = random.choice(A_list)
                        a_add = random.choice(available)
                        if a_remove == a_add: continue
                        new_A = A - {a_remove} | {a_add}
                        new_M = calc_mult(new_A)
                        if is_valid(new_M):
                            new_max = int(np.max(new_M))
                            if new_max < current_max:
                                A = new_A
                                M = new_M
                                current_max = new_max
                                improved = True
                                break
                            elif new_max == current_max and random.random() < 0.1:
                                A = new_A
                                M = new_M
                                improved = True
                                break
    
    # Phase 2: Final polish with simulated annealing steps
    if time.time() - start_time < DEADLINE - 5:
        A = set(best_A)
        M = calc_mult(A)
        current_max = int(np.max(M))
        
        temp = 1.0
        for i in range(1000):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not A: break
            
            move_type = random.choice(['remove', 'add', 'swap'])
            
            if move_type == 'remove' and len(A) > 1:
                a = random.choice(list(A))
                new_A = A - {a}
            elif move_type == 'add':
                available = [x for x in range(N + 1) if x not in A]
                if not available: continue
                a = random.choice(available)
                new_A = A | {a}
            else: # swap
                if len(A) == 0: continue
                available = [x for x in range(N + 1) if x not in A]
                if not available: continue
                a_remove = random.choice(list(A))
                a_add = random.choice(available)
                new_A = A - {a_remove} | {a_add}
            
            if not new_A: continue
            
            new_M = calc_mult(new_A)
            if not is_valid(new_M):
                continue
                
            new_max = int(np.max(new_M))
            delta = new_max - current_max
            
            if delta < 0:
                A = new_A
                M = new_M
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(A)
            else:
                if random.random() < np.exp(-delta / temp):
                    A = new_A
                    M = new_M
                    current_max = new_max
                    
            temp *= 0.99 

    return list(best_A)

construct()
