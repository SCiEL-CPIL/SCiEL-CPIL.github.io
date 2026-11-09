import random
import numpy as np
import time

def construct():
    best_count = 23
    best_coeffs = [1]*12
    
    def count_nonzero_fast(coeffs):
        """Use numpy for fast convolution and counting."""
        arr = np.array(coeffs, dtype=np.int64)
        square = np.convolve(arr, arr)
        return int(np.count_nonzero(square))
    
    def simulated_annealing(initial_coeffs, steps=150000, start_temp=8.0, end_temp=0.005):
        current_coeffs = initial_coeffs[:]
        current_score = count_nonzero_fast(current_coeffs)
        best_local_coeffs = current_coeffs[:]
        best_local_score = current_score
        
        for i in range(steps):
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            idx = random.randint(0, 11)
            old_val = current_coeffs[idx]
            # Propose new value in wider range but weighted toward small values
            r = random.random()
            if r < 0.4:
                new_val = random.choice([-2, -1, 1, 2])
            elif r < 0.7:
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
            elif r < 0.9:
                new_val = random.randint(-5, 5)
            else:
                new_val = random.randint(-15, 15)
            if new_val == old_val or new_val == 0:
                continue
            
            current_coeffs[idx] = new_val
            new_score = count_nonzero_fast(current_coeffs)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_coeffs = current_coeffs[:]
            else:
                if delta < 5 and random.random() < np.exp(-delta / max(temp, 0.01)):
                    current_score = new_score
                else:
                    current_coeffs[idx] = old_val
        
        return best_local_coeffs, best_local_score

    def local_search(coeffs, max_iter=500):
        """Intensive local search: try all single-coefficient changes."""
        current = coeffs[:]
        current_score = count_nonzero_fast(current)
        for _ in range(max_iter):
            improved = False
            for i in range(12):
                old = current[i]
                for new_c in range(-20, 21):
                    if new_c == 0 or new_c == old:
                        continue
                    current[i] = new_c
                    new_score = count_nonzero_fast(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current[i] = old
            if not improved:
                break
        return current, current_score

    def get_anti_palindromic(vars6):
        a = vars6[:]
        return a + [-a[5], -a[4], -a[3], -a[2], -a[1], -a[0]]

    def get_palindromic(vars6):
        a = vars6[:]
        return a + a[5::-1]

    def get_even_palindromic(vars6):
        """a_i = a_{11-i} with even symmetry"""
        a = vars6[:]
        return a + a[5::-1]

    def sa_structured(get_func, nvars, seed, steps=80000):
        random.seed(seed)
        best_local_score = 999
        best_local_vars = None
        current_vars = [random.randint(-5, 5) for _ in range(nvars)]
        for i in range(nvars):
            if current_vars[i] == 0: current_vars[i] = 1
        current_coeffs = get_func(current_vars)
        current_score = count_nonzero_fast(current_coeffs)
        best_local_score = current_score
        best_local_vars = current_vars[:]
        
        for i in range(steps):
            temp = 5.0 * (0.01 / 5.0) ** (i / steps)
            idx = random.randint(0, nvars - 1)
            old_val = current_vars[idx]
            r = random.random()
            if r < 0.5:
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
            else:
                new_val = random.randint(-8, 8)
            if new_val == 0 or new_val == old_val: continue
            
            current_vars[idx] = new_val
            new_coeffs = get_func(current_vars)
            new_score = count_nonzero_fast(new_coeffs)
            
            delta = new_score - current_score
            if delta < 0 or (delta < 4 and random.random() < np.exp(-delta / max(temp, 0.01))):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_vars = current_vars[:]
            else:
                current_vars[idx] = old_val
        
        return get_func(best_local_vars), best_local_score

    # Phase 1: General SA with multiple random starts
    for run in range(8):
        random.seed(42 + run)
        init = []
        for _ in range(12):
            r = random.random()
            if r < 0.5:
                init.append(random.choice([-2, -1, 1, 2]))
            elif r < 0.8:
                init.append(random.choice([-3, -2, -1, 1, 2, 3]))
            else:
                init.append(random.randint(-5, 5))
        if 0 in init:
            init = [x if x != 0 else 1 for x in init]
        b_c, b_s = simulated_annealing(init, steps=100000, start_temp=6.0, end_temp=0.005)
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # Phase 2: Anti-palindromic search
    for run in range(6):
        b_c, b_s = sa_structured(get_anti_palindromic, 6, seed=100 + run, steps=60000)
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # Phase 3: Palindromic search
    for run in range(6):
        b_c, b_s = sa_structured(get_palindromic, 6, seed=200 + run, steps=60000)
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # Phase 4: Local search refinement on best found
    best_coeffs, best_count = local_search(best_coeffs, max_iter=100)

    # Phase 5: More SA runs seeded from the best found
    for run in range(4):
        random.seed(300 + run)
        init = best_coeffs[:]
        # Perturb
        for i in range(3):
            idx = random.randint(0, 11)
            init[idx] = random.randint(-8, 8)
            if init[idx] == 0:
                init[idx] = 1
        b_c, b_s = simulated_annealing(init, steps=80000, start_temp=4.0, end_temp=0.005)
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # Phase 6: Final local search
    best_coeffs, best_count = local_search(best_coeffs, max_iter=200)

    # Verify constraints
    assert len(best_coeffs) == 12
    assert all(c != 0 for c in best_coeffs)
    assert all(-64 <= c <= 64 for c in best_coeffs)
    assert best_coeffs[-1] != 0

    return best_coeffs
