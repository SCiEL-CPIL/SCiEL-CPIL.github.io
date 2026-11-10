"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    def max_sidon_subset_size(arr):
        """Compute the largest Sidon subset size of arr (12 elements)."""
        n = len(arr)
        best = 0
        for size in range(n, 0, -1):
            for subset in combinations(range(n), size):
                vals = [arr[i] for i in subset]
                sums = set()
                is_sidon = True
                for i in range(len(vals)):
                    vi = vals[i]
                    for j in range(i, len(vals)):
                        s = vi + vals[j]
                        if s in sums:
                            is_sidon = False
                            break
                        sums.add(s)
                    if not is_sidon:
                        break
                if is_sidon:
                    best = size
                    break
            if best > 0:
                break
        return best
    
    # Initial candidates
    candidates = [list(range(12)), list(range(0, 24, 2)), list(range(1, 23, 2))]
    random.seed(42)
    for _ in range(30):
        candidates.append(random.sample(range(129), 12))
    
    best_set = None
    best_score = 13
    for c in candidates:
        score = max_sidon_subset_size(c)
        if score < best_score:
            best_score = score
            best_set = c[:]
    
    # Simulated annealing
    current_set = best_set[:]
    current_score = best_score
    T = 5.0
    T_min = 0.5
    alpha = 0.9995
    
    max_iter = 8000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        new_set = current_set[:]
        i = random.randint(0, 11)
        new_val = random.randint(0, 128)
        if new_val in new_set:
            available = [v for v in range(129) if v not in new_set]
            new_val = random.choice(available)
        new_set[i] = new_val
        
        new_score = max_sidon_subset_size(new_set)
        
        if new_score < current_score:
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta / T):
                current_set = new_set
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_set
# EVOLVE-BLOCK-END

