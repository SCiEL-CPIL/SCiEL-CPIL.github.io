"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

def _set_sizes(A):
    """Compute |A+A| and |A*A| for set A (with equal elements allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _set_sizes(A)
    return max(s, p)

# EVOLVE-BLOCK-START
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    
    random.seed(42)
    
    best_set = list(range(1, 21))
    best_obj = _objective(best_set)
    
    def try_candidate(A):
        """Validate and evaluate candidate, update best if improved."""
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)
    
    # --- Structured initial candidates ---
    # 1. Small consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Arithmetic progressions with various starts and steps
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20]:
            cand = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 20:
                try_candidate(sorted(cand))
    
    # 3. Geometric progressions
    for base in [2, 3, 4]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(val)
            val *= base
        if len(cand) >= 20:
            try_candidate(sorted(cand[:20]))
    
    # 4. Near-arithmetic perturbations of [1..20]
    for _ in range(50):
        cand = list(range(1, 21))
        for _ in range(random.randint(1, 5)):
            idx = random.randint(0, 19)
            cand[idx] += random.randint(-3, 3)
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            try_candidate(sorted(cand))
    
    # 5. Random sets with small values
    for _ in range(30):
        cand = sorted(random.sample(range(1, 100), 20))
        try_candidate(cand)
    
    # 6. Random sets with mixed range
    for _ in range(30):
        cand = sorted(random.sample(range(1, 500), 20))
        try_candidate(cand)
    
    # 7. Hybrid: some small, some large
    for _ in range(20):
        small = random.sample(range(1, 20), 10)
        large = random.sample(range(100, 1000), 10)
        cand = sorted(small + large)
        if len(set(cand)) == 20:
            try_candidate(cand)
    
    # --- Simulated Annealing refinement ---
    current_set = best_set[:]
    current_obj = best_obj
    T = 50.0  # initial temperature
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.1:
            T = 50.0  # restart with higher temperature
            current_set = best_set[:]
            current_obj = best_obj
        
        # Generate a neighbor
        move_type = random.random()
        new_set = current_set[:]
        
        if move_type < 0.5:
            # Perturb one element
            i = random.randint(0, 19)
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-max(5, int(0.1 * current_set[i])), 
                                       max(5, int(0.1 * current_set[i])))
                new_val = current_set[i] + delta
            else:
                # Random replacement
                new_val = random.randint(1, 10000)
            
            if 1 <= new_val <= 10000 and new_val not in current_set:
                new_set[i] = new_val
            else:
                continue
                
        elif move_type < 0.8:
            # Swap two elements with new random values
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, 10000)
            new_j = random.randint(1, 10000)
            if new_i != new_j and new_i not in current_set and new_j not in current_set:
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue
                
        else:
            # Replace 2-3 elements at once
            k = random.randint(2, 3)
            indices = random.sample(range(20), k)
            new_vals = random.sample(range(1, 10000), k)
            if len(set(new_vals)) == k and not any(v in current_set for v in new_vals):
                for idx, val in zip(indices, new_vals):
                    new_set[idx] = val
            else:
                continue
        
        new_set.sort()
        
        # Validate
        if len(new_set) != 20 or len(set(new_set)) != 20 or any(x < 1 or x > 10000 for x in new_set):
            continue
        
        new_obj = _objective(new_set)
        
        # Acceptance criterion
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / T):
            current_set = new_set
            current_obj = new_obj
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_set = new_set[:]
        
        T *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
