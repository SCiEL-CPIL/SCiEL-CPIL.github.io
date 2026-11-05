"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

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
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    
    random.seed(42)
    
    def validate_and_obj(A):
        if len(A) != 20:
            return None
        if len(set(A)) != 20:
            return None
        if any(x < 1 or x > 10000 for x in A):
            return None
        s, p = _set_sizes(A)
        return max(s, p)
    
    # Generate diverse structured initial candidates
    candidates = []
    
    # 1. Small consecutive
    candidates.append(list(range(1, 21)))
    
    # 2. Arithmetic progressions with different starts and steps
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20, 25, 30]:
            cand = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in cand):
                candidates.append(sorted(cand))
    
    # 3. Near-arithmetic: AP with small perturbations
    base_ap = list(range(1, 21))
    for _ in range(80):
        cand = base_ap[:]
        for _ in range(random.randint(1, 8)):
            idx = random.randint(0, 19)
            cand[idx] += random.randint(-5, 5)
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            candidates.append(sorted(cand))
    
    # 4. Powers of small bases
    for base in [2, 3, 4, 5]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(val)
            val *= base
        if len(cand) >= 20:
            candidates.append(sorted(cand[:20]))
    
    # 5. Mixed: small AP + geometric tail
    for small_n in [8, 10, 12, 15]:
        for gap in [1, 2]:
            ap_part = list(range(1, 1 + small_n * gap, gap))
            if len(ap_part) < small_n:
                continue
            geo_start = ap_part[-1] + 1
            geo_part = []
            val = geo_start
            while len(geo_part) < 20 - small_n and val <= 10000:
                geo_part.append(val)
                val *= 2
            if len(geo_part) == 20 - small_n:
                cand = sorted(ap_part[:small_n] + geo_part)
                if len(set(cand)) == 20:
                    candidates.append(cand)
    
    # 6. Random sets in various ranges
    for rng_max in [30, 50, 100, 200, 500, 1000, 5000]:
        for _ in range(30):
            cand = sorted(random.sample(range(1, rng_max + 1), 20))
            candidates.append(cand)
    
    # 7. Hybrid: some small, some large
    for _ in range(50):
        small = random.sample(range(1, 30), 10)
        large = random.sample(range(50, 2000), 10)
        cand = sorted(small + large)
        if len(set(cand)) == 20:
            candidates.append(cand)
    
    # 8. Logarithmically spaced
    for base_log in [1.1, 1.2, 1.3, 1.5, 2.0]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(int(val))
            val *= base_log
        cand = sorted(set(cand))
        if len(cand) >= 20:
            candidates.append(sorted(cand[:20]))
    
    # Evaluate all candidates
    best_set = None
    best_obj = float('inf')
    
    for cand in candidates:
        obj = validate_and_obj(cand)
        if obj is not None and obj < best_obj:
            best_obj = obj
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = _objective(best_set)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.5
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random element to perturb
        i = random.randint(0, 19)
        orig_val = best_set[i]
        
        # Generate a new value
        if random.random() < 0.7:
            # Local perturbation
            delta_range = max(10, int(0.15 * orig_val))
            new_val = orig_val + random.randint(-delta_range, delta_range)
        else:
            # Global perturbation
            new_val = random.randint(1, 10000)
        
        if new_val < 1 or new_val > 10000 or new_val == orig_val:
            continue
        if new_val in best_set:
            continue
        
        new_set = best_set[:]
        new_set[i] = new_val
        new_set.sort()
        
        new_obj = validate_and_obj(new_set)
        if new_obj is None:
            continue
        
        # Accept if better, or with probability exp(-(new_obj - best_obj)/T)
        if new_obj < best_obj:
            best_set = new_set
            best_obj = new_obj
        else:
            delta_obj = new_obj - best_obj
            if delta_obj > 0:
                prob = math.exp(-delta_obj / T)
                if random.random() < prob:
                    best_set = new_set
                    best_obj = new_obj
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally do multi-element perturbations
        if random.random() < 0.15 and time.time() < deadline:
            k = random.randint(2, 4)
            indices = random.sample(range(20), k)
            new_vals = random.sample(range(1, 10000), k)
            
            if len(set(new_vals)) != k:
                continue
            if any(v in best_set for v in new_vals):
                continue
            
            new_set = best_set[:]
            for idx, val in zip(indices, new_vals):
                new_set[idx] = val
            new_set.sort()
            
            new_obj = validate_and_obj(new_set)
            if new_obj is not None:
                if new_obj < best_obj:
                    best_set = new_set
                    best_obj = new_obj
                else:
                    delta_obj = new_obj - best_obj
                    if delta_obj > 0:
                        prob = math.exp(-delta_obj / T)
                        if random.random() < prob:
                            best_set = new_set
                            best_obj = new_obj
    
    return best_set
# EVOLVE-BLOCK-END
