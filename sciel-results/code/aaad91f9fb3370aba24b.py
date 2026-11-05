import time
import random
import numpy as np

def _objective_fast(A):
    """Fast objective using numpy for vectorized computation."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # All pairwise sums (i <= j)
    i_idx = np.repeat(np.arange(n), n)
    j_idx = np.tile(np.arange(n), n)
    mask = i_idx <= j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _generate_structural_seed(rng, strategy):
    """Generate a structurally informed initial set."""
    if strategy == 0:
        # Small consecutive integers
        return list(range(1, 21))
    elif strategy == 1:
        # Geometric progression with ratio ~1.5
        return [int(round(1.5 ** i)) for i in range(20)]
    elif strategy == 2:
        # Powers of 2
        return [2 ** i for i in range(20)]
    elif strategy == 3:
        # Mixed: small base + geometric tail
        base = list(range(1, 11))
        tail = [int(round(10 * 1.4 ** i)) for i in range(10)]
        return base + tail
    elif strategy == 4:
        # Arithmetic progression
        return [1 + i * 2 for i in range(20)]
    elif strategy == 5:
        # Triangular numbers
        return [i * (i + 1) // 2 for i in range(1, 21)]
    else:
        # Random but bounded
        return sorted(rng.sample(range(1, 200), 20))

def _perturb(A, rng, perturbation_type):
    """Generate a perturbed version of A."""
    A = list(A)
    n = len(A)
    
    if perturbation_type == 0:
        # Swap two elements
        i, j = rng.sample(range(n), 2)
        A[i], A[j] = A[j], A[i]
    elif perturbation_type == 1:
        # Replace one element with a nearby value
        i = rng.randint(0, n - 1)
        old = A[i]
        delta = rng.randint(-20, 20)
        new_val = max(1, min(10000, old + delta))
        if new_val not in A or new_val == old:
            A[i] = new_val
    elif perturbation_type == 2:
        # Replace one element with a scaled value
        i = rng.randint(0, n - 1)
        old = A[i]
        scale = rng.choice([0.5, 0.7, 1.3, 1.5, 2.0, 3.0])
        new_val = int(round(old * scale))
        new_val = max(1, min(10000, new_val))
        if new_val not in A or new_val == old:
            A[i] = new_val
    elif perturbation_type == 3:
        # Insert a new value, remove a random one
        remove_idx = rng.randint(0, n - 1)
        A.pop(remove_idx)
        new_val = rng.randint(1, 10000)
        if new_val not in A:
            A.append(new_val)
            A.sort()
    elif perturbation_type == 4:
        # Shift all elements by a small amount
        shift = rng.randint(-3, 3)
        if shift != 0:
            new_A = [max(1, min(10000, x + shift)) for x in A]
            # Ensure distinctness
            if len(set(new_A)) == n:
                A = new_A
                A.sort()
    elif perturbation_type == 5:
        # Replace with a value from a structural pattern
        i = rng.randint(0, n - 1)
        patterns = [
            lambda k: k + 1,
            lambda k: 2 ** k,
            lambda k: k * (k + 1) // 2,
            lambda k: int(round(1.5 ** k)),
        ]
        pat = rng.choice(patterns)
        k = rng.randint(1, 30)
        new_val = pat(k)
        if 1 <= new_val <= 10000 and new_val not in A:
            A[i] = new_val
            A.sort()
    
    return A

def _simulated_annealing(A, deadline, rng, t0=1000, t1=100, alpha=0.9995):
    """Simulated annealing to minimize the objective."""
    A = list(A)
    A.sort()
    cur_obj = _objective_fast(A)
    best_obj = cur_obj
    best_set = A[:]
    cur_temp = t0
    step = 0
    
    while time.time() < deadline and cur_temp > t1:
        # Choose perturbation type
        ptype = rng.randint(0, 5)
        A_new = _perturb(A, rng, ptype)
        
        # Validate
        if len(A_new) != 20 or len(set(A_new)) != 20 or any(x < 1 or x > 10000 for x in A_new):
            cur_temp *= alpha
            step += 1
            continue
        
        new_obj = _objective_fast(A_new)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / cur_temp):
            A = A_new
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_set = A[:]
        
        cur_temp *= alpha
        step += 1
    
    return best_set, best_obj

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    
    rng = random.Random(42)
    np.random.seed(42)
    
    best_set = None
    best_obj = float('inf')
    
    # Generate diverse structural seeds
    strategies = [0, 1, 2, 3, 4, 5]
    
    # Phase 1: Run SA on each structural seed
    per_seed_time = 12.0
    for strat in strategies:
        if time.time() > deadline - 10:
            break
        
        seed = _generate_structural_seed(rng, strat)
        # Ensure validity
        seed = sorted(set(x for x in seed if 1 <= x <= 10000))
        while len(seed) < 20:
            seed.append(rng.randint(1, 10000))
            seed = sorted(set(seed))
        seed = seed[:20]
        
        seed_deadline = min(time.time() + per_seed_time, deadline - 5)
        result, obj = _simulated_annealing(seed, seed_deadline, rng)
        
        if obj < best_obj:
            best_obj = obj
            best_set = result[:]
    
    # Phase 2: Run SA on the best found so far with more time
    if best_set is not None and time.time() < deadline - 10:
        phase2_deadline = deadline - 5
        result, obj = _simulated_annealing(best_set, phase2_deadline, rng, t0=500, t1=50, alpha=0.9998)
        if obj < best_obj:
            best_obj = obj
            best_set = result[:]
    
    # Phase 3: Final local refinement
    if best_set is not None and time.time() < deadline - 3:
        # Try simple local moves
        A = list(best_set)
        cur_obj = _objective_fast(A)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(20):
                old_val = A[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000 or new_val in A:
                        continue
                    A[i] = new_val
                    new_obj = _objective_fast(A)
                    if new_obj < cur_obj:
                        cur_obj = new_obj
                        improved = True
                    A[i] = old_val
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
    
    return best_set
