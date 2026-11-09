import numpy as np
import time
import random
from math import gcd

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe
    
    # 1. Generate pool of candidate numbers
    # Using 5-smooth numbers (Hamming numbers) as they have many multiplicative relations
    primes = [2, 3, 5]
    max_val = 10000
    pool_set = {1}
    for p in primes:
        current = list(pool_set)
        for x in current:
            val = x
            while val * p <= max_val:
                val *= p
                pool_set.add(val)
    
    pool = sorted(pool_set)
    n_pool = len(pool)
    pool_index = {v: i for i, v in enumerate(pool)}

    # Precompute factor-aware candidates for each pool element
    factor_candidates = {}
    for i, a in enumerate(pool):
        cands = []
        for j, b in enumerate(pool):
            if i != j and gcd(a, b) > 1:
                cands.append(j)
        if not cands:
            cands = list(range(n_pool))
        factor_candidates[i] = cands

    # 2. Fast evaluation
    def evaluate(A):
        arr = np.array(A, dtype=np.int64)
        n = len(arr)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        prods = arr[i_idx] * arr[j_idx]
        s_set = set(sums.tolist())
        p_set = set(prods.tolist())
        return max(len(s_set), len(p_set))

    # 3. Simulated Annealing with factor-aware moves
    best_score = float('inf')
    best_A = None

    initial_states = []
    initial_states.append(sorted(pool[:20]))
    p2 = [2**i for i in range(14)]
    others = [x for x in pool if x not in p2]
    initial_states.append(sorted(p2 + others[:6]))
    initial_states.append(sorted(random.sample(pool, 20)))
    initial_states.append(sorted(random.sample(pool, 20)))

    def sa_run(init_A, temp_start, cooling_rate, min_temp):
        nonlocal best_score, best_A
        current_A = list(init_A)
        current_score = evaluate(current_A)

        if current_score < best_score:
            best_score = current_score
            best_A = list(current_A)

        temp = temp_start
        while temp > min_temp:
            if time.time() > deadline:
                break

            idx = random.randint(0, 19)
            old_val = current_A[idx]
            old_idx = pool_index.get(old_val)

            if old_idx is not None and random.random() < 0.7:
                cands = factor_candidates[old_idx]
                new_idx = random.choice(cands)
                new_val = pool[new_idx]
            else:
                new_idx = random.randint(0, n_pool - 1)
                new_val = pool[new_idx]

            if new_val == old_val or new_val in current_A:
                continue

            current_A[idx] = new_val
            new_score = evaluate(current_A)
            delta = new_score - current_score

            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
            else:
                current_A[idx] = old_val

            temp *= cooling_rate

        return current_A, current_score

    for init_A in initial_states:
        if time.time() > deadline:
            break
        sa_run(init_A, 100.0, 0.9995, 0.1)

    # 4. Final polish pass: tighter SA on best solution
    if best_A is not None and time.time() < deadline - 5.0:
        sa_run(best_A, 5.0, 0.999, 0.01)

    return best_A
