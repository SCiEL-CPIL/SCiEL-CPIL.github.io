import numpy as np
import time
import random
from math import gcd, exp

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe (90s wall-clock limit)
    
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

    # 2. Fast pure-Python evaluation (avoids numpy overhead for 210 pairs)
    def evaluate(A):
        s_set = set()
        p_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                s_set.add(ai + aj)
                p_set.add(ai * aj)
        return max(len(s_set), len(p_set))

    # 3. Simulated Annealing with factor-aware moves and 2-swap
    best_score = float('inf')
    best_A = None

    initial_states = []
    initial_states.append(sorted(pool[:20]))
    p2 = [2**i for i in range(14)]
    others = [x for x in pool if x not in p2]
    initial_states.append(sorted(p2 + others[:6]))
    initial_states.append(sorted(random.sample(pool, 20)))
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

            # 10% chance of 2-swap move
            if random.random() < 0.10:
                i1, i2 = random.sample(range(20), 2)
                old1, old2 = current_A[i1], current_A[i2]
                
                # Pick new values for both positions
                def pick_new(idx, old_val):
                    old_idx = pool_index.get(old_val)
                    if old_idx is not None and random.random() < 0.7:
                        cands = factor_candidates[old_idx]
                        return pool[random.choice(cands)]
                    else:
                        return pool[random.randint(0, n_pool - 1)]
                
                new1 = pick_new(i1, old1)
                new2 = pick_new(i2, old2)
                
                # Ensure no duplicates and not same as old
                if new1 == new2 or new1 == old1 or new1 == old2 or new2 == old1 or new2 == old2:
                    continue
                if new1 in current_A or new2 in current_A:
                    continue
                
                current_A[i1] = new1
                current_A[i2] = new2
                new_score = evaluate(current_A)
                delta = new_score - current_score

                if delta < 0 or random.random() < exp(-delta / temp):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(current_A)
                else:
                    current_A[i1] = old1
                    current_A[i2] = old2
            else:
                # Single replacement move
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

                if delta < 0 or random.random() < exp(-delta / temp):
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

    # 4. Final polish pass: tighter SA on best solution with more iterations
    if best_A is not None and time.time() < deadline - 3.0:
        sa_run(best_A, 5.0, 0.998, 0.01)

    return best_A
