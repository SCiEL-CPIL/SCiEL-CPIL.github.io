import numpy as np
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 20
    max_val = 10000

    # Fast evaluation using NumPy
    def evaluate(A):
        A = np.array(sorted(A), dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        unique_sums = np.unique(sums)
        unique_prods = np.unique(prods)
        return max(len(unique_sums), len(unique_prods))

    best_score = float('inf')
    best_A = None

    # Generate structured candidates
    candidates = []

    # 1. AP + GP hybrid candidates
    def build_hybrid(ap_start, ap_step, ap_len, gp_base, gp_ratio, gp_len):
        ap_terms = set()
        for i in range(ap_len):
            val = ap_start + i * ap_step
            if 1 <= val <= max_val:
                ap_terms.add(val)
        gp_terms = set()
        val = gp_base
        for i in range(gp_len):
            if 1 <= val <= max_val:
                gp_terms.add(val)
            else:
                break
            val *= gp_ratio
        combined = ap_terms | gp_terms
        if len(combined) > n:
            combined = set(sorted(combined)[:n])
        if len(combined) < n:
            for x in range(1, max_val + 1):
                if x not in combined:
                    temp = x
                    for p in [2, 3, 5, 7]:
                        while temp % p == 0:
                            temp //= p
                    if temp == 1:
                        combined.add(x)
                        if len(combined) == n:
                            break
        if len(combined) != n:
            return None
        return sorted(combined)

    # Generate AP+GP hybrids
    for ap_len in range(6, 15):
        for gp_len in range(6, 15):
            if ap_len + gp_len > n + 4:
                continue
            for ap_start in range(1, 20):
                for ap_step in range(1, 10):
                    for gp_base in range(1, 15):
                        for gp_ratio in [2, 3, 4]:
                            A = build_hybrid(ap_start, ap_step, ap_len, gp_base, gp_ratio, gp_len)
                            if A is not None:
                                candidates.append(A)
            if len(candidates) > 2000:
                break
        if len(candidates) > 2000:
            break

    # 2. Pure APs
    for start in range(1, 20):
        for step in range(1, 10):
            A = [start + i * step for i in range(n)]
            if max(A) <= max_val:
                candidates.append(sorted(A))

    # 3. Perturbed APs
    for start in [1, 2, 3]:
        for step in [1, 2, 3]:
            base_cand = [start + i * step for i in range(n)]
            for _ in range(5):
                cand = list(base_cand)
                for i in range(n):
                    if random.random() < 0.3:
                        offset = random.randint(-2, 2)
                        new_val = cand[i] + offset
                        if 1 <= new_val <= max_val and new_val not in cand:
                            cand[i] = new_val
                if len(set(cand)) == n:
                    candidates.append(sorted(cand))

    # 4. Random small integers
    for _ in range(30):
        cand = sorted(random.sample(range(1, 100), n))
        candidates.append(cand)

    # 5. Symmetric sets
    for center in [10, 20, 50]:
        for spread in [1, 2, 3]:
            cand = []
            for i in range(n):
                offset = i - n // 2
                val = center + offset * spread
                if 1 <= val <= max_val:
                    cand.append(val)
            if len(set(cand)) == n:
                candidates.append(sorted(cand))

    # Evaluate all candidates
    for A in candidates:
        if len(set(A)) != n or max(A) > max_val:
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_A = list(A)

    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = evaluate(best_A)

    # Build pool for local search
    pool_set = set()
    for x in range(1, 200):
        pool_set.add(x)
    for x in range(1, max_val + 1):
        temp = x
        for p in [2, 3, 5, 7]:
            while temp % p == 0:
                temp //= p
        if temp == 1 and x <= 500:
            pool_set.add(x)
    pool = sorted(pool_set)

    # Local search with multiple move types
    current_A = list(best_A)
    current_score = best_score

    T = 15.0
    cooling_rate = 0.998
    min_T = 0.5
    no_improve = 0
    max_no_improve = 8000

    while T > min_T and no_improve < max_no_improve:
        if time.time() > deadline - 5.0:
            break

        move_type = random.random()
        new_set = list(current_A)

        if move_type < 0.35:
            # Nudge: change one element slightly
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            for _ in range(5):
                delta = random.randint(-10, 10)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if 1 <= new_val <= max_val and new_val not in new_set:
                    new_set[idx] = new_val
                    break
            if new_set == current_A:
                continue

        elif move_type < 0.6:
            # Swap: exchange two elements
            i, j = random.sample(range(n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]

        elif move_type < 0.85:
            # Factor-aware replace: pick a value that shares factors with existing elements
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            # Pick a random existing element and create a related value
            ref_idx = random.randint(0, n - 1)
            ref_val = new_set[ref_idx]
            # Try multiples or divisors
            candidates_for_replace = []
            for mult in [2, 3, 4, 5]:
                v = ref_val * mult
                if 1 <= v <= max_val and v not in new_set and v != old_val:
                    candidates_for_replace.append(v)
            for div in [2, 3, 4, 5]:
                if ref_val % div == 0:
                    v = ref_val // div
                    if 1 <= v <= max_val and v not in new_set and v != old_val:
                        candidates_for_replace.append(v)
            if candidates_for_replace:
                new_val = random.choice(candidates_for_replace)
                new_set[idx] = new_val
            else:
                # Fallback to pool
                for _ in range(10):
                    new_val = random.choice(pool)
                    if new_val != old_val and new_val not in new_set:
                        new_set[idx] = new_val
                        break
                else:
                    continue

        else:
            # Random replace from pool
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            for _ in range(10):
                if random.random() < 0.7:
                    new_val = random.randint(1, 100)
                else:
                    new_val = random.randint(1, 500)
                if new_val != old_val and new_val not in new_set:
                    new_set[idx] = new_val
                    break
            else:
                continue

        # Evaluate neighbor
        new_score = evaluate(new_set)

        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score < 0:
            current_A = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
                no_improve = 0
            else:
                no_improve += 1
        else:
            if random.random() < math.exp(-delta_score / T):
                current_A = new_set
                current_score = new_score
                no_improve += 1
            else:
                no_improve += 1

        T *= cooling_rate

    # Final polish: try all single-element replacements with pool values
    if time.time() < deadline - 3.0:
        for idx in range(n):
            old_val = best_A[idx]
            for new_val in pool[:50]:  # Try smallest pool values
                if new_val == old_val or new_val in best_A:
                    continue
                trial = list(best_A)
                trial[idx] = new_val
                score = evaluate(trial)
                if score < best_score:
                    best_score = score
                    best_A = trial

    return best_A
