import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    best_score = -1
    best_set = [0]

    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # Vectorized sum and diff computation
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Generate diverse initial candidates
    # 1. Mersenne numbers (2^k - 1)
    for k in range(1, 15):
        cand = [0] + [2**i - 1 for i in range(1, k+1)]
        try_set(cand)
        
    # 2. Powers of 2
    for k in range(1, 15):
        cand = [0] + [2**i for i in range(k)]
        try_set(cand)
        
    # 3. Golomb rulers (greedy)
    def generate_golomb_like(n, max_val=5000):
        U = [0, 1]
        for _ in range(2, n):
            best_pos = -1
            best_unique = -1
            current_diffs = set()
            for i in range(len(U)):
                for j in range(len(U)):
                    current_diffs.add(U[i] - U[j])
            for pos in range(2, max_val + 1):
                if pos in U:
                    continue
                new_diffs = set()
                for u in U:
                    new_diffs.add(pos - u)
                    new_diffs.add(u - pos)
                new_unique = len(new_diffs - current_diffs)
                if new_unique > best_unique:
                    best_unique = new_unique
                    best_pos = pos
                if best_unique == len(U): # Perfect extension
                    break
            if best_pos > 0:
                U.append(best_pos)
            else:
                break
        return U
        
    for n in range(3, 15):
        g = generate_golomb_like(n, 5000)
        if len(g) == n:
            try_set(g)

    # 4. Random sets
    for _ in range(100):
        n = random.randint(3, 12)
        cand = [0]
        for i in range(n-1):
            cand.append(random.randint(1, 500))
        try_set(cand)

    # Phase 2: Local Search
    current = list(best_set)
    no_improve = 0
    max_no_improve = 1000

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n_cur = len(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        for i in range(1, min(max_cur + 100, 2000)):
            if i not in current:
                candidates_to_add.add(i)
        for _ in range(50):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 500))
        
        for new_el in candidates_to_add:
            if new_el in current or new_el < 1:
                continue
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                current = list(cand)
                improved = True
                no_improve = 0
                break
        
        # 2. Try replacing an element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                for delta in range(-50, 51):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current or new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.5:
                # Swap
                if len(current) > 2:
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                # Add random
                new_el = random.randint(1, max_cur + 200)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    try_set(cand)
                    current = list(best_set)

    return list(best_set)
