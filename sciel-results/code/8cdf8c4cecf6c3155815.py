import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_score = -1
    best_set = [0]

    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        
        # Vectorized sumset and diffset
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
        if not cand:
            return
        cand = sorted(set(cand))
        if 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Try known good Golomb rulers as initial candidates
    # Precomputed optimal/near-optimal Golomb rulers for small n
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 36, 40],
        [0, 1, 4, 10, 18, 23, 29, 36, 45, 52],
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline - 20:
            break
        try_set(gr)

    # Phase 2: Exhaustive search for very small M (to find any hidden gems < 16)
    # Note: This is fast enough for M up to ~15-16 depending on pruning, but let's be conservative
    for M in range(1, 13):
        if time.time() > deadline - 20:
            break
        # Only check sets that contain 0 and at least one other element
        # Bit 0 corresponds to element 1, ..., Bit M-1 corresponds to element M
        # Element 0 is always present
        for mask in range(1, 1 << M):
            if time.time() > deadline - 20:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)

    # Phase 3: Heuristic Local Search
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        # Nearby positions
        for i in range(1, max_cur + 40):
            if i not in current:
                candidates_to_add.add(i)
        # Larger jumps
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 300))
        # Random small positions
        for _ in range(20):
            candidates_to_add.add(random.randint(1, max_cur + 10))
            
        for new_el in candidates_to_add:
            if new_el in current:
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
                # Wider neighborhood for replacement
                for delta in range(-40, 41):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
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
        
        # 4. Try swapping two elements (non-zero)
        if len(current) > 3 and not improved:
            for i in range(1, len(current)):
                for j in range(i + 1, len(current)):
                    cand = list(current)
                    cand[i], cand[j] = cand[j], cand[i]
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
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local optima
            if random.random() < 0.6:
                perturbation_type = random.random()
                if perturbation_type < 0.3:
                    # Swap two random elements
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        j = random.randint(1, len(current)-1)
                        if i != j:
                            cand = list(current)
                            cand[i], cand[j] = cand[j], cand[i]
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                elif perturbation_type < 0.6:
                    # Add a random element
                    new_el = random.randint(1, max_cur + 150)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
                elif perturbation_type < 0.8:
                    # Replace a random element
                    if len(current) > 1:
                        i = random.randint(1, len(current)-1)
                        new_el = random.randint(1, max_cur + 60)
                        if new_el not in current:
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                else:
                    # Remove a random element
                    if len(current) > 3:
                        i = random.randint(1, len(current)-1)
                        cand = [x for j, x in enumerate(current) if j != i]
                        try_set(cand)
                        current = list(best_set)
        
        # If we've been stuck for a while, try a more drastic restart
        if no_improve > 100:
            # Restart from a random Golomb ruler or a random subset of best_set
            if random.random() < 0.5:
                start_idx = random.randint(0, len(golomb_rulers)-1)
                current = list(golomb_rulers[start_idx])
            else:
                # Keep 0 and a random subset of others
                others = [x for x in current if x != 0]
                if others:
                    k = random.randint(1, len(others))
                    subset = random.sample(others, k)
                    current = sorted([0] + subset)
                else:
                    current = [0, 1]
            no_improve = 0

    return best_set

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    print(f"Result: {res}")
    print(f"Score: {compute_score(res)}")
    print(f"Time: {time.time() - t0:.2f}s")
