import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    best_score = -1
    best_set = [0, 1, 3]
    score_cache = {}

    def compute_score(U):
        key = tuple(sorted(set(U)))
        if key in score_cache:
            return score_cache[key]
        
        U = list(key)
        if not U or 0 not in U:
            score_cache[key] = -1
            return -1
        if len(U) < 2:
            score_cache[key] = 0
            return 0
        max_u = max(U)
        if max_u == 0:
            score_cache[key] = 0
            return 0
        
        # Use NumPy for fast sum/diff set calculation
        arr = np.array(U, dtype=np.int64)
        # Sums
        sums = np.add.outer(arr, arr)
        n_sum = len(np.unique(sums))
        
        # Diffs
        diffs = np.subtract.outer(arr, arr)
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            score_cache[key] = -1
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            score_cache[key] = 0
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            score_cache[key] = 1.0
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        score_cache[key] = score
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Mersenne-like: 2^k - 1
    for k in range(1, 15):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = val * 2 + 1
        candidates.append(U)
        
    # 2. Powers of 2
    for k in range(1, 15):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val *= 2
        candidates.append(U)
        
    # 3. Random sparse sets
    for _ in range(50):
        n = random.randint(3, 12)
        max_val = random.randint(10, 1000)
        U = [0]
        current = 0
        for _ in range(n-1):
            gap = random.randint(1, max_val // n + 10)
            current += gap
            U.append(current)
        candidates.append(U)
        
    # 4. Arithmetic Progressions (Baseline)
    for n in range(3, 15):
        for step in range(1, 10):
            U = list(range(0, n * step, step))
            candidates.append(U)
            
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        try_set(cand)

    # Phase 2: Hill Climbing
    def mutate_set(U, mutation_rate=0.2):
        if time.time() > deadline - 5:
            return U
        
        U = list(U)
        n = len(U)
        if n < 2:
            return U
        
        new_U = U.copy()
        
        # Mutate a few elements
        num_mutations = max(1, int(n * mutation_rate))
        non_zero_indices = [i for i in range(n) if U[i] != 0]
        
        for _ in range(num_mutations):
            if not non_zero_indices:
                break
            idx = random.choice(non_zero_indices)
            old_val = U[idx]
            
            max_val = max(U)
            # Local move
            if random.random() < 0.7:
                # Small perturbation
                delta = random.randint(-max(1, int(old_val*0.2)), max(1, int(old_val*0.2)))
                new_val = max(1, old_val + delta)
            else:
                # Larger jump
                new_val = random.randint(1, max_val + 20)
            
            # Avoid duplicates
            if new_val in new_U and new_val != old_val:
                continue
                
            new_U[idx] = new_val
        
        if 0 not in new_U:
            new_U.append(0)
        return sorted(set(new_U))

    def hill_climb(start_set, max_iters=2000):
        nonlocal best_score, best_set
        curr = sorted(set(start_set))
        curr_score = compute_score(curr)
        
        if curr_score > best_score:
            best_score = curr_score
            best_set = list(curr)
            
        no_improve_count = 0
        max_no_improve = 50
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Generate neighbor
            if random.random() < 0.5:
                neighbor = mutate_set(curr, mutation_rate=0.1)
            else:
                neighbor = mutate_set(curr, mutation_rate=0.3)
            
            if neighbor == curr:
                continue
                
            n_score = compute_score(neighbor)
            
            if n_score > curr_score:
                curr = neighbor
                curr_score = n_score
                no_improve_count = 0
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_set = list(curr)
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            # Occasionally add/remove element
            if random.random() < 0.05 and len(curr) < 15:
                max_val = max(curr)
                new_el = random.randint(1, max_val + 10)
                if new_el not in curr:
                    cand = sorted(curr + [new_el])
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        no_improve_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_set = list(curr)
            elif random.random() < 0.05 and len(curr) > 3:
                non_zero = [x for x in curr if x != 0]
                if non_zero:
                    rem = random.choice(non_zero)
                    cand = [x for x in curr if x != rem]
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        no_improve_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_set = list(curr)
        
        return curr

    # Run Hill Climbing from best candidate and a few others
    if time.time() < deadline - 10:
        hill_climb(best_set, max_iters=5000)
        
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Random start
        n = random.randint(3, 10)
        U = [0]
        current = 0
        for _ in range(n-1):
            current += random.randint(1, 50)
            U.append(current)
        hill_climb(U, max_iters=2000)

    # Final Polish: Try single element replacements
    if time.time() < deadline - 2:
        max_cur = max(best_set)
        search_limit = max_cur + 5
        
        for _ in range(2):
            if time.time() > deadline - 2:
                break
            changed = False
            
            # Try adding
            for new_el in range(1, search_limit + 1):
                if new_el in best_set:
                    continue
                cand = sorted(best_set + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    changed = True
                    break
            
            if changed:
                continue

            # Try replacing
            for i in range(len(best_set)):
                if best_set[i] == 0:
                    continue
                for new_el in range(1, search_limit + 1):
                    if new_el in best_set:
                        continue
                    cand = list(best_set)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand or len(cand) < 2:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        changed = True
                        break
                if changed:
                    break
                    
            if not changed:
                break

    return list(best_set)
