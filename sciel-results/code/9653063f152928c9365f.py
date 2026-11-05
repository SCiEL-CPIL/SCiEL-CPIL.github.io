import time
import random

def _max_sidon_subset_size(S):
    n = len(S)
    S_sorted = sorted(S)
    best = [0]
    
    def backtrack(start, selected, current_sums):
        if len(selected) > best[0]:
            best[0] = len(selected)
        remaining = n - start
        if len(selected) + remaining <= best[0]:
            return
        for i in range(start, n):
            val = S_sorted[i]
            new_sums = set(current_sums)
            valid = True
            for j in selected:
                s = val + S_sorted[j]
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            self_sum = val + val
            if self_sum in new_sums:
                continue
            new_sums.add(self_sum)
            selected.append(i)
            backtrack(i + 1, selected, new_sums)
            selected.pop()
    
    backtrack(0, [], set())
    return best[0]

def _evaluate(S):
    return _max_sidon_subset_size(S)

def construct():
    start_time = time.time()
    deadline = start_time + 90.0
    
    best_set = None
    best_val = 13
    
    # Generate structured candidates
    candidates = []
    for base in range(2, 12):
        for offset in range(0, base):
            s = set()
            for k in range(12):
                v = k * base + offset
                if 0 <= v <= 128:
                    s.add(v)
            if len(s) == 12:
                candidates.append(sorted(s))
    
    for _ in range(300):
        s = random.sample(range(129), 12)
        candidates.append(sorted(s))
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        val = _evaluate(cand)
        if val < best_val:
            best_val = val
            best_set = cand[:]
    
    if best_set is None:
        best_set = sorted(random.sample(range(129), 12))
        best_val = _evaluate(best_set)
    
    # Local search with simulated annealing
    current = best_set[:]
    current_val = best_val
    temp = 1.0
    cooling = 0.995
    
    while time.time() < deadline - 3:
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        if new_val == old_val or new_val in current:
            continue
        
        candidate = current[:]
        candidate[idx] = new_val
        candidate.sort()
        
        new_score = _evaluate(candidate)
        
        delta = new_score - current_val
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.01)):
            current = candidate
            current_val = new_score
            if current_val < best_val:
                best_val = current_val
                best_set = current[:]
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_set
