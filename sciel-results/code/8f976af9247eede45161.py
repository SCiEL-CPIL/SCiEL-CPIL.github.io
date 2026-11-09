import time
import random

def _largest_sidon_subset_size_fast(ambient):
    """Compute the largest subset of ambient where all unordered pair sums (i<=j) are unique."""
    n = len(ambient)
    best = 0
    ambient_sorted = sorted(ambient)
    
    def search(start, depth, current_elems, current_sums):
        nonlocal best
        if depth > best:
            best = depth
        if depth + (n - start) <= best:
            return
        for i in range(start, n):
            x = ambient_sorted[i]
            # Check if x can be added without conflict
            ok = True
            for y in current_elems:
                if (x + y) in current_sums:
                    ok = False
                    break
            if ok and (x + x) in current_sums:
                ok = False
            if ok:
                new_sums = set(current_sums)
                for y in current_elems:
                    new_sums.add(x + y)
                new_sums.add(x + x)
                search(i + 1, depth + 1, current_elems + [x], new_sums)
    
    search(0, 0, [], set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best_set = list(range(12))
    best_score = _largest_sidon_subset_size_fast(best_set)
    
    random.seed(777)
    
    # Random search
    for _ in range(5000):
        if time.time() > deadline:
            break
        cand = random.sample(range(129), 12)
        score = _largest_sidon_subset_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            if best_score <= 4:
                break
    
    # Local search
    if best_score > 4 and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old = best_set[i]
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(0, 128)
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    score = _largest_sidon_subset_size_fast(best_set)
                    if score < best_score:
                        best_score = score
                        improved = True
                        break
                    best_set[i] = old
                if improved:
                    break
    
    return best_set
