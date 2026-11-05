import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Precompute weights and gcd table for speed
    weights = [0.0] * 512
    for a in range(512):
        weights[a] = 1.0 / (512 - a) if a < 512 else 0.0
    
    # Precompute gcd for all pairs (for fast coprimality checks)
    gcd_table = [[0] * 512 for _ in range(512)]
    for a in range(512):
        for b in range(512):
            gcd_table[a][b] = math.gcd(a, b)
    
    def is_coprime_with_all(val, chosen_set):
        for b in chosen_set:
            if gcd_table[val][b] != 1:
                return False
        return True
    
    def score_of(s):
        return sum(weights[a] for a in s)
    
    # Greedy initialization: pick highest weight (largest a) first
    chosen = []
    for a in range(511, 0, -1):
        if is_coprime_with_all(a, chosen):
            chosen.append(a)
    
    best_set = chosen[:]
    best_score = score_of(chosen)
    
    # Merge pass: try to replace two coprime elements with their product if < 512
    # and the gain is positive
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                a, b = best_set[i], best_set[j]
                prod = a * b
                if prod >= 512:
                    continue
                if prod == a or prod == b:
                    continue
                # Check if prod is coprime with all other elements
                others = [x for k, x in enumerate(best_set) if k != i and k != j]
                if not is_coprime_with_all(prod, others):
                    continue
                # Check gain
                gain = weights[prod] - weights[a] - weights[b]
                if gain > 1e-12:
                    new_set = others + [prod]
                    new_score = score_of(new_set)
                    if new_score > best_score + 1e-12:
                        best_set = new_set
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
            if time.time() > deadline:
                break
    
    # Local search with multiple move types
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Move 1: Remove one element and re-greedy-fill
        for i, a in enumerate(best_set):
            remaining = [x for j, x in enumerate(best_set) if j != i]
            new_set = remaining[:]
            for b in range(511, 0, -1):
                if b in new_set:
                    continue
                if is_coprime_with_all(b, new_set):
                    new_set.append(b)
            new_score = score_of(new_set)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_set = new_set
                improved = True
                break
            if time.time() > deadline:
                break
        
        if improved or time.time() > deadline:
            continue
        
        # Move 2: Swap one element for another
        for i, a in enumerate(best_set):
            for b in range(511, 0, -1):
                if b in best_set:
                    continue
                others = [x for j, x in enumerate(best_set) if j != i]
                if not is_coprime_with_all(b, others):
                    continue
                new_set = others + [b]
                new_score = score_of(new_set)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_set = new_set
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        
        if improved or time.time() > deadline:
            continue
        
        # Move 3: Add one element if possible
        for b in range(511, 0, -1):
            if b in best_set:
                continue
            if is_coprime_with_all(b, best_set):
                new_set = best_set + [b]
                new_score = score_of(new_set)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_set = new_set
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    return best_set
