import time
import random
import math

def _count_comparable_pairs(family):
    """Count unordered pairs (i,j) where family[i] is a strict subset of family[j] or vice versa."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family[i], family[j]
            if (a & b) == a:  # a subset of b
                if a != b:
                    count += 1
            elif (a & b) == b:  # b subset of a
                if a != b:
                    count += 1
    return count

def _greedy_initial():
    """Greedy: repeatedly pick the mask maximizing comparable pairs with current set."""
    N = 12
    M = 40
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    
    F = []
    F_set = set()
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in F_set:
                continue
            score = 0
            for m in F:
                if m != mask:
                    if (m & mask) == m or (m & mask) == mask:
                        score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
                if best_score == len(F):
                    break
        if best_mask != -1:
            F.append(best_mask)
            F_set.add(best_mask)
        else:
            # fallback
            for mask in all_masks:
                if mask not in F_set:
                    F.append(mask)
                    F_set.add(mask)
                    break
    return F

def _local_search(F, time_budget):
    """Swap-based local search with simulated annealing."""
    N = 12
    M = len(F)
    deadline = time.time() + time_budget
    F = F[:]
    F_set = set(F)
    
    def calc_score(fam):
        s = 0
        for i in range(M):
            for j in range(i + 1, M):
                a, b = fam[i], fam[j]
                if a != b and ((a & b) == a or (a & b) == b):
                    s += 1
        return s
    
    current_score = calc_score(F)
    best_score = current_score
    best_F = F[:]
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9997
    
    while time.time() < deadline and T > T_min:
        idx = random.randint(0, M - 1)
        old_m = F[idx]
        
        # Generate new mask by flipping 1-3 bits
        r = random.random()
        if r < 0.6:
            flips = 1
        elif r < 0.9:
            flips = 2
        else:
            flips = 3
        
        new_m = old_m
        for _ in range(flips):
            bit = random.randint(0, N - 1)
            new_m ^= (1 << bit)
        
        if new_m in F_set:
            T *= alpha
            continue
        
        # Compute delta
        delta = 0
        for i, m in enumerate(F):
            if i == idx:
                continue
            # Remove old contribution
            if old_m != m and ((old_m & m) == old_m or (old_m & m) == m):
                delta -= 1
            # Add new contribution
            if new_m != m and ((new_m & m) == new_m or (new_m & m) == m):
                delta += 1
        
        if delta > 0 or random.random() < math.exp(delta / T):
            F_set.remove(old_m)
            F_set.add(new_m)
            F[idx] = new_m
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_F = F[:]
        
        T *= alpha
    
    return best_F, best_score

def construct():
    N = 12
    M = 40
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    best_F = None
    best_score = -1
    
    # Multiple restarts with greedy init + local search
    num_restarts = 0
    while time.time() < deadline:
        # Greedy initialization
        F = _greedy_initial()
        num_restarts += 1
        
        # Local search with decreasing time budget per iteration
        elapsed = time.time() - (deadline - 55)
        remaining = deadline - time.time()
        if remaining < 3:
            # Not enough time for another full local search
            break
        
        # Allocate time: first few restarts get more, later ones less
        if num_restarts <= 3:
            ls_time = min(remaining - 2, 10)
        elif num_restarts <= 6:
            ls_time = min(remaining - 2, 5)
        else:
            ls_time = min(remaining - 2, 2)
        
        if ls_time < 0.5:
            break
        
        F, score = _local_search(F, ls_time)
        
        if score > best_score:
            best_score = score
            best_F = F[:]
        
        # Check if we've hit a good plateau; reduce restarts
        if best_score >= 100 and time.time() > (deadline - 55) + 20:
            # We're doing well, spend remaining time on one more long polish
            if time.time() < deadline - 2:
                F, score = _local_search(best_F, max(0, deadline - 2 - time.time()))
                if score > best_score:
                    best_score = score
                    best_F = F[:]
            break
    
    # If we never ran local search, at least use the greedy result
    if best_F is None:
        best_F = _greedy_initial()
    
    # Final polish: one more local search pass on the best found
    if time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 0.5:
            best_F, best_score = _local_search(best_F, remaining)
    
    # Convert bitmasks to list of lists of integers
    result = []
    for mask in best_F:
        subset = [i for i in range(N) if (mask >> i) & 1]
        result.append(subset)
    
    return result
