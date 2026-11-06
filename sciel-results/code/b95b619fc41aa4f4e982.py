"""Design: Simulated Annealing + local repair for 4-AP-free set with max 3-APs."""
import time
import random

N = 256
DEADLINE = 80.0  # safety margin before 90s kill

def count_3aps(mask):
    """Count distinct 3-term APs (positive diff, counted once) in a bitmask set."""
    count = 0
    for a in range(N):
        if not (mask >> a) & 1:
            continue
        # find all b > a in set, check if (2b-a) is in set and > b
        b = a + 1
        while b < N:
            if (mask >> b) & 1:
                c = 2 * b - a
                if c < N and (mask >> c) & 1:
                    count += 1
            b += 1
    return count

def has_4ap(mask):
    """Check if mask contains a nonconstant 4-term AP."""
    for a in range(N):
        if not (mask >> a) & 1:
            continue
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if (mask >> b) & 1 and (mask >> c) & 1 and (mask >> e) & 1:
                return True
    return False

def count_4aps(mask):
    """Count nonconstant 4-term APs."""
    count = 0
    for a in range(N):
        if not (mask >> a) & 1:
            continue
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if (mask >> b) & 1 and (mask >> c) & 1 and (mask >> e) & 1:
                count += 1
    return count

def repair(mask):
    """Remove elements to eliminate all 4-APs. Greedy: remove the element in each 4-AP
    that appears in the most 4-APs (or just remove one per AP found)."""
    mask = mask
    changed = True
    while changed:
        changed = False
        # Find all 4-APs
        four_aps = []
        for a in range(N):
            if not (mask >> a) & 1:
                continue
            for d in range(1, (N - 1 - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if (mask >> b) & 1 and (mask >> c) & 1 and (mask >> e) & 1:
                    four_aps.append((a, b, c, e))
        if not four_aps:
            break
        # Count frequency of each element in 4-APs
        freq = {}
        for ap in four_aps:
            for x in ap:
                freq[x] = freq.get(x, 0) + 1
        # Remove the most frequent element
        worst = max(freq, key=lambda x: freq[x])
        mask &= ~(1 << worst)
        changed = True
    return mask

def initial_greedy():
    """Greedy construction: add elements in order, skip if it creates a 4-AP."""
    mask = 0
    for x in range(N):
        candidate = mask | (1 << x)
        if not has_4ap(candidate):
            mask = candidate
    return mask

def simulated_annealing(mask, time_budget):
    """SA: toggle bits, accept 4-AP violations with penalty, optimize 3-AP count."""
    best_mask = mask
    best_score = count_3aps(mask)
    current_mask = mask
    current_score = best_score
    temp = 10.0
    cooling = 0.9995
    start_time = time.time()
    
    # Precompute: for each element, which 4-APs does it participate in? Too expensive.
    # Instead, just check after each move.
    
    while time.time() - start_time < time_budget:
        if temp < 0.01:
            break
        # Random move: toggle a random bit
        x = random.randint(0, N - 1)
        new_mask = current_mask ^ (1 << x)
        
        # Check 4-AP constraint
        four_count = count_4aps(new_mask)
        if four_count > 0:
            # Penalty for 4-APs
            penalty = four_count * 50
        else:
            penalty = 0
        
        new_score = count_3aps(new_mask) - penalty
        
        if new_score > current_score or (random.random() < min(1.0, (new_score - current_score) / (temp + 1e-9))):
            current_mask = new_mask
            current_score = new_score
            if four_count == 0 and new_score > best_score:
                best_mask = new_mask
                best_score = new_score
        
        temp *= cooling
        
        # Periodic repair
        if four_count > 0 and random.random() < 0.1:
            repaired = repair(new_mask)
            rep_score = count_3aps(repaired)
            if rep_score > best_score:
                best_mask = repaired
                best_score = rep_score
                current_mask = repaired
                current_score = rep_score
    
    return best_mask, best_score

def local_improvement(mask):
    """Try adding elements (that don't create 4-APs) and removing elements to improve 3-AP count."""
    mask = mask
    improved = True
    while improved:
        improved = False
        # Try adding each element
        for x in range(N):
            if (mask >> x) & 1:
                continue
            candidate = mask | (1 << x)
            if not has_4ap(candidate):
                if count_3aps(candidate) > count_3aps(mask):
                    mask = candidate
                    improved = True
        # Try removing each element
        for x in range(N):
            if not (mask >> x) & 1:
                continue
            candidate = mask & ~(1 << x)
            if count_3aps(candidate) > count_3aps(mask):
                mask = candidate
                improved = True
    return mask

def construct():
    random.seed(42)
    start = time.time()
    
    # Phase 1: Greedy initial solution
    mask = initial_greedy()
    best_mask = mask
    best_score = count_3aps(mask)
    
    # Phase 2: Local improvement
    mask = local_improvement(mask)
    if count_3aps(mask) > best_score:
        best_mask = mask
        best_score = count_3aps(mask)
    
    elapsed = time.time() - start
    remaining = DEADLINE - elapsed - 2.0  # 2s safety margin
    
    if remaining > 5.0:
        # Phase 3: Simulated Annealing
        sa_mask, sa_score = simulated_annealing(best_mask, remaining * 0.7)
        if sa_score > best_score:
            best_mask = sa_mask
            best_score = sa_score
        
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed - 2.0
        
        if remaining > 3.0:
            # Phase 4: Local improvement on SA result
            mask = local_improvement(best_mask)
            if count_3aps(mask) > best_score:
                best_mask = mask
                best_score = count_3aps(mask)
    
    # Final check: ensure no 4-APs
    if has_4ap(best_mask):
        best_mask = repair(best_mask)
    
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
