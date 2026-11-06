"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a set of integers is Sidon (all pair sums i<=j distinct)."""
    s = sorted(subset)
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _max_sidon_size(arr):
    """Find the largest Sidon subset size of arr (size <= 12)."""
    n = len(arr)
    best = 0
    # Iterate all subsets, track max Sidon size
    for mask in range(1, 1 << n):
        subset = [arr[i] for i in range(n) if mask & (1 << i)]
        if len(subset) <= best:
            continue
        if _is_sidon(subset):
            best = len(subset)
            if best == n:
                break
    return best

def _try_construct_modular(base, step, n=12):
    """Generate a modular structured set."""
    result = []
    for i in range(n):
        val = base + (i // 3) * step + (i % 3)
        result.append(val)
    return result

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Generate structured candidates that force sum collisions
    candidates = []
    
    # 1. Dense consecutive blocks
    for start in range(0, 20):
        candidates.append(list(range(start, start + 12)))
    
    # 2. Modular structures: clusters of 3 with gaps
    for base in range(0, 10):
        for step in range(3, 20):
            cand = _try_construct_modular(base, step)
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # 3. Two-cluster structures
    for gap in range(4, 30):
        for offset in range(0, 5):
            cand = list(range(0, 6)) + [gap + offset + i for i in range(6)]
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # 4. Three-cluster structures
    for g1 in range(3, 15):
        for g2 in range(g1, g1 + 15):
            cand = [0, 1, 2, g1, g1+1, g1+2, g1+g2, g1+g2+1, g1+g2+2, 
                    g1+g2+g1, g1+g2+g1+1, g1+g2+g1+2]
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # 5. Random sets with bias toward dense regions
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 30:
            break
        # Mix of strategies
        strategy = random.randint(0, 3)
        if strategy == 0:
            # Dense in small range
            start = random.randint(0, 30)
            cand = sorted(random.sample(range(start, min(start+20, 129)), 12))
        elif strategy == 1:
            # Two clusters
            c1_start = random.randint(0, 20)
            c2_start = random.randint(c1_start + 5, c1_start + 40)
            cand = sorted(random.sample(range(c1_start, c1_start+6), 6) + 
                         random.sample(range(c2_start, min(c2_start+6, 129)), 6))
        elif strategy == 2:
            # Modular with small step
            base = random.randint(0, 5)
            step = random.randint(3, 8)
            cand = [base + (i // 3) * step + (i % 3) for i in range(12)]
        else:
            # Random
            cand = sorted(random.sample(range(129), 12))
        
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        sz = _max_sidon_size(cand)
        if sz < best_score:
            best_score = sz
            best_set = cand[:]
            if sz == 3:
                # Found optimal, but let's verify and maybe find more
                # Don't break immediately - keep searching for confirmation
                pass
    
    # Local search to improve if not at 3
    if best_score > 3:
        current = best_set[:]
        current_score = best_score
        random.seed(123)
        
        while time.time() < deadline - 5:
            improved = False
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                sz = _max_sidon_size(current)
                if sz < current_score:
                    current_score = sz
                    improved = True
                    if sz == 3:
                        break
                else:
                    current[i] = old_val
            
            if current_score == 3:
                break
            if not improved:
                # Random restart
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    current = sorted(random.sample(range(129), 12))
                    sz = _max_sidon_size(current)
                    if sz < current_score:
                        current_score = sz
                        improved = True
                if not improved:
                    break
        
        if current_score < best_score:
            best_set = current[:]
            best_score = current_score
    
    return best_set
# EVOLVE-BLOCK-END
