"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _max_sidon_size_fast(arr):
    """Fast exact largest Sidon subset size using bitset collision detection."""
    n = len(arr)
    if n <= 2:
        return n
    # Precompute collision bitmask: col[i] has bit j set if arr[i]+arr[j] collides with some pair
    # Actually, for subset checking we need: for a candidate subset, check if all pair sums are unique.
    # Better approach: for each pair (i,j) with i<=j, compute sum. Two pairs collide if same sum.
    # For a subset S, it's Sidon iff no two distinct pairs in S have the same sum.
    
    # Precompute: for each pair (i,j), i<j, the sum arr[i]+arr[j].
    # Also diagonal sums: 2*arr[i].
    # Two pairs collide if they have the same sum.
    
    # For efficiency with n=12, use a different approach:
    # Check all subsets of size k from n down to 3.
    # For each subset, verify Sidon property.
    
    # Actually, let's use a smarter approach:
    # A set is NOT Sidon if there exist distinct pairs (a,b) and (c,d) with a+b=c+d.
    # This includes a=a, b=b type (diagonal) and cross terms.
    
    # For n=12, checking all C(12,k) subsets for k=12,11,...,3:
    # C(12,12)=1, C(12,11)=12, C(12,10)=66, C(12,9)=220, C(12,8)=495
    # Total ~800 subsets to check. Each check is O(k^2) ~ 144 ops. Very fast.
    
    def is_sidon(indices):
        """Check if the elements at given indices form a Sidon set."""
        vals = [arr[i] for i in indices]
        k = len(vals)
        seen = set()
        for i in range(k):
            for j in range(i, k):
                s = vals[i] + vals[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    # Try from largest to smallest
    for k in range(n, 2, -1):
        for combo in combinations(range(n), k):
            if is_sidon(combo):
                return k
    return 2  # Every 2-element set is Sidon

def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Strategy: We want score 3. This means no 4-element subset is Sidon.
    # Key structural idea: use sets with high additive energy (many sum collisions).
    # Dense clusters force collisions.
    
    candidates = []
    
    # 1. Dense consecutive blocks (known to have low Sidon number)
    for start in range(0, 20):
        candidates.append(list(range(start, start + 12)))
    
    # 2. Two-cluster structures: 6+6 with small gap
    for gap in range(3, 25):
        for off in range(0, 4):
            cand = list(range(0, 6)) + [gap + off + i for i in range(6)]
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # 3. Three-cluster: 4+4+4
    for g1 in range(3, 12):
        for g2 in range(g1, g1 + 12):
            cand = list(range(0, 4)) + [g1 + i for i in range(4)] + [g1+g2 + i for i in range(4)]
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # 4. Four-cluster: 3+3+3+3
    for g1 in range(3, 10):
        for g2 in range(g1, g1 + 10):
            for g3 in range(g2, g2 + 10):
                cand = list(range(0, 3)) + [g1+i for i in range(3)] + \
                       [g1+g2+i for i in range(3)] + [g1+g2+g3+i for i in range(3)]
                if len(set(cand)) == 12 and max(cand) <= 128:
                    candidates.append(cand)
    
    # 5. Modular structures: clusters of 3 with varying steps
    for base in range(0, 8):
        for s1 in range(3, 12):
            for s2 in range(3, 12):
                cand = [base + (i//3)*s1 + (i%3) for i in range(6)]
                cand += [base + 5*s1 + (i%3)*s2 + ((i//3)-2)*s2 for i in range(6)]
                # Simpler: just use arithmetic-like patterns
                cand2 = []
                for i in range(12):
                    block = i // 3
                    pos = i % 3
                    val = base + block * s1 + pos
                    if val > 128:
                        break
                    cand2.append(val)
                if len(cand2) == 12 and len(set(cand2)) == 12 and max(cand2) <= 128:
                    candidates.append(cand2)
    
    # 6. Random dense sets
    random.seed(42)
    for _ in range(3000):
        if time.time() > deadline - 20:
            break
        strategy = random.randint(0, 4)
        if strategy == 0:
            # Very dense in small range
            start = random.randint(0, 25)
            end = min(start + random.randint(12, 20), 129)
            if end - start >= 12:
                cand = sorted(random.sample(range(start, end), 12))
            else:
                continue
        elif strategy == 1:
            # Two tight clusters
            c1s = random.randint(0, 15)
            c2s = c1s + random.randint(5, 20)
            try:
                cand = sorted(random.sample(range(c1s, c1s+6), 6) + 
                             random.sample(range(c2s, min(c2s+6, 129)), 6))
            except:
                continue
        elif strategy == 2:
            # Three tight clusters
            c1s = random.randint(0, 10)
            c2s = c1s + random.randint(4, 12)
            c3s = c2s + random.randint(4, 12)
            if c3s + 4 > 129:
                continue
            try:
                cand = sorted(random.sample(range(c1s, c1s+4), 4) + 
                             random.sample(range(c2s, c2s+4), 4) +
                             random.sample(range(c3s, c3s+4), 4))
            except:
                continue
        elif strategy == 3:
            # Modular with small step
            base = random.randint(0, 5)
            step = random.randint(3, 7)
            cand = [base + (i // 3) * step + (i % 3) for i in range(12)]
            if max(cand) > 128 or len(set(cand)) != 12:
                continue
        else:
            # Random in medium range
            start = random.randint(0, 50)
            cand = sorted(random.sample(range(start, min(start + 30, 129)), 12))
        
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        sz = _max_sidon_size_fast(cand)
        if sz < best_score:
            best_score = sz
            best_set = cand[:]
            if sz == 3:
                # Found score 3 - this is optimal (can't go below 3)
                # Do a quick local search to confirm stability
                break
    
    # If we found score 3, verify and try to confirm with local search
    if best_score == 3:
        # Quick verification: ensure it's really 3
        # Try a few local perturbations to see if we can maintain 3
        random.seed(999)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            trial = best_set[:]
            i = random.randint(0, 11)
            old = trial[i]
            trial[i] = random.randint(0, 128)
            if len(set(trial)) == 12 and all(0 <= x <= 128 for x in trial):
                sz = _max_sidon_size_fast(trial)
                if sz == 3:
                    best_set = trial[:]
    
    # If still > 3, do local search
    if best_score > 3:
        current = best_set[:]
        current_score = best_score
        random.seed(123)
        
        while time.time() < deadline - 3:
            improved = False
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                sz = _max_sidon_size_fast(current)
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
                for _ in range(20):
                    if time.time() > deadline - 3:
                        break
                    current = sorted(random.sample(range(129), 12))
                    sz = _max_sidon_size_fast(current)
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
