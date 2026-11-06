import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    n = len(S)
    # Check from largest possible size down
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Key insight: We need to find a set of 12 distinct integers in [0, 128]
    # that minimizes the largest Sidon subset.
    # The seed program finds sets with max Sidon size 5 or 6.
    # We want to find a set with max Sidon size 4 if possible, or at least 5.
    
    # Strategy: Start with dense small-range sets and perturb them to break Sidon subsets.
    # A set with many 3-term arithmetic progressions is less likely to have large Sidon subsets.
    
    # First, let's try a systematic search around dense sets
    # Start with [0, 1, ..., 11] and try replacing elements
    
    base = list(range(12))
    
    # Try all single replacements first
    best_set = None
    best_val = 10  # Upper bound
    
    # Quick check on base set
    val = max_sidon_size(base)
    if val < best_val:
        best_val = val
        best_set = base[:]
    
    # Try replacing each element with values 0-20
    for i in range(12):
        if time.time() > deadline - 10:
            break
        orig = base[i]
        for new_val in range(0, 25):
            if new_val == orig:
                continue
            cand = base[:]
            cand[i] = new_val
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            v = max_sidon_size(cand)
            if v < best_val:
                best_val = v
                best_set = cand[:]
                if best_val == 4:
                    return best_set
    
    # Try double replacements
    if time.time() < deadline - 5:
        # Try pairs of replacements
        for i in range(12):
            for j in range(i+1, 12):
                if time.time() > deadline - 10:
                    break
                for new_i in range(0, 20):
                    if new_i == base[i]:
                        continue
                    for new_j in range(0, 20):
                        if new_j == base[j] or new_j == new_i:
                            continue
                        cand = base[:]
                        cand[i] = new_i
                        cand[j] = new_j
                        cand = sorted(set(cand))
                        if len(cand) != 12:
                            continue
                        v = max_sidon_size(cand)
                        if v < best_val:
                            best_val = v
                            best_set = cand[:]
                            if best_val == 4:
                                return best_set
    
    # Random search with local optimization
    if time.time() < deadline - 3:
        random.seed(42)
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            
            # Start from best_set or a random dense set
            if random.random() < 0.7:
                cand = best_set[:]
            else:
                # Random dense set
                start = random.randint(0, 10)
                cand = list(range(start, start + 12))
                if cand[-1] > 128:
                    continue
            
            # Perturb 1-3 elements
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 30)
                attempts = 0
                while new_val in cand and attempts < 20:
                    new_val = random.randint(0, 30)
                    attempts += 1
                if new_val in cand:
                    continue
                cand[idx] = new_val
            
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            
            v = max_sidon_size(cand)
            if v < best_val:
                best_val = v
                best_set = cand[:]
                if best_val == 4:
                    return best_set
    
    # If we haven't found 4, try a more targeted approach
    # Look for sets that are "AP-heavy"
    if time.time() < deadline - 3:
        # Try sets with structure like {0,1,2,3,4,5,6,7,8,9,10,x} for various x
        for x in range(12, 25):
            if time.time() > deadline - 3:
                break
            cand = [0,1,2,3,4,5,6,7,8,9,10,x]
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            v = max_sidon_size(cand)
            if v < best_val:
                best_val = v
                best_set = cand[:]
                if best_val == 4:
                    return best_set
        
        # Try {0,1,2,3,4,5,6,7,8,9,x,y}
        for x in range(10, 20):
            for y in range(x+1, 25):
                if time.time() > deadline - 3:
                    break
                cand = [0,1,2,3,4,5,6,7,8,9,x,y]
                cand = sorted(set(cand))
                if len(cand) != 12:
                    continue
                v = max_sidon_size(cand)
                if v < best_val:
                    best_val = v
                    best_set = cand[:]
                    if best_val == 4:
                        return best_set
    
    return best_set
