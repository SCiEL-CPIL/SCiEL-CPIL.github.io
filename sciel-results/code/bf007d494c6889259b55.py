"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset is Sidon: all unordered pair sums (including diagonals) are unique."""
    n = len(subset)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A, max_check=6):
    """Compute the largest Sidon subset size of A, checking up to max_check."""
    n = len(A)
    # Check from max_check down to 3 (any 3-element set might be Sidon or not)
    for k in range(min(max_check, n), 2, -1):
        # Quick upper bound: number of sums for k elements is k(k+1)/2
        # For k=6: 21 sums, range [0, 256] so possible
        # Check all C(n,k) subsets
        for sub in combinations(A, k):
            if _is_sidon(sub):
                return k
    # Check size 3
    for sub in combinations(A, 3):
        if _is_sidon(sub):
            return 3
    return 2

def _score(A):
    """Lower is better: return the max Sidon subset size."""
    return _max_sidon_size(A, max_check=6)

def _try_improve(A, deadline, rng, max_iters=50000):
    """Local search: try to reduce max Sidon subset size."""
    A = list(A)
    current_score = _score(A)
    
    best_score = current_score
    best_A = A[:]
    
    # If we already have score 4 or lower, great
    if current_score <= 4:
        return A
    
    # Target: get score to 5 (no 6-element Sidon subset)
    # Try swaps
    n = 12
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random swap: pick two indices, swap values (one stays, one changes)
        i = rng.randint(0, n - 1)
        new_val = rng.randint(0, 128)
        
        if new_val in A and A.index(new_val) != i:
            continue
        
        old_val = A[i]
        A[i] = new_val
        
        new_score = _score(A)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A[:]
            if best_score <= 4:
                return best_A
        else:
            A[i] = old_val
        
        # Occasionally try a different perturbation
        if iteration % 1000 == 0:
            # Try moving one element to a random position
            i = rng.randint(0, n - 1)
            old_val = A[i]
            A[i] = rng.randint(0, 128)
            while A[i] in A and A.index(A[i]) != i:
                A[i] = rng.randint(0, 128)
            new_score = _score(A)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
                    if best_score <= 4:
                        return best_A
            else:
                A[i] = old_val
    
    return best_A

def _random_set(rng):
    """Generate a random 12-element set in [0, 128]."""
    while True:
        s = set()
        for _ in range(12):
            s.add(rng.randint(0, 128))
        if len(s) == 12:
            return list(s)

def _structured_set(rng):
    """Generate a set with intentional additive structure to create collisions."""
    # Strategy: cluster elements so that many pairs sum to the same value
    # Use a base point and small offsets
    base = rng.randint(20, 100)
    # Create pairs that collide: (base+d1, base+d2) and (base+d3, base+d4) with d1+d2=d3+d4
    offsets = []
    # Add some structured pairs
    # Pair 1: offsets 1, 10 -> sum 11
    # Pair 2: offsets 2, 9 -> sum 11 (collision)
    # Pair 3: offsets 3, 8 -> sum 11 (collision)
    # Pair 4: offsets 4, 7 -> sum 11 (collision)
    # Pair 5: offsets 5, 6 -> sum 11 (collision)
    # That's 10 elements with many collisions
    # Add 2 more random-ish elements
    for d in [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]:
        v = base + d + rng.randint(-5, 5)
        v = max(0, min(128, v))
        offsets.append(v)
    # Add 2 more
    for _ in range(2):
        v = rng.randint(0, 128)
        offsets.append(v)
    
    # Deduplicate
    s = list(set(offsets))
    while len(s) < 12:
        s.append(rng.randint(0, 128))
    s = list(set(s))[:12]
    # Ensure exactly 12
    while len(s) < 12:
        s.append(rng.randint(0, 128))
        s = list(set(s))
    return s[:12]

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    best_A = None
    best_score = 12  # Worst possible
    
    rng = random.Random(42)
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Random sets with local search
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        A = _random_set(rng)
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
        # Local search
        A = _try_improve(A, deadline - 5, rng)
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
    
    # Strategy 2: Structured sets
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        A = _structured_set(rng)
        if len(set(A)) != 12:
            continue
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
        # Local search
        A = _try_improve(A, deadline - 5, rng)
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
    
    # Strategy 3: Greedy construction with collision awareness
    # Build set element by element, preferring values that create sum collisions
    if time.time() < deadline - 5:
        for trial in range(3):
            A = []
            used_sums = {}  # sum -> count of pairs producing it
            
            for step in range(12):
                best_candidate = None
                best_criterion = -1
                candidates = list(range(129))
                rng.shuffle(candidates)
                
                for cand in candidates[:200]:  # Only check first 200 random candidates
                    if cand in A:
                        continue
                    # Count how many new sum collisions this would create
                    # Actually we want to CREATE collisions, not avoid them
                    # But we need to balance: too many collisions might still allow large Sidon subsets
                    # Let's just pick randomly from a biased distribution
                    # Prefer values close to existing elements (small gaps create more collisions)
                    if A:
                        # Compute distance to nearest element
                        dists = [abs(cand - a) for a in A]
                        min_dist = min(dists)
                        # Prefer smaller distances
                        criterion = -min_dist + rng.random() * 5
                    else:
                        criterion = rng.random() * 100
                    
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_candidate = cand
                
                if best_candidate is not None:
                    A.append(best_candidate)
            
            if len(set(A)) == 12:
                s = _score(A)
                if s < best_score:
                    best_score = s
                    best_A = A[:]
                A = _try_improve(A, deadline - 5, rng)
                s = _score(A)
                if s < best_score:
                    best_score = s
                    best_A = A[:]
    
    # Strategy 4: Specific hand-crafted patterns
    # Pattern: arithmetic progression (many collisions)
    for start in [0, 1, 2, 5, 10]:
        if time.time() > deadline - 5:
            break
        A = [start + 10 * i for i in range(12)]
        if max(A) <= 128:
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
            A = _try_improve(A, deadline - 5, rng)
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
    
    # Pattern: two clusters
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        c1 = rng.randint(5, 40)
        c2 = rng.randint(80, 120)
        A = []
        for _ in range(6):
            A.append(c1 + rng.randint(-8, 8))
        for _ in range(6):
            A.append(c2 + rng.randint(-8, 8))
        # Deduplicate
        A = list(set(A))
        while len(A) < 12:
            A.append(rng.randint(0, 128))
            A = list(set(A))
        A = A[:12]
        if len(set(A)) == 12:
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
            A = _try_improve(A, deadline - 5, rng)
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
    
    if best_A is None or len(set(best_A)) != 12 or any(x < 0 or x > 128 for x in best_A):
        return list(range(12))
    
    return best_A
# EVOLVE-BLOCK-END
