"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _sumset_size(A):
    """Compute |A+A| for a set/list of positive integers."""
    s = set(A)
    sums = set()
    for a in s:
        for b in s:
            sums.add(a + b)
    return len(sums)

def _prodset_size(A):
    """Compute |A*A| for a set/list of positive integers."""
    s = set(A)
    prods = set()
    for a in s:
        for b in s:
            prods.add(a * b)
    return len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A):
    """Check that A is 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _initial_geometric():
    """Generate a geometric-progression-like set."""
    # Start with powers of 2, which gives small product set
    # 2^0..2^19: products are 2^(i+j) for i,j in 0..19 -> 20+19=39 distinct
    # Sums: 2^i + 2^j, many distinct. Let's compute.
    A = [2**i for i in range(20)]
    # 2^19 = 524288 > 10000, so this is invalid
    # Need values <= 10000. Max exponent for 2: 2^13=8192, 2^14=16384
    # So at most 14 powers of 2.
    
    # Try a mixed approach: use small numbers with structure
    # Geometric with ratio close to 1 won't work well for products
    # Let's try: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # That's 14 numbers. Need 6 more.
    
    # Alternative: use 5-smooth numbers (Hamming numbers)
    # These have small product sets because products stay in the same family
    # 5-smooth numbers <= 10000:
    smooth = []
    for a in range(0, 15):  # 2^a
        for b in range(0, 10):  # 3^b
            for c in range(0, 8):  # 5^c
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= 10000:
                    smooth.append(v)
    smooth = sorted(set(smooth))
    
    # Pick 20 from smooth numbers, preferring those that keep product set small
    # The product of two 5-smooth numbers is 5-smooth, so A*A subset of 5-smooth
    # |A*A| <= number of 5-smooth products of pairs from A
    
    # Let's just take the first 20 5-smooth numbers
    A = smooth[:20]
    return A

def _local_search(A, max_iter=5000, deadline=None):
    """Local search: try swapping elements to reduce objective."""
    A = list(A)
    best_obj = _objective(A)
    
    # Build a pool of candidate values
    # Include 5-smooth numbers and some random small numbers
    pool = set(A)
    for a in range(0, 14):
        for b in range(0, 9):
            for c in range(0, 7):
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= 10000:
                    pool.add(v)
    # Add some random values
    for _ in range(200):
        pool.add(random.randint(1, 10000))
    pool = sorted(pool)
    
    improved = True
    while improved and max_iter > 0:
        if deadline and time.time() > deadline:
            break
        improved = False
        max_iter -= 1
        
        # Try swapping each element with a random pool element
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try several candidates
        for _ in range(20):
            new_val = random.choice(pool)
            if new_val == old_val:
                continue
            A[idx] = new_val
            if _is_valid(A):
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    improved = True
                    break
                else:
                    A[idx] = old_val
            else:
                A[idx] = old_val
        
        # Also try pairwise swaps
        if not improved and max_iter > 0:
            max_iter -= 1
            i, j = random.sample(range(20), 2)
            A[i], A[j] = A[j], A[i]
            if _is_valid(A):
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    improved = True
                else:
                    A[i], A[j] = A[j], A[i]
    
    return A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    random.seed(42)
    
    # Start with 5-smooth numbers
    best_A = _initial_geometric()
    best_obj = _objective(best_A)
    
    # Local search from initial
    A, obj = _local_search(best_A, max_iter=3000, deadline=deadline)
    if obj < best_obj:
        best_obj = obj
        best_A = A
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random starting point
        if restart % 3 == 0:
            # 5-smooth based
            smooth = []
            for a in range(0, 14):
                for b in range(0, 9):
                    for c in range(0, 7):
                        v = (2**a) * (3**b) * (5**c)
                        if 1 <= v <= 10000:
                            smooth.append(v)
            smooth = sorted(set(smooth))
            if len(smooth) >= 20:
                # Random selection of 20 from smooth
                A = random.sample(smooth, 20)
            else:
                A = smooth + [random.randint(1, 10000) for _ in range(20 - len(smooth))]
        elif restart % 3 == 1:
            # Geometric-ish
            base = random.uniform(1.05, 1.5)
            A = []
            for i in range(20):
                v = int(round(base**i))
                if v < 1:
                    v = 1
                if v > 10000:
                    v = 10000
                A.append(v)
            # Ensure distinct
            A = sorted(set(A))
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            A = A[:20]
        else:
            # Random
            A = random.sample(range(1, 10001), 20)
        
        if not _is_valid(A):
            continue
        
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
        
        # Local search from this start
        A2, obj2 = _local_search(A, max_iter=1000, deadline=deadline)
        if obj2 < best_obj:
            best_obj = obj2
            best_A = A2
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
