import time
import random
import math

def _score(U):
    """Compute the score for a given set U."""
    if not U or U[0] != 0:
        return 0.0
    # Ensure unique and sorted
    U = sorted(list(set(U)))
    n = len(U)
    if n < 2:
        return 0.0
    M = U[-1]
    if M == 0:
        return 0.0
    
    sums = set()
    diffs = set()
    # Optimization: use list comprehension or loops
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    num = math.log(len(diffs)) - math.log(len(sums))
    den = math.log(2 * M + 1)
    if den == 0:
        return 0.0
    return 1 + num / den

def _modular_lattice(m, k, offset=0):
    """
    Construct a set using modular lattice structure.
    Select elements from {0, 1, ..., m-1} that form a dense sublattice,
    then optionally scale and offset.
    """
    # Generate residues: pick k residues from Z_m
    # Strategy: use arithmetic progression mod m for structured collisions
    residues = set()
    for i in range(k):
        residues.add((i * m // k) % m)
    
    # Convert to actual values (smallest non-negative representatives)
    U = sorted(residues)
    
    if offset > 0 and len(U) > 0:
        U = [x + offset for x in U]
        U[0] = 0  # Ensure 0 is in the set
    
    return sorted(set(U))

def _mixed_radix_construction(radixes):
    """
    Construct U using mixed-radix representation.
    Each element is sum of digits * radix^position.
    This creates structured sum collisions.
    """
    U = set([0])
    n = len(radixes)
    for pos in range(n):
        r = radixes[pos]
        # Add multiples of r^pos for each digit
        base = 1
        for j in range(pos):
            base *= radixes[j]
        for d in range(r):
            U.add(d * base)
    
    return sorted(U)

def _local_search(U, max_iters=500, deadline=None):
    """
    Perform local search with adaptive step sizes.
    Returns improved set and score.
    """
    current_U = sorted(set(U))
    current_score = _score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    if deadline is None:
        deadline = time.time() + 10.0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random index to perturb
        idx = random.randint(1, len(current_U) - 1)
        
        # Adaptive step size based on local neighborhood
        gap_below = current_U[idx] - current_U[idx-1] if idx > 0 else 1
        gap_above = current_U[idx+1] - current_U[idx] if idx < len(current_U) - 1 else 1
        max_step = max(1, min(gap_below, gap_above) // 2)
        
        if max_step < 1:
            continue
        
        # Random step within valid range
        delta = random.randint(-max_step, max_step)
        if delta == 0:
            continue
        
        new_val = current_U[idx] + delta
        
        # Validity checks
        if new_val < 0:
            continue
        if idx > 0 and new_val <= current_U[idx-1]:
            continue
        if idx < len(current_U) - 1 and new_val >= current_U[idx+1]:
            continue
        
        new_U = current_U[:]
        new_U[idx] = new_val
        new_score = _score(new_U)
        
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
    
    return best_U, best_score

def _hill_climbing(U, deadline=None):
    """
    More aggressive hill climbing with multiple restarts.
    """
    best_U = sorted(set(U))
    best_score = _score(best_U)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    for restart in range(3):
        if time.time() > deadline:
            break
        
        current_U = best_U[:]
        # Add some randomness to escape local optima
        for _ in range(random.randint(1, 3)):
            idx = random.randint(1, len(current_U) - 1)
            gap = min(
                current_U[idx] - current_U[idx-1] if idx > 0 else 1,
                current_U[idx+1] - current_U[idx] if idx < len(current_U) - 1 else 1
            )
            if gap > 2:
                delta = random.randint(-gap//2, gap//2)
                if delta != 0:
                    new_val = current_U[idx] + delta
                    if new_val > 0 and (idx == 0 or new_val > current_U[idx-1]) and \
                       (idx == len(current_U)-1 or new_val < current_U[idx+1]):
                        current_U[idx] = new_val
        
        current_U, current_score = _local_search(current_U, max_iters=200, deadline=deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_U = current_U[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Try various construction families
    constructions = []
    
    # Geometric sequences
    for base in [2, 3, 4, 5]:
        for n in range(4, 12):
            U = [0]
            val = 1
            for i in range(n-1):
                U.append(val)
                val *= base
            constructions.append(U)
    
    # Prime-gap sequences
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    for n in range(4, 15):
        U = [0]
        val = 0
        for i in range(n-1):
            val += primes[i % len(primes)]
            U.append(val)
        constructions.append(U)
    
    # Mixed-radix constructions
    radix_combos = [
        [2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 3, 2], [2, 3, 2, 2],
        [3, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 3],
        [2, 3, 3, 2], [3, 2, 3, 2], [2, 2, 3, 3],
    ]
    for radices in radix_combos:
        U = _mixed_radix_construction(radices)
        if len(U) >= 4:
            constructions.append(U)
    
    # Modular lattice constructions
    for m in [10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]:
        for k in range(3, min(12, m // 2 + 1)):
            U = _modular_lattice(m, k)
            if len(U) >= 3:
                constructions.append(U)
    
    # Random gap sequences
    random.seed(42)
    for trial in range(20):
        n = random.randint(5, 15)
        U = [0]
        val = 0
        for i in range(n-1):
            val += random.randint(1, 10)
            U.append(val)
        constructions.append(U)
    
    # Evaluate all constructions and keep top ones
    scored_constructions = []
    for U in constructions:
        U = sorted(set(U))
        if len(U) < 3:
            continue
        score = _score(U)
        scored_constructions.append((score, U))
    
    scored_constructions.sort(key=lambda x: -x[0])
    
    # Take top 10 for local search
    for score, U in scored_constructions[:10]:
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Phase 2: Local search on promising candidates
    if time.time() < deadline - 20:
        for score, U in scored_constructions[:5]:
            if time.time() > deadline - 15:
                break
            
            # Try hill climbing
            improved_U, improved_score = _hill_climbing(U, deadline=deadline - 10)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_U[:]
    
    # Phase 3: More aggressive search with varied strategies
    if time.time() < deadline - 10:
        # Try different set sizes with local search
        for n in [5, 6, 7, 8, 9, 10, 12, 15]:
            if time.time() > deadline - 8:
                break
            
            # Initialize with random gaps
            U = [0]
            val = 0
            for i in range(n-1):
                val += random.randint(1, 8)
                U.append(val)
            
            improved_U, improved_score = _hill_climbing(U, deadline=deadline - 5)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_U[:]
    
    # Final polish: very fine-grained local search on the best set
    if time.time() < deadline - 3 and best_set:
        final_U, final_score = _local_search(best_set, max_iters=1000, deadline=deadline - 1)
        if final_score > best_score:
            best_score = final_score
            best_set = final_U[:]
    
    return best_set
