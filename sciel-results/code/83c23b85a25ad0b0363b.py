"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _score(A):
    """Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers."""
    A = sorted(set(A))
    if len(A) != 20:
        return float('inf')
    if A[0] < 1 or A[-1] > 10000:
        return float('inf')
    sums = set()
    prods = set()
    for i in range(20):
        for j in range(i, 20):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _generate_geometric(base, exponents):
    """Generate A = {base^e_i} and check validity."""
    A = [base ** e for e in exponents]
    if len(set(A)) != 20:
        return None
    if A[0] < 1 or A[-1] > 10000:
        return None
    return sorted(A)

def _generate_perturbed(base, exponents, k_values):
    """Generate A = {base^e_i * k_i} where k_i are small coprime integers."""
    A = []
    for i in range(20):
        val = (base ** exponents[i]) * k_values[i]
        if val > 10000 or val < 1:
            return None
        A.append(val)
    if len(set(A)) != 20:
        return None
    return sorted(A)

def _local_search(A, max_iter=5000, time_limit=80.0):
    """Hill-climbing local search to minimize the score."""
    A = sorted(set(A))
    if len(A) != 20:
        return None
    best_A = A[:]
    best_score = _score(A)
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try several perturbations
        candidates = []
        for delta in range(-50, 51):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            # Check distinctness
            if new_val not in A or new_val == old_val:
                candidates.append(new_val)
        
        if not candidates:
            continue
            
        new_val = random.choice(candidates)
        A[idx] = new_val
        A.sort()
        
        new_score = _score(A)
        if new_score < best_score:
            best_score = new_score
            best_A = A[:]
        else:
            A[idx] = old_val
            A.sort()
    
    return best_A

def _generate_2smooth_candidates():
    """Generate candidates from 2-smooth numbers (powers of 2 times small odd factors)."""
    candidates = []
    # 2-smooth numbers up to 10000
    smooth = []
    for a in range(0, 14):  # 2^a <= 8192
        for b in range(0, 14):  # 3^b <= 4782969, but we'll cap at 10000
            val = (2 ** a) * (3 ** b)
            if val <= 10000 and val >= 1:
                smooth.append(val)
    smooth = sorted(set(smooth))
    
    # Try to find 20-element subsets with good structure
    random.seed(42)
    for _ in range(100):
        if len(smooth) < 20:
            break
        subset = random.sample(smooth, 20)
        candidates.append(sorted(subset))
    
    return candidates

def _generate_3smooth_candidates():
    """Generate candidates from 3-smooth numbers."""
    candidates = []
    smooth = []
    for a in range(0, 14):
        for b in range(0, 10):
            for c in range(0, 8):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000 and val >= 1:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    
    random.seed(123)
    for _ in range(100):
        if len(smooth) < 20:
            break
        subset = random.sample(smooth, 20)
        candidates.append(sorted(subset))
    
    return candidates

def _generate_geometric_candidates():
    """Generate candidates from geometric progressions with perturbation."""
    candidates = []
    
    # Try different bases and exponent ranges
    for base in [2, 3, 4, 5]:
        # Find valid exponent sets
        max_exp = 1
        while (base ** (max_exp + 1)) <= 10000:
            max_exp += 1
        
        if max_exp < 19:
            continue
        
        # Generate geometric sequences
        for start_exp in range(0, max_exp - 18):
            exponents = list(range(start_exp, start_exp + 20))
            A = _generate_geometric(base, exponents)
            if A is not None:
                candidates.append(A)
        
        # Generate perturbed geometric sequences
        small_coprimes = [1, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
        random.seed(777)
        for _ in range(50):
            exponents = list(range(0, 20))
            k_vals = random.sample(small_coprimes, 20)
            A = _generate_perturbed(base, exponents, k_vals)
            if A is not None:
                candidates.append(A)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    best_A = None
    best_score = float('inf')
    
    # Generate multiple candidate families
    all_candidates = []
    
    # 1. Geometric progression candidates
    geo_candidates = _generate_geometric_candidates()
    all_candidates.extend(geo_candidates)
    
    # 2. 2-smooth candidates
    smooth2_candidates = _generate_2smooth_candidates()
    all_candidates.extend(smooth2_candidates)
    
    # 3. 3-smooth candidates
    smooth3_candidates = _generate_3smooth_candidates()
    all_candidates.extend(smooth3_candidates)
    
    # 4. Simple arithmetic progression as baseline
    all_candidates.append(list(range(1, 21)))
    all_candidates.append(list(range(2, 22)))
    all_candidates.append([2**i for i in range(20)] if 2**19 <= 10000 else list(range(1, 21)))
    
    # Evaluate all candidates
    for A in all_candidates:
        if time.time() > deadline - 10.0:
            break
        score = _score(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
    
    # Local search from the best candidate
    if best_A is not None and time.time() < deadline - 5.0:
        improved_A = _local_search(best_A, max_iter=10000, time_limit=deadline - time.time() - 2.0)
        if improved_A is not None:
            improved_score = _score(improved_A)
            if improved_score < best_score:
                best_score = improved_score
                best_A = improved_A
    
    # Try local search from a few other good candidates
    if time.time() < deadline - 5.0:
        for A in all_candidates[:10]:
            if time.time() > deadline - 3.0:
                break
            if _score(A) <= best_score + 5:  # Only search from near-best candidates
                improved_A = _local_search(A, max_iter=3000, time_limit=min(5.0, deadline - time.time() - 2.0))
                if improved_A is not None:
                    improved_score = _score(improved_A)
                    if improved_score < best_score:
                        best_score = improved_score
                        best_A = improved_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
