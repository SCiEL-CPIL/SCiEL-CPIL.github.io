import time
import math
import random
import sys
import numpy as np
from scipy.optimize import differential_evolution

def _score_set_fast(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
    """
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset and diffset sizes using numpy for speed
    U_arr = np.array(list(U_set), dtype=np.int64)
    
    # Sumset: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Diffset: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes(max_val):
    if max_val < 2:
        return []
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    return [i for i in range(2, max_val + 1) if sieve[i]]

def _construct_golomb(n):
    """Construct a Golomb ruler of order n."""
    if n <= 1:
        return [0]
    
    # Known optimal Golomb rulers
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96, 108],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96, 108, 121],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96, 108, 121, 136],
    }
    if n in known:
        return known[n]
    
    # Greedy construction for larger n
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _construct_sidon(n, max_val=None):
    """Construct a Sidon sequence of length n."""
    if max_val is None:
        max_val = n * n * 10
    
    seq = [0]
    used_sums = set()
    # Pre-compute sums for initial seq
    for a in seq:
        for b in seq:
            used_sums.add(a + b)
            
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            # Check if pos creates a new sum collision
            for a in seq:
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Add pos and update used_sums
                seq.append(pos)
                for a in seq:
                    used_sums.add(a + pos)
                break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _scipy_relaxation(start_set, deadline, n_vars=15, max_iter=100):
    """
    Use scipy differential evolution to optimize a continuous relaxation.
    The idea: represent set elements as continuous variables, round to integers,
    and evaluate the score.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() > deadline - 5:
            return 0.0
        # Round to integers and ensure non-negative, distinct, contains 0
        x_int = np.round(x).astype(int)
        x_int = np.clip(x_int, 0, 1000)
        
        # Ensure 0 is included
        if 0 not in x_int:
            x_int = np.insert(x_int, 0, 0)
        
        # Remove duplicates and sort
        x_int = np.unique(x_int)
        
        if len(x_int) < 2:
            return 0.0
        
        # Limit size to avoid too slow scoring
        if len(x_int) > 25:
            x_int = x_int[:25]
        
        try:
            return -_score_set_fast(list(x_int))
        except:
            return 0.0
    
    # Bounds: each variable can be 0 to 1000
    bounds = [(0, 1000) for _ in range(n_vars)]
    
    # Initial population based on start_set
    if len(start_set) > 0:
        initial = [list(start_set[:n_vars]) + [random.randint(1, 500) for _ in range(n_vars - len(start_set[:n_vars]))]]
        # Ensure length matches
        while len(initial[0]) < n_vars:
            initial[0].append(random.randint(1, 500))
        initial[0] = initial[0][:n_vars]
    else:
        initial = None
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=10,
            seed=42,
            init='latinhypercube',
            tol=1e-4,
            polish=False
        )
        
        # Convert result to integer set
        x_int = np.round(result.x).astype(int)
        x_int = np.clip(x_int, 0, 1000)
        
        if 0 not in x_int:
            x_int = np.insert(x_int, 0, 0)
        
        x_int = np.unique(x_int)
        
        if len(x_int) < 2:
            return None, 0.0
        
        if len(x_int) > 25:
            x_int = x_int[:25]
        
        U = list(x_int)
        score = _score_set_fast(U)
        return U, score
        
    except Exception as e:
        return None, 0.0

def _local_search(U, deadline, max_iters=2000):
    """
    Local search to improve the score of set U.
    """
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + U
    best_score = _score_set_fast(U)
    best_U = U[:]
    
    # Determine search space bounds
    max_u = max(U)
    upper_bound = max_u + 100
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        current_U = best_U[:]
        n = len(current_U)
        
        # 1. Try modifying one element
        # Pick a random element to modify (except 0)
        if n > 1:
            idx = random.randint(1, n - 1)
            old_val = current_U[idx]
            
            # Generate candidates for new value
            candidates = set()
            # Neighbors
            for d in range(-10, 11):
                nv = old_val + d
                if nv > 0:
                    candidates.add(nv)
            # Random jumps
            for _ in range(10):
                nv = random.randint(1, upper_bound)
                candidates.add(nv)
            # Prime candidates
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
                if _is_prime(old_val + p): candidates.add(old_val + p)
                if _is_prime(old_val - p) and old_val > p: candidates.add(old_val - p)
            
            for new_val in candidates:
                if new_val in current_U and new_val != old_val:
                    continue
                new_U = current_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(current_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # 2. Try adding an element
        if not improved and len(best_U) < 25:
            # Try primes and small integers
            add_candidates = set()
            for v in range(1, upper_bound + 1):
                if v not in best_U:
                    add_candidates.add(v)
            
            # Prioritize primes
            primes = [v for v in add_candidates if _is_prime(v)]
            random.shuffle(primes)
            random.shuffle(list(add_candidates - set(primes)))
            add_candidates = set(primes[:30]) | (add_candidates - set(primes))
            
            for new_val in add_candidates:
                new_U = best_U + [new_val]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = sorted(new_U)
                    improved = True
                    break
        
        # 3. Try removing an element
        if not improved and len(best_U) > 4:
            for i in range(1, len(best_U)): # Don't remove 0
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # 4. Random restart / Mutation if stuck
        if not improved and random.random() < 0.15:
            mutated = best_U[:]
            # Add a random element
            if len(mutated) < 25:
                nv = random.randint(1, upper_bound + 50)
                if nv not in mutated:
                    mutated.append(nv)
            # Remove a random element (not 0)
            elif len(mutated) > 4:
                idx = random.randint(1, len(mutated) - 1)
                mutated.pop(idx)
            
            mutated = sorted(set(mutated))
            if 0 not in mutated:
                mutated.append(0)
                mutated = sorted(mutated)
            
            s = _score_set_fast(mutated)
            if s > best_score + 1e-12:
                best_score = s
                best_U = mutated
                improved = True
        
        if not improved and it > 30:
            # Expand search space
            upper_bound += 20
            max_u = max(best_U)
            upper_bound = max(upper_bound, max_u + 100)

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    candidates = []
    
    # 1. Golomb Rulers
    for n in range(4, 25):
        if time.time() > deadline - 40:
            break
        gr = _construct_golomb(n)
        candidates.append(gr)
        
    # 2. Sidon Sequences
    for n in range(4, 25):
        if time.time() > deadline - 40:
            break
        ss = _construct_sidon(n, max_val=n*n*15)
        if len(ss) >= 4:
            candidates.append(ss)
            
    # 3. Powers of 2
    for n in range(4, 18):
        p2 = [0]
        curr = 1
        for _ in range(n-1):
            curr *= 2
            p2.append(curr)
        candidates.append(p2)
        
    # 4. Prime-based cumulative sums
    primes = _generate_primes(2000)
    for n in range(4, 18):
        if time.time() > deadline - 40:
            break
        curr = 0
        U = [0]
        for i in range(1, n):
            if i-1 < len(primes):
                curr += primes[i-1]
                U.append(curr)
        if len(U) == n:
            candidates.append(U)
            
    # 5. Random sets
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        n = random.randint(4, 25)
        max_val = random.randint(100, 500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))
        
    # 6. Scipy relaxation for better initial candidates
    if time.time() < deadline - 30:
        scipy_U, scipy_score = _scipy_relaxation(best_U, deadline - 20, n_vars=12, max_iter=50)
        if scipy_U is not None and scipy_score > 0:
            candidates.append(scipy_U)
    
    # Score all candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 25:
            break
        s = _score_set_fast(U)
        scored_candidates.append((s, U))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Update best
    if scored_candidates:
        top_score, top_U = scored_candidates[0]
        if top_score > best_score:
            best_score = top_score
            best_U = top_U
            
    # Local search on top candidates
    # We will run local search on the top 8 distinct candidates
    # and potentially some random restarts
    top_candidates = scored_candidates[:8]
    # Ensure diversity: if top 8 are very similar, add others
    seen = set()
    unique_top = []
    for s, U in scored_candidates:
        key = tuple(U)
        if key not in seen:
            seen.add(key)
            unique_top.append((s, U))
        if len(unique_top) >= 8:
            break
            
    for s, U in unique_top:
        if time.time() > deadline - 10:
            break
        improved_U, improved_score = _local_search(U, deadline=deadline - 3)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
            
    # Final polish on the best found
    if time.time() < deadline - 2:
        polished_U, polished_score = _local_search(best_U, deadline=deadline - 0.5, max_iters=800)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
            
    return sorted(set(best_U))

