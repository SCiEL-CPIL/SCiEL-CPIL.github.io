import time
import math
import random
import sys
import numpy as np

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
    
    # Calculate sumset and diffset sizes
    sumset = set()
    diffset = set()
    U_list = list(U_set)
    for i in range(n):
        a = U_list[i]
        for j in range(n):
            b = U_list[j]
            sumset.add(a + b)
            diffset.add(a - b)
            
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
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
    for a in seq:
        for b in seq:
            used_sums.add(a + b)
            
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for a in seq:
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                seq.append(pos)
                for a in seq:
                    used_sums.add(a + pos)
                break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _local_search(U, deadline, max_iters=2000):
    """
    Local search to improve the score of set U.
    """
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + U
    best_score = _score_set_fast(U)
    best_U = U[:]
    
    max_u = max(U)
    upper_bound = max_u + 50
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        current_U = best_U[:]
        n = len(current_U)
        
        # 1. Try modifying one element
        if n > 1:
            idx = random.randint(1, n - 1)
            old_val = current_U[idx]
            
            candidates = set()
            for d in range(-5, 6):
                nv = old_val + d
                if nv > 0:
                    candidates.add(nv)
            for _ in range(5):
                nv = random.randint(1, upper_bound)
                candidates.add(nv)
            if _is_prime(old_val + 1): candidates.add(old_val + 1)
            if _is_prime(old_val - 1) and old_val > 1: candidates.add(old_val - 1)
            
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
        if not improved and len(best_U) < 30:
            add_candidates = set()
            for v in range(1, upper_bound + 1):
                if v not in best_U:
                    add_candidates.add(v)
            
            primes = [v for v in add_candidates if _is_prime(v)]
            random.shuffle(primes)
            random.shuffle(list(add_candidates - set(primes)))
            add_candidates = set(primes[:20]) | (add_candidates - set(primes))
            
            for new_val in add_candidates:
                new_U = best_U + [new_val]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = sorted(new_U)
                    improved = True
                    break
        
        # 3. Try removing an element
        if not improved and len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # 4. Random restart / Mutation if stuck
        if not improved and random.random() < 0.1:
            mutated = best_U[:]
            if len(mutated) < 30:
                nv = random.randint(1, upper_bound + 20)
                if nv not in mutated:
                    mutated.append(nv)
            elif len(mutated) > 3:
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
        
        if not improved and it > 20:
            upper_bound += 10
            max_u = max(best_U)
            upper_bound = max(upper_bound, max_u + 50)

    return best_U, best_score

def _de_continuous_search(deadline):
    """
    Use scipy's differential evolution to optimize a continuous relaxation
    of the set indicator vector. We define a grid of size M, and variables x_i in [0,1].
    We round x to binary. The objective is the score.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return None, 0.0

    # Grid size
    M = 60
    # We want to maximize score. DE minimizes.
    # Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    # To make it continuous and smooth-ish, we can use a soft count.
    # But the score is highly discontinuous.
    # Strategy: Maximize a proxy that correlates with score.
    # |U-U|/|U+U| is high when U is sparse (Golomb-like).
    # Let's try to optimize a penalty for collisions in differences and sums.
    # Actually, let's just optimize the score directly on the rounded set,
    # but DE needs a continuous function.
    # We can use the score of the set obtained by rounding x_i > 0.5.
    # This is non-differentiable, but DE is global search and handles non-differentiable functions.
    
    def objective(x):
        # x is a vector of length M
        # Ensure 0 is always included. x[0] corresponds to value 0.
        # We force x[0] = 1 by not including it in the optimization or fixing it.
        # Let's optimize x[1:] for values 1..M-1.
        # Values are 0, 1, ..., M-1.
        # U = { i in 0..M-1 | x[i] > 0.5 }
        # Ensure 0 is in U.
        U = [i for i in range(M) if x[i] > 0.5]
        if 0 not in U:
            U.append(0)
        U = sorted(set(U))
        if len(U) < 2:
            return 0.0 # Minimize 0 -> Maximize 0
        score = _score_set_fast(U)
        # We want to maximize score, so return -score
        return -score

    # Bounds: (0,1) for each variable
    bounds = [(0, 1) for _ in range(M)]
    
    # DE parameters
    # maxiter: limit by time
    # popsize: default 15
    
    start_time = time.time()
    # We don't know how long DE will take.
    # Let's set a strict timeout by checking time in a callback? No, DE is C-level.
    # We can just run it with a small maxiter and see.
    # Or use a wrapper that stops?
    # DE doesn't support early stopping easily in Python without interrupting.
    # Let's limit the search space or iterations.
    # M=60, 60 variables. DE is slow.
    # Maybe reduce M to 30?
    M = 30
    bounds = [(0, 1) for _ in range(M)]
    
    # Run DE with a small budget
    # maxiter=10, popsize=10
    # This might take a few seconds.
    
    try:
        res = differential_evolution(objective, bounds, maxiter=10, popsize=10, tol=0.01, polish=False, seed=42)
        if res.success or res.fun < 0: # If it found something better than 0
            x_opt = res.x
            U = [i for i in range(M) if x_opt[i] > 0.5]
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            score = _score_set_fast(U)
            return U, score
    except Exception:
        pass
        
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    candidates = []
    
    # 1. Golomb Rulers
    for n in range(4, 20):
        if time.time() > deadline - 30:
            break
        gr = _construct_golomb(n)
        candidates.append(gr)
        
    # 2. Sidon Sequences
    for n in range(4, 20):
        if time.time() > deadline - 30:
            break
        ss = _construct_sidon(n, max_val=n*n*15)
        if len(ss) >= 4:
            candidates.append(ss)
            
    # 3. Powers of 2
    for n in range(4, 15):
        p2 = [0]
        curr = 1
        for _ in range(n-1):
            curr *= 2
            p2.append(curr)
        candidates.append(p2)
        
    # 4. Prime-based cumulative sums
    primes = _generate_primes(1000)
    for n in range(4, 15):
        if time.time() > deadline - 30:
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
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        n = random.randint(4, 20)
        max_val = random.randint(50, 300)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))
        
    # 6. Mian-Chowla style (greedy sum-free)
    # Sequence where no element is the sum of two previous ones
    def _mian_chowla(n):
        seq = [0, 1]
        for i in range(2, n):
            used = set()
            for j in range(len(seq)):
                for k in range(j, len(seq)):
                    used.add(seq[j] + seq[k])
            val = 2
            while val in used:
                val += 1
            seq.append(val)
        return seq
    
    for n in range(4, 15):
        if time.time() > deadline - 30:
            break
        mc = _mian_chowla(n)
        candidates.append(mc)

    # Score all candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 20:
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
    top_candidates = scored_candidates[:5]
    seen = set()
    unique_top = []
    for s, U in scored_candidates:
        key = tuple(U)
        if key not in seen:
            seen.add(key)
            unique_top.append((s, U))
        if len(unique_top) >= 5:
            break
            
    for s, U in unique_top:
        if time.time() > deadline - 5:
            break
        improved_U, improved_score = _local_search(U, deadline=deadline - 2)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
            
    # Try DE if time permits
    if time.time() < deadline - 15:
        de_U, de_score = _de_continuous_search(deadline - 15)
        if de_U and de_score > best_score:
            best_score = de_score
            best_U = de_U
            # Polish the DE result
            if time.time() < deadline - 5:
                polished_U, polished_score = _local_search(best_U, deadline=deadline - 2)
                if polished_score > best_score:
                    best_score = polished_score
                    best_U = polished_U

    # Final polish on the best found
    if time.time() < deadline - 1:
        polished_U, polished_score = _local_search(best_U, deadline=deadline - 0.5, max_iters=500)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
            
    return sorted(set(best_U))
