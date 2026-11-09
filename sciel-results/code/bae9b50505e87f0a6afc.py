"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U using NumPy for speed."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if len(U) == 0:
        return 0.0
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U using broadcasting
    # This creates n x n arrays
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _local_search(U, max_iterations=50000, timeout=10.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates for addition
        candidates = set()
        
        # 1. Small perturbations
        for x in list(U_set):
            for offset in [-2, -1, 1, 2, 3, 5]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # 2. Structure-preserving: sums and diffs of pairs
        # Limit to first 20 elements to keep it fast
        elems = list(U_set)
        if n > 20:
            elems = elems[:20]
        for idx_a in range(len(elems)):
            for idx_b in range(len(elems)):
                s = elems[idx_a] + elems[idx_b]
                if s <= 2 * max_u + 100:
                    candidates.add(s)
                d = abs(elems[idx_a] - elems[idx_b])
                if d > 0 and d <= 2 * max_u + 100:
                    candidates.add(d)
        
        # 3. Powers
        p = 1
        while p <= 2 * max_u + 100:
            candidates.add(p)
            p *= 2
        p = 1
        while p <= 2 * max_u + 100:
            candidates.add(p)
            p *= 3
            
        # 4. Random candidates in range
        for _ in range(20):
            candidates.add(random.randint(0, 2 * max_u + 100))
            
        # 5. Specific "gap" candidates
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u * 2)
            candidates.add(max_u * 2 + 1)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:30]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 50:
            # Try a swap: remove one, add another
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    # Try adding a random element in a wider range
                    new_elem = random.randint(0, 2 * max_u + 200)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with known good small sets
    best_result = [0, 1, 2, 4]
    best_score = _compute_score(best_result)
    
    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Powers of 2
    p2 = [0]
    v = 1
    while v <= 5000:
        p2.append(v)
        v *= 2
    initial_sets.append(p2)
    
    # 2. Powers of 3
    p3 = [0]
    v = 1
    while v <= 5000:
        p3.append(v)
        v *= 3
    initial_sets.append(p3)
    
    # 3. Mixed powers
    mixed = [0, 1]
    v = 3
    while v <= 5000:
        mixed.append(v)
        v *= 2
    initial_sets.append(mixed)
    
    # 4. Quadratic residues modulo a prime (often good difference sets)
    # Try a few primes
    for p in [101, 103, 107, 109]:
        qr = [0]
        for i in range(1, p):
            if (i * i) % p == 1 or (i * i) % p == 0: # Simplified, just take i^2 mod p
                val = (i * i) % p
                if val not in qr:
                    qr.append(val)
        # Scale to avoid small numbers? Or keep as is?
        # The set size is ~p/2. Max element ~p.
        initial_sets.append(sorted(set(qr)))
        
    # 5. Random sparse sets
    for _ in range(5):
        n = random.randint(10, 50)
        max_val = random.randint(100, 1000)
        rand_set = {0}
        while len(rand_set) < n:
            rand_set.add(random.randint(0, max_val))
        initial_sets.append(sorted(rand_set))
        
    # 6. Dense low, sparse high
    dense_sparse = list(range(10))
    v = 20
    while v <= 1000:
        dense_sparse.append(v)
        v *= 2
    initial_sets.append(sorted(set(dense_sparse)))
    
    # 7. Fibonacci
    fib = [0, 1]
    while fib[-1] <= 5000:
        fib.append(fib[-1] + fib[-2])
    initial_sets.append(fib)
    
    # Sort initial sets by score to try best ones first
    initial_sets.sort(key=lambda s: _compute_score(s), reverse=True)
    
    # Run local search on each initial set
    for init_set in initial_sets:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 2
        if remaining < 2:
            break
            
        # Allocate time based on potential
        timeout = min(15.0, remaining)
        improved, score = _local_search(init_set, max_iterations=20000, timeout=timeout)
        
        if score > best_score:
            best_score = score
            best_result = improved
    
    # Final polish on the best result
    if time.time() < deadline - 5:
        polish_timeout = min(10.0, deadline - time.time() - 1)
        if polish_timeout > 1:
            polished, polished_score = _local_search(best_result, max_iterations=10000, timeout=polish_timeout)
            if polished_score > best_score:
                best_score = polished_score
                best_result = polished
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
