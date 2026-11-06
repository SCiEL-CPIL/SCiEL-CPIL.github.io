import time
import random
from fractions import Fraction
import numpy as np

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes = _sieve(1000)
    all_primes = primes
    N = 100000
    n_primes = len(all_primes)
    
    best_L = 0
    best_classes = []
    
    # Pre-compute costs as floats for fast comparison, and exact for final check
    cost_float = {p: 1.0 / p for p in all_primes}
    cost_frac = {p: Fraction(1, p) for p in all_primes}
    
    def compute_L(covered):
        # Find the first gap using numpy
        if not covered[1]:
            return 0
        arr = np.frombuffer(covered[1:], dtype=np.uint8)
        zeros = np.nonzero(arr == 0)[0]
        if len(zeros) == 0:
            return N
        return int(zeros[0])
    
    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline, seed=0):
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget_float = 1.0  # Fast float budget
        classes = []
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            # n is uncovered - find best prime
            best_p = None
            best_r = 0
            best_score = -1e18
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost_f = cost_float[p]
                if budget_float < cost_f - 1e-15:
                    continue
                
                r = n % p
                
                # Count new coverage in [n, min(n+window_size, N)]
                if p > window_size:
                    cnt = 1  # Only n itself
                else:
                    max_k = n + window_size
                    if max_k > N:
                        max_k = N
                    cnt = 0
                    for k in range(n, max_k + 1, p):
                        if not covered[k]:
                            cnt += 1
                
                # Calculate Score
                if use_efficiency:
                    score = cnt * p
                else:
                    score = cnt
                    if prefer_small_on_tie:
                        score += (1000 - p)
                    else:
                        score += p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            # Commit to best_p
            used_primes.add(best_p)
            budget_float -= cost_float[best_p]
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        # Verify budget with exact rationals
        total_cost = sum(cost_frac[p] for p, r in classes)
        if total_cost > Fraction(1):
            return
        
        L = compute_L(covered)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L

    def run_greedy_v2(deadline, seed=0):
        """
        Variant: score = cnt * p, tie-break by larger p (save budget)
        """
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget_float = 1.0
        classes = []
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1e18
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost_f = cost_float[p]
                if budget_float < cost_f - 1e-15:
                    continue
                
                r = n % p
                
                # Count new coverage in [n, min(n+p, N)]
                max_k = min(n + p, N)
                cnt = 0
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                score = cnt * p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget_float -= cost_float[best_p]
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        total_cost = sum(cost_frac[p] for p, r in classes)
        if total_cost > Fraction(1):
            return
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return L

    def run_greedy_v3(deadline, seed=0):
        """
        Variant: score = cnt * sqrt(p) - balances density and cost
        """
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget_float = 1.0
        classes = []
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1e18
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost_f = cost_float[p]
                if budget_float < cost_f - 1e-15:
                    continue
                
                r = n % p
                
                # Count new coverage in [n, min(n+5000, N)]
                max_k = min(n + 5000, N)
                cnt = 0
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                score = cnt * (p ** 0.5)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget_float -= cost_float[best_p]
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        total_cost = sum(cost_frac[p] for p, r in classes)
        if total_cost > Fraction(1):
            return
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return L

    def local_improve(classes, deadline):
        """
        Try to improve by swapping classes: remove one class, add another to extend L.
        """
        nonlocal best_L, best_classes
        
        if not classes:
            return
        
        # Build coverage
        covered = bytearray(N + 1)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, N + 1, p):
                covered[k] = 1
        
        L = compute_L(covered)
        used_primes = set(p for p, r in classes)
        
        # Try removing each class and adding a better one
        for idx in range(len(classes)):
            if time.time() > deadline:
                return
            
            p_remove, r_remove = classes[idx]
            
            # Temporarily remove
            covered2 = bytearray(covered)
            for k in range(r_remove if r_remove != 0 else p_remove, N + 1, p_remove):
                covered2[k] = 0
            
            # Find first gap in covered2
            arr = np.frombuffer(covered2[1:], dtype=np.uint8)
            zeros = np.nonzero(arr == 0)[0]
            if len(zeros) == 0:
                L2 = N
            else:
                L2 = int(zeros[0])
            
            # Try to fill the gap by adding a new prime
            if L2 >= L:
                continue
            
            # Find the first uncovered number in covered2
            n_gap = L2 + 1 if L2 < N else N
            
            # Try adding each unused prime to cover n_gap
            best_new_p = None
            best_new_r = 0
            best_new_L = L2
            
            for p in all_primes:
                if p in used_primes and p != p_remove:
                    continue
                # Check budget: total cost without p_remove + 1/p <= 1
                total_cost = sum(cost_frac[pp] for pp, rr in classes if pp != p_remove) + cost_frac[p]
                if total_cost > Fraction(1):
                    continue
                
                r = n_gap % p
                # Apply this class
                covered3 = bytearray(covered2)
                start_k = r
                if start_k == 0:
                    start_k = p
                for k in range(start_k, N + 1, p):
                    covered3[k] = 1
                
                L3 = compute_L(covered3)
                if L3 > best_new_L:
                    best_new_L = L3
                    best_new_p = p
                    best_new_r = r
            
            if best_new_p is not None and best_new_L > L:
                # Apply the improvement
                new_classes = [c for i, c in enumerate(classes) if i != idx]
                new_classes.append([best_new_p, best_new_r])
                
                # Verify
                total_cost = sum(cost_frac[pp] for pp, rr in new_classes)
                if total_cost <= Fraction(1):
                    covered_new = bytearray(N + 1)
                    for pp, rr in new_classes:
                        start_k = rr
                        if start_k == 0:
                            start_k = pp
                        for k in range(start_k, N + 1, pp):
                            covered_new[k] = 1
                    L_new = compute_L(covered_new)
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = [list(c) for c in new_classes]
                        classes = new_classes
                        covered = covered_new
                        used_primes = set(pp for pp, rr in new_classes)
    
    # Run a mix of strategies
    strategies = [
        # (window, prefer_small, use_eff, seed)
        (1000, True, True, 1),
        (1000, False, True, 2),
        (5000, True, True, 3),
        (5000, False, True, 4),
        (10000, True, True, 5),
        (10000, False, True, 6),
        (2000, True, True, 7),
        (2000, False, True, 8),
        (500, True, True, 9),
        (500, False, True, 10),
        # Add some non-efficiency strategies
        (1000, True, False, 11),
        (1000, False, False, 12),
        (5000, True, False, 13),
        (5000, False, False, 14),
    ]
    
    # First pass: fixed strategies
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 10:
            break
        run_greedy(w, ps, ue, deadline, s)
    
    # Second pass: v2
    run_greedy_v2(deadline)
    
    # Third pass: v3
    run_greedy_v3(deadline)
    
    # Fourth pass: Randomized perturbations
    if time.time() < deadline - 20:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy(w, ps, ue, deadline, s)
    
    # Fifth pass: Local improvement on best solution
    if best_classes and time.time() < deadline - 5:
        local_improve(best_classes, deadline)
    
    if not best_classes:
        # Fallback
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
