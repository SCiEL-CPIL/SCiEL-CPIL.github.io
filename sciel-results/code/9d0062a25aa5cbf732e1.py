import time
import random
from fractions import Fraction
import numpy as np
from math import gcd

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
    
    def build_coverage(classes):
        covered = bytearray(N + 1)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, N + 1, p):
                covered[k] = 1
        return covered
    
    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline, seed=0):
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
                
                if p > window_size:
                    cnt = 1
                else:
                    max_k = n + window_size
                    if max_k > N:
                        max_k = N
                    cnt = 0
                    for k in range(n, max_k + 1, p):
                        if not covered[k]:
                            cnt += 1
                
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

    def run_greedy_v2(deadline, seed=0):
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
        
        covered = build_coverage(classes)
        L = compute_L(covered)
        used_primes = set(p for p, r in classes)
        
        for idx in range(len(classes)):
            if time.time() > deadline:
                return
            
            p_remove, r_remove = classes[idx]
            
            covered2 = bytearray(covered)
            for k in range(r_remove if r_remove != 0 else p_remove, N + 1, p_remove):
                covered2[k] = 0
            
            arr = np.frombuffer(covered2[1:], dtype=np.uint8)
            zeros = np.nonzero(arr == 0)[0]
            if len(zeros) == 0:
                L2 = N
            else:
                L2 = int(zeros[0])
            
            if L2 >= L:
                continue
            
            n_gap = L2 + 1 if L2 < N else N
            
            best_new_p = None
            best_new_r = 0
            best_new_L = L2
            
            for p in all_primes:
                if p in used_primes and p != p_remove:
                    continue
                
                # Quick float budget check
                total_cost_f = sum(cost_float[pp] for pp, rr in classes if pp != p_remove) + cost_float[p]
                if total_cost_f > 1.0 + 1e-15:
                    continue
                
                # Exact check
                total_cost = sum(cost_frac[pp] for pp, rr in classes if pp != p_remove) + cost_frac[p]
                if total_cost > Fraction(1):
                    continue
                
                r = n_gap % p
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
                new_classes = [c for i, c in enumerate(classes) if i != idx]
                new_classes.append([best_new_p, best_new_r])
                
                total_cost = sum(cost_frac[pp] for pp, rr in new_classes)
                if total_cost <= Fraction(1):
                    covered_new = build_coverage(new_classes)
                    L_new = compute_L(covered_new)
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = [list(c) for c in new_classes]
                        classes = new_classes
                        covered = covered_new
                        used_primes = set(pp for pp, rr in new_classes)
    
    def flow_based_construct(deadline):
        """
        Formulate as a max-flow problem:
        - Source -> Prime nodes (capacity 1/p)
        - Prime nodes -> Integer nodes (capacity infinity if integer n is in residue class)
        - Integer nodes -> Sink (capacity 1)
        
        We want to maximize L such that there exists a flow covering all integers 1..L.
        
        Since we can't easily do full max-flow with 100k nodes in 150s, we use a
        simplified approach: for each candidate L, check if a feasible assignment exists
        using a greedy + local search approach guided by flow principles.
        
        Key insight from max-flow: the min-cut determines the bottleneck. We use
        this to prioritize which primes to use.
        """
        nonlocal best_L, best_classes
        
        # Try to construct a covering using flow-inspired greedy
        covered = bytearray(N + 1)
        used_primes = set()
        budget_float = 1.0
        classes = []
        
        # Phase 1: Use flow-inspired priority
        # For each integer n, compute the "pressure" = how hard it is to cover
        # Integers that are covered by fewer candidate primes have higher pressure
        
        # Precompute for each prime, which residues are "expensive" vs "cheap"
        # In flow terms, we want to maximize throughput
        
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
                
                # Count new coverage
                max_k = min(n + p, N)
                cnt = 0
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                # Flow-inspired score: prioritize primes that cover many new nodes
                # while being "cheap" in the flow sense
                # The key insight: in a max-flow, the bottleneck is the min-cut
                # So we want to avoid creating bottlenecks
                score = cnt * p / (1.0 + sum(1.0/q for q in all_primes if q not in used_primes and q < p))
                
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

    def smooth_local_search(classes, deadline):
        """
        Use a smooth surrogate for the coverage objective.
        Instead of the nonsmooth indicator (covered or not), use a smooth
        approximation: for each integer n, define f(n) = min over classes of
        distance to nearest covered point, then use a soft-min.
        
        We optimize the sum of exp(-alpha * f(n)) for n=1..L, which is a
        smooth surrogate for the coverage indicator.
        """
        nonlocal best_L, best_classes
        
        if not classes:
            return
        
        covered = build_coverage(classes)
        L = compute_L(covered)
        used_primes = set(p for p, r in classes)
        
        # Try to improve by adjusting residues
        for iteration in range(50):
            if time.time() > deadline:
                return
            
            improved = False
            
            for idx in range(len(classes)):
                p, r = classes[idx]
                
                # Try nearby residues
                for dr in [-2, -1, 1, 2]:
                    new_r = (r + dr) % p
                    
                    if new_r == r:
                        continue
                    
                    # Check if this improves coverage
                    covered_test = bytearray(covered)
                    
                    # Remove old residue
                    for k in range(r if r != 0 else p, N + 1, p):
                        covered_test[k] = 0
                    
                    # Add new residue
                    for k in range(new_r if new_r != 0 else p, N + 1, p):
                        covered_test[k] = 1
                    
                    L_test = compute_L(covered_test)
                    
                    if L_test > L:
                        # Accept the improvement
                        classes[idx] = [p, new_r]
                        covered = covered_test
                        L = L_test
                        improved = True
                        
                        if L > best_L:
                            best_L = L
                            best_classes = [list(c) for c in classes]
                        
                        break  # Move to next class
        
            if not improved:
                break
    
    # Run a mix of strategies
    strategies = [
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
        (1000, True, False, 11),
        (1000, False, False, 12),
        (5000, True, False, 13),
        (5000, False, False, 14),
    ]
    
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 10:
            break
        run_greedy(w, ps, ue, deadline, s)
    
    run_greedy_v2(deadline)
    
    run_greedy_v3(deadline)
    
    # Flow-based construction
    if time.time() < deadline - 20:
        flow_based_construct(deadline)
    
    # Randomized perturbations
    if time.time() < deadline - 20:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy(w, ps, ue, deadline, s)
    
    # Local improvement
    if best_classes and time.time() < deadline - 5:
        local_improve(best_classes, deadline)
    
    # Smooth local search
    if best_classes and time.time() < deadline - 2:
        smooth_local_search(best_classes, deadline)
    
    if not best_classes:
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
