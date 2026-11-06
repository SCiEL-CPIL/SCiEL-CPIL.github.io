import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
L_MAX = 5000

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def compute_L(covered):
        L = 0
        for i in range(1, L_MAX + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def get_coverage(classes):
        covered = bytearray(L_MAX + 1)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, L_MAX + 1, p):
                covered[k] = 1
        return covered

    def greedy_build(deadline):
        covered = bytearray(L_MAX + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= L_MAX:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in PRIMES:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                gain = 0
                k = n
                while k <= L_MAX:
                    if not covered[k]:
                        gain += 1
                    k += p
                    
                score = gain * p
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
                
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, L_MAX + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        return classes, budget

    def local_search(classes, budget, deadline):
        used_primes = set(p for p, r in classes)
        covered = get_coverage(classes)
        L = compute_L(covered)
        
        improved = True
        no_improve_count = 0
        max_no_improve = 30
        
        while improved and time.time() < deadline - 5:
            improved = False
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            U = L + 1
            if U > L_MAX:
                break
            
            best_new_L = L
            best_move = None
            best_new_classes = None
            best_new_budget = budget
            
            # 1. Add new prime
            for p in PRIMES:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = U % p
                
                new_covered = bytearray(covered)
                start_k = r
                if start_k == 0:
                    start_k = p
                for k in range(start_k, L_MAX + 1, p):
                    new_covered[k] = 1
                
                new_L = L
                for i in range(U, L_MAX + 1):
                    if new_covered[i]:
                        new_L = i
                    else:
                        break
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_move = ('add', p, r)
                    best_new_classes = classes + [[p, r]]
                    best_new_budget = budget - cost
            
            # 2. Change existing prime's residue
            for idx, (p, r) in enumerate(classes):
                for r_new in range(p):
                    if r_new == r:
                        continue
                    new_classes = classes.copy()
                    new_classes[idx] = [p, r_new]
                    new_covered = get_coverage(new_classes)
                    new_L = compute_L(new_covered)
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_move = ('change', idx, r_new)
                        best_new_classes = new_classes
                        best_new_budget = budget
            
            # 3. Swap: remove one, add one
            for remove_idx in range(len(classes)):
                p_rem = classes[remove_idx][0]
                remaining_budget = budget + Fraction(1, p_rem)
                for p_add in PRIMES:
                    if p_add in used_primes or p_add == p_rem:
                        continue
                    cost = Fraction(1, p_add)
                    if remaining_budget < cost:
                        continue
                    r_add = U % p_add
                    new_classes = classes.copy()
                    new_classes.pop(remove_idx)
                    new_classes.append([p_add, r_add])
                    new_covered = get_coverage(new_classes)
                    new_L = compute_L(new_covered)
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_move = ('swap', remove_idx, p_add, r_add)
                        best_new_classes = new_classes
                        best_new_budget = remaining_budget - cost
            
            if best_move is not None and best_new_L > L:
                classes = best_new_classes
                budget = best_new_budget
                used_primes = set(p for p, r in classes)
                covered = get_coverage(classes)
                L = best_new_L
                improved = True
                no_improve_count = 0
        
        return classes, budget, L

    best_classes = []
    best_L = 0
    best_budget = Fraction(0)

    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        classes, remaining_budget = greedy_build(deadline)
        if not classes:
            continue
        classes, remaining_budget, L = local_search(classes, remaining_budget, deadline)
        if L > best_L:
            best_L = L
            best_classes = classes
            best_budget = remaining_budget

    if not best_classes:
        return [[2, 1]]

    return best_classes
