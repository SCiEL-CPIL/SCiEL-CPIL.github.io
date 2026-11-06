import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _sieve(1000)
    L_MAX = 20000
    
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
            
            for p in primes:
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

    # Phase 1: Greedy initialization
    classes, remaining_budget = greedy_build(deadline - 30)
    
    # Phase 2: Local search with prefix preservation
    used_primes = set(p for p, r in classes)
    
    def is_prefix_covered(covered, U):
        for i in range(1, U):
            if not covered[i]:
                return False
        return True

    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        covered = get_coverage(classes)
        L = compute_L(covered)
        
        U = L + 1
        if U > L_MAX:
            break
        
        best_L = L
        best_new_classes = None
        best_new_budget = remaining_budget
        
        # Try all possible moves that cover U
        # Option A: Add new prime
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
            
            r = U % p
            new_classes = classes + [[p, r]]
            new_budget = remaining_budget - cost
            
            new_covered = get_coverage(new_classes)
            if not is_prefix_covered(new_covered, U):
                continue
            
            new_L = compute_L(new_covered)
            if new_L > best_L:
                best_L = new_L
                best_new_classes = new_classes
                best_new_budget = new_budget
        
        # Option B: Change existing prime residue
        for idx, (p, r) in enumerate(classes):
            if r == U % p:
                continue
            
            r_new = U % p
            new_classes = [list(c) for c in classes]
            new_classes[idx] = [p, r_new]
            new_budget = remaining_budget
            
            new_covered = get_coverage(new_classes)
            if not is_prefix_covered(new_covered, U):
                continue
            
            new_L = compute_L(new_covered)
            if new_L > best_L:
                best_L = new_L
                best_new_classes = new_classes
                best_new_budget = new_budget
        
        # Option C: Swap - remove one prime, add another
        for idx_remove in range(len(classes)):
            p_remove = classes[idx_remove][0]
            r_remove = classes[idx_remove][1]
            
            trial_classes = [c for i, c in enumerate(classes) if i != idx_remove]
            trial_budget = remaining_budget + Fraction(1, p_remove)
            
            for p_add in primes:
                if p_add in used_primes and p_add != p_remove:
                    continue
                cost = Fraction(1, p_add)
                if trial_budget < cost:
                    continue
                
                r_add = U % p_add
                new_classes = trial_classes + [[p_add, r_add]]
                new_budget = trial_budget - cost
                
                new_covered = get_coverage(new_classes)
                if not is_prefix_covered(new_covered, U):
                    continue
                
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_new_classes = new_classes
                    best_new_budget = new_budget
        
        if best_new_classes is not None:
            classes = best_new_classes
            remaining_budget = best_new_budget
            used_primes = set(p for p, r in classes)
            improved = True
            no_improve_count = 0
    
    # Final validation
    total_cost = sum(Fraction(1, p) for p, r in classes)
    if total_cost > Fraction(1, 1):
        # Shouldn't happen, but trim if needed
        classes.sort(key=lambda x: -1.0/x[0])
        while classes and sum(Fraction(1, p) for p, r in classes) > Fraction(1, 1):
            classes.pop()
    
    return classes
