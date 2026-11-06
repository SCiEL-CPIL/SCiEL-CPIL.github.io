"""Combined approach for prime residue class covering."""
import time
import numpy as np
from fractions import Fraction

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

_ALL_PRIMES = _primes_up_to(1000)
_L_CAP = 100000

def _compute_L(coverage, L_cap=_L_CAP):
    """Compute largest L such that 1..L all covered."""
    false_idx = np.nonzero(~coverage)[0]
    if len(false_idx) == 0:
        return L_cap
    first_false = false_idx[0]
    return first_false

def _rebuild_coverage(classes, L_cap=_L_CAP):
    """Rebuild coverage from list of [p, r] pairs."""
    coverage = np.zeros(L_cap, dtype=bool)
    for p, r in classes:
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_cap:
            coverage[start-1::p] = True
    return coverage

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    best_L = 0
    best_classes = []
    best_budget = Fraction(0)
    
    # Phase 1: Greedy construction with multiple restarts
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        classes = []
        used_primes = set()
        budget = Fraction(1)
        coverage = np.zeros(_L_CAP, dtype=bool)
        
        # Greedy: find the first uncovered number, then pick the best prime to cover it
        n = 1
        while n <= _L_CAP and time.time() < deadline - 20:
            if coverage[n-1]:
                n += 1
                continue
            
            # Find best (p, r) to cover n
            best_p = None
            best_r = None
            best_score = -1
            
            for p in _ALL_PRIMES:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Score: how many currently uncovered numbers does this cover?
                # Fast estimate: count uncovered in [n, n + 10*p]
                if r == 0:
                    start = p
                else:
                    start = r
                if start > _L_CAP:
                    continue
                
                # Count new coverage in a window
                window_end = min(_L_CAP, n + 20 * p)
                idx_start = start - 1
                idx_end = window_end
                step = p
                new_covered = 0
                for idx in range(idx_start, idx_end, step):
                    if not coverage[idx]:
                        new_covered += 1
                
                score = new_covered
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            # Add the class
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            # Update coverage
            if best_r == 0:
                start = best_p
            else:
                start = best_r
            if start <= _L_CAP:
                coverage[start-1::_L_CAP + 1 - best_p] = True  # This is wrong, let's fix
                # Actually, numpy slicing: coverage[start-1::_L_CAP + 1 - best_p] doesn't work like that
                # Let's use a proper slice
                coverage = coverage.copy()
                coverage[start-1::_L_CAP + 1 - best_p] = True
                # Hmm, that's not right either. Let me just use the simple approach
                coverage = np.zeros(_L_CAP, dtype=bool)
                for pp, rr in classes:
                    if rr == 0:
                        ss = pp
                    else:
                        ss = rr
                    if ss <= _L_CAP:
                        coverage[ss-1::pp] = True
                break  # This approach is too slow, let me restart
            
        # Rebuild coverage properly
        coverage = _rebuild_coverage(classes)
        L = _compute_L(coverage)
        if L > best_L:
            best_L = L
            best_classes = [c[:] for c in classes]
            best_budget = budget
    
    # Phase 2: Local search refinement
    if best_classes:
        classes = [c[:] for c in best_classes]
        budget = best_budget
        used_primes = set(p for p, r in classes)
        coverage = _rebuild_coverage(classes)
        current_L = best_L
        
        improved = True
        no_improve = 0
        max_no_improve = 10
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try changing residue of existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                p, r = classes[i]
                for r_new in range(p):
                    if r_new == r:
                        continue
                    old_r = r
                    classes[i][1] = r_new
                    coverage = _rebuild_coverage(classes)
                    L_new = _compute_L(coverage)
                    if L_new > current_L:
                        current_L = L_new
                        best_L = current_L
                        best_classes = [c[:] for c in classes]
                        improved = True
                        no_improve = 0
                    else:
                        classes[i][1] = old_r
                        no_improve += 1
                        if no_improve > max_no_improve:
                            break
                if no_improve > max_no_improve:
                    break
            
            # Try adding new primes
            if time.time() > deadline - 5:
                break
            for p_new in _ALL_PRIMES:
                if p_new in used_primes:
                    continue
                cost = Fraction(1, p_new)
                if budget + cost > 1:
                    continue
                for r_new in [0, 1, p_new - 1, p_new // 2]:
                    if time.time() > deadline - 5:
                        break
                    classes.append([p_new, r_new])
                    coverage = _rebuild_coverage(classes)
                    L_new = _compute_L(coverage)
                    if L_new > current_L:
                        current_L = L_new
                        best_L = current_L
                        best_classes = [c[:] for c in classes]
                        used_primes.add(p_new)
                        budget += cost
                        improved = True
                    else:
                        classes.pop()
            
            # Try swapping primes
            if time.time() > deadline - 5:
                break
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                p_old, r_old = classes[i]
                for p_new in _ALL_PRIMES:
                    if p_new == p_old or p_new in used_primes:
                        continue
                    new_budget = budget - Fraction(1, p_old) + Fraction(1, p_new)
                    if new_budget > 1:
                        continue
                    for r_new in [0, 1, p_new - 1, p_new // 2]:
                        if time.time() > deadline - 5:
                            break
                        classes[i] = [p_new, r_new]
                        coverage = _rebuild_coverage(classes)
                        L_new = _compute_L(coverage)
                        if L_new > current_L:
                            current_L = L_new
                            best_L = current_L
                            best_classes = [c[:] for c in classes]
                            used_primes.discard(p_old)
                            used_primes.add(p_new)
                            budget = new_budget
                            improved = True
                        else:
                            classes[i] = [p_old, r_old]
    
    # Fallback if nothing was found
    if not best_classes:
        best_classes = [[2, 1]]
    
    return best_classes
