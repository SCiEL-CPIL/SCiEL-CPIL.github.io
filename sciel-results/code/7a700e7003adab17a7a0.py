import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    primes = sieve(1000)
    max_L_target = 20000

    def compute_L(covered, max_L):
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def build_coverage(classes, max_L):
        covered = bytearray(max_L + 1)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, max_L + 1, p):
                covered[k] = 1
        return covered

    def get_first_uncovered(covered, start=1, max_L=None):
        if max_L is None:
            max_L = len(covered) - 1
        for i in range(start, max_L + 1):
            if not covered[i]:
                return i
        return -1

    def greedy_build(alpha, max_L_target, deadline):
        """Original greedy as fallback."""
        covered = bytearray(max_L_target + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= max_L_target:
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
                while k <= max_L_target:
                    if not covered[k]:
                        gain += 1
                    k += p
                
                if gain == 0:
                    continue
                
                score = gain * p * (1 + alpha * (p / 1000.0))
                
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
            for k in range(start_k, max_L_target + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        L = compute_L(covered, max_L_target)
        return L, classes

    def backtrack_search(max_depth, max_L_target, deadline):
        """
        Backtracking search that tries to extend coverage.
        At each step, find first uncovered number U.
        Try all (p, r) pairs where r = U % p and p not used.
        Choose by efficiency score, but backtrack if stuck.
        """
        covered = bytearray(max_L_target + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        best_L = 0
        best_classes = []
        
        def search(depth, current_L):
            nonlocal best_L, best_classes, covered, used_primes, budget, classes
            
            if time.time() > deadline:
                return False
            
            if current_L > best_L:
                best_L = current_L
                best_classes = [list(c) for c in classes]
            
            if depth >= max_depth:
                return False
            
            # Find first uncovered number
            U = get_first_uncovered(covered, 1, max_L_target)
            if U == -1:
                return True  # All covered
            
            # Try all possible primes that can cover U
            candidates = []
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = U % p
                # Calculate gain
                gain = 0
                k = U
                while k <= max_L_target:
                    if not covered[k]:
                        gain += 1
                    k += p
                if gain > 0:
                    # Score: prefer large primes (cheaper) and high gain
                    score = gain * p
                    candidates.append((score, p, r, gain))
            
            if not candidates:
                return False
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            # Try top K candidates
            K = min(3, len(candidates))
            for score, p, r, gain in candidates[:K]:
                if time.time() > deadline:
                    return False
                
                # Add this class
                used_primes.add(p)
                old_budget = budget
                budget -= Fraction(1, p)
                classes.append([p, r])
                
                # Update coverage
                old_covered = bytearray(covered)
                start_k = r
                if start_k == 0:
                    start_k = p
                for k in range(start_k, max_L_target + 1, p):
                    covered[k] = 1
                
                # Compute new L
                new_L = compute_L(covered, max_L_target)
                
                if new_L > current_L:
                    # Recurse
                    search(depth + 1, new_L)
                
                # Backtrack
                covered = old_covered
                used_primes.remove(p)
                budget = old_budget
                classes.pop()
                
                if time.time() > deadline:
                    return False
            
            return False
        
        search(0, 0)
        return best_L, best_classes

    def local_improve(classes, max_L_target, deadline):
        """
        Try to improve by swapping residues or adding/removing classes.
        """
        covered = build_coverage(classes, max_L_target)
        current_L = compute_L(covered, max_L_target)
        best_L = current_L
        best_classes = [list(c) for c in classes]
        used_primes = set(p for p, r in classes)
        budget_used = sum(Fraction(1, p) for p, r in classes)
        remaining = Fraction(1, 1) - budget_used
        
        # Try to cover first uncovered number
        U = get_first_uncovered(covered, 1, max_L_target)
        if U == -1:
            return best_L, best_classes
        
        # Option 1: Add new prime
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if cost > remaining:
                continue
            r = U % p
            new_classes = best_classes + [[p, r]]
            new_covered = build_coverage(new_classes, max_L_target)
            new_L = compute_L(new_covered, max_L_target)
            if new_L > best_L:
                best_L = new_L
                best_classes = new_classes
                used_primes.add(p)
                remaining -= cost
                U = get_first_uncovered(new_covered, 1, max_L_target)
                if U == -1:
                    break
        
        # Option 2: Change residue of existing prime
        for i, (p, old_r) in enumerate(best_classes):
            U = get_first_uncovered(build_coverage(best_classes, max_L_target), 1, max_L_target)
            if U == -1:
                break
            new_r = U % p
            if new_r == old_r:
                continue
            best_classes[i] = [p, new_r]
            new_covered = build_coverage(best_classes, max_L_target)
            new_L = compute_L(new_covered, max_L_target)
            if new_L > best_L:
                best_L = new_L
                # Keep the change
            else:
                # Revert
                best_classes[i] = [p, old_r]
        
        return best_L, best_classes

    best_L = 0
    best_classes = []
    
    # Strategy 1: Greedy with different alphas
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        if time.time() > deadline:
            break
        L, classes = greedy_build(alpha, max_L_target, deadline)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
    
    # Strategy 2: Backtracking search
    if time.time() < deadline - 20:
        L, classes = backtrack_search(15, max_L_target, deadline)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
    
    # Strategy 3: Local improvement
    if time.time() < deadline - 10 and best_classes:
        L, classes = local_improve(best_classes, max_L_target, deadline)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
    
    # Final verification
    if best_classes:
        covered = build_coverage(best_classes, max_L_target)
        final_L = compute_L(covered, max_L_target)
        budget_used = sum(Fraction(1, p) for p, r in best_classes)
        if budget_used <= Fraction(1, 1):
            best_L = final_L
        else:
            # Should not happen, but safety
            best_classes = []
            best_L = 0
    
    return best_classes
