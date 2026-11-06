import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for q in range(2, int(n**0.5) + 1):
            if n % q == 0:
                return False
        return True
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    def compute_L(classes):
        """Compute the largest L such that every integer 1..L is covered."""
        if not classes:
            return 0
        # Use a bitset approach for efficiency
        # We'll check up to 100000
        max_check = 100000
        covered = [False] * (max_check + 1)
        for p, r in classes:
            start = r
            if start == 0:
                start = p
            for i in range(start, max_check + 1, p):
                covered[i] = True
        
        # Find largest L where all 1..L are covered
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    def greedy_cover(L):
        """Greedy algorithm to cover 1..L."""
        uncovered = [True] * (L + 1)
        uncovered[0] = False
        covered_count = L
        budget = Fraction(0, 1)
        used_primes = set()
        classes = []
        
        while covered_count > 0:
            if time.time() > deadline:
                return None
            best_p = None
            best_r = None
            best_count = 0
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget + cost > 1:
                    continue
                counts = [0] * p
                for i in range(1, L + 1):
                    if uncovered[i]:
                        counts[i % p] += 1
                max_count = max(counts)
                if max_count > best_count:
                    best_count = max_count
                    best_p = p
                    best_r = counts.index(max_count)
            
            if best_p is None or best_count == 0:
                break
            
            classes.append([best_p, best_r])
            used_primes.add(best_p)
            budget += Fraction(1, best_p)
            start = best_r
            if start == 0:
                start = best_p
            for i in range(start, L + 1, best_p):
                if uncovered[i]:
                    uncovered[i] = False
                    covered_count -= 1
        
        if covered_count == 0:
            return classes
        return None
    
    def local_search(classes, max_iterations=5000):
        """Local search to improve coverage by flipping residues or swapping primes."""
        current_L = compute_L(classes)
        best_classes = [c[:] for c in classes]
        best_L = current_L
        used_primes = set(p for p, r in classes)
        budget = sum(Fraction(1, p) for p, r in classes)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try flipping a residue
            improved = False
            for idx in range(len(classes)):
                p, r = classes[idx]
                # Try all other residues
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Compute new L
                    test_classes = [c[:] for c in classes]
                    test_classes[idx][1] = new_r
                    test_L = compute_L(test_classes)
                    if test_L > best_L:
                        best_L = test_L
                        best_classes = test_classes
                        classes = [c[:] for c in test_classes]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try removing a class and re-adding a different prime
                for idx in range(len(classes)):
                    p, r = classes[idx]
                    # Try adding a new prime
                    for new_p in primes:
                        if new_p in used_primes:
                            continue
                        if budget + Fraction(1, new_p) > 1:
                            continue
                        # Try all residues for new_p
                        for new_r in range(new_p):
                            test_classes = [c[:] for c in classes]
                            test_classes[idx] = [new_p, new_r]
                            test_L = compute_L(test_classes)
                            if test_L > best_L:
                                best_L = test_L
                                best_classes = test_classes
                                classes = [c[:] for c in test_classes]
                                used_primes.remove(p)
                                used_primes.add(new_p)
                                budget = sum(Fraction(1, pp) for pp, rr in classes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_classes, best_L
    
    def iterated_local_search(L_target, num_restarts=5):
        """Iterated local search with multiple restarts."""
        best_classes = None
        best_L = 0
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Greedy initialization
            classes = greedy_cover(L_target)
            if classes is None:
                # Try with a slightly smaller target
                for shrink in range(1, 10):
                    classes = greedy_cover(L_target - shrink)
                    if classes is not None:
                        break
                if classes is None:
                    continue
            
            # Local search
            classes, L = local_search(classes)
            
            if L > best_L:
                best_L = L
                best_classes = classes
        
        return best_classes, best_L
    
    # Main search
    best_classes = None
    best_L = 0
    
    # Start with a reasonable target
    test_L = 500
    
    while test_L <= 100000 and time.time() < deadline - 20:
        classes, L = iterated_local_search(test_L)
        
        if classes is not None and L > best_L:
            best_L = L
            best_classes = classes
        
        # Adjust test_L
        if L >= test_L:
            # Try larger
            if test_L < 1000:
                test_L = min(test_L * 2, 100000)
            else:
                test_L += 200
        else:
            # Try smaller
            if test_L <= 100:
                break
            test_L = max(test_L // 2, 100)
    
    # Final local search on the best result
    if best_classes is not None and time.time() < deadline - 5:
        best_classes, best_L = local_search(best_classes, max_iterations=10000)
    
    if best_classes is None:
        # Fallback to seed
        primes_seed = [p for p in range(13, 1001) if is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    
    return best_classes
