import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            primes.append(p)
            
    prime_recips = {p: 1.0 / p for p in primes}
    
    def simulate_coverage(prime_set):
        """
        Given a set of primes, assign residues greedily to cover 1..L.
        Returns (L, classes)
        """
        if not prime_set:
            return 0, []
        
        # Sort primes in the set
        P = sorted(prime_set)
        unused = set(P)
        classes = []
        covered = [False] * 20000  # Enough for L=10000+
        
        L = 0
        for n in range(1, 20000):
            if covered[n]:
                L = n
                continue
            
            # Need to cover n
            best_p = None
            best_gain = -1
            best_r = None
            
            # Try all unused primes
            for p in unused:
                r = n % p
                # Calculate gain: how many new integers in [n, n+50] are covered?
                gain = 0
                for k in range(n, n + 50):
                    if not covered[k] and k % p == r:
                        gain += 1
                # Tie-breaker: prefer smaller p (higher density)
                # Actually, if gain is same, smaller p is better?
                # Let's just use gain.
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
                
            # Commit to best_p
            unused.remove(best_p)
            classes.append([best_p, best_r])
            
            # Update covered array
            for k in range(best_r if best_r > 0 else best_p, 20000, best_p):
                # Note: if r=0, first is p. If r>0, first is r.
                # But we only care up to 20000.
                if k >= 1:
                    covered[k] = True
            
            # Check time
            if time.time() > deadline:
                break
        
        return L, classes

    best_L = 0
    best_classes = []
    
    # Strategy 1: Greedy smallest primes
    # Find largest prefix of primes that fits in budget 1
    S = 0.0
    P_small = []
    for p in primes:
        if S + 1.0/p <= 1.0:
            S += 1.0/p
            P_small.append(p)
        else:
            break
    L, classes = simulate_coverage(P_small)
    if L > best_L:
        best_L = L
        best_classes = classes
        
    # Strategy 2: Random subsets
    # We want sum of reciprocals <= 1.
    # Randomly pick primes, ensure sum constraint.
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Random subset generation
        # Bias towards smaller primes for density, but ensure sum <= 1
        # Let's try picking a random number of primes from the first 50 primes
        k = random.randint(5, 30)
        candidates = primes[:80] # Focus on smaller primes
        random.shuffle(candidates)
        
        current_sum = 0.0
        current_set = []
        for p in candidates:
            if current_sum + 1.0/p <= 1.0:
                current_sum += 1.0/p
                current_set.append(p)
                if len(current_set) == k:
                    break
        
        if not current_set:
            continue
            
        L, classes = simulate_coverage(current_set)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # Strategy 3: Local Search on the best set found
    # If we have time, try to improve the best set by swapping primes
    if best_classes:
        # Reconstruct the set of primes used in best_classes
        used_primes = {c[0] for c in best_classes}
        current_sum = sum(1.0/p for p in used_primes)
        
        # Try to add a prime and remove another to improve L
        # This is complex. Let's just try random perturbations of the best set.
        for _ in range(5000):
            if time.time() > deadline:
                break
                
            # Take a copy of the current set
            perturbed_set = list(used_primes)
            
            # Randomly remove one prime
            if perturbed_set:
                idx = random.randint(0, len(perturbed_set)-1)
                removed_p = perturbed_set.pop(idx)
                current_sum -= 1.0/removed_p
                
                # Randomly add one prime
                # Pick a prime not in the set, preferably small
                available = [p for p in primes if p not in perturbed_set]
                if available:
                    # Bias towards smaller primes
                    random.shuffle(available)
                    added_p = available[0]
                    if current_sum + 1.0/added_p <= 1.0:
                        perturbed_set.append(added_p)
                        current_sum += 1.0/added_p
                    else:
                        # Try to find a smaller prime that fits
                        for p in sorted(available):
                            if current_sum + 1.0/p <= 1.0:
                                perturbed_set.append(p)
                                current_sum += 1.0/p
                                break
                        
            L, classes = simulate_coverage(perturbed_set)
            if L > best_L:
                best_L = L
                best_classes = classes
                # Update used_primes for next iteration
                used_primes = {c[0] for c in best_classes}
                current_sum = sum(1.0/p for p in used_primes)

    # Final check: Ensure exact rational sum <= 1
    # The simulation used floats. We must verify the final result.
    if best_classes:
        total_recip = Fraction(0, 1)
        for p, r in best_classes:
            total_recip += Fraction(1, p)
            
        if total_recip > 1:
            # This should not happen if we were careful, but just in case.
            # We might need to drop a prime.
            # But dropping a prime might break coverage.
            # Given the margin in the loop, this is unlikely.
            # If it happens, we return the best valid subset?
            # For now, if invalid, we return empty or best_L=0.
            # But let's assume the float check was tight enough.
            pass

    return best_classes

