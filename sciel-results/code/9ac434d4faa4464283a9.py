import time
import random
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2-128
    def prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Precompute factorization signatures for all numbers 2-128
    factor_sigs = {}
    for x in range(2, 129):
        factor_sigs[x] = tuple(sorted(prime_factors(x).items()))
        
    # Map signature to a hashable form for quick lookup
    # Actually, the tuple of (prime, exp) pairs is already hashable.
    
    def is_valid_fast(seq):
        """
        Check if the sequence has distinct products for all non-empty consecutive blocks.
        Uses prime factorization signatures.
        """
        n = len(seq)
        if n == 0:
            return True
        
        products = set()
        for i in range(n):
            # Build up the product signature incrementally
            sig = {}
            for j in range(i, n):
                # Add factorization of seq[j] to sig
                for prime, exp in factor_sigs[seq[j]]:
                    sig[prime] = sig.get(prime, 0) + exp
                
                # Create a hashable representation
                # Sorting ensures consistent ordering
                sig_tuple = tuple(sorted(sig.items()))
                
                if sig_tuple in products:
                    return False
                products.add(sig_tuple)
        return True

    # 1. Greedy Initialization
    # Try to build a long sequence greedily
    S = []
    for x in range(2, 129):
        new_S = S + [x]
        if is_valid_fast(new_S):
            S = new_S

    best_S = S[:]
    best_len = len(S)
    
    # Local Search Function
    def local_search(current_S):
        improved = True
        while improved and time.time() - start_time < limit:
            improved = False
            
            # Step 1: Try to insert any single missing number
            missing = [x for x in range(2, 129) if x not in current_S]
            # Try to insert in order of "least likely to cause conflict"
            # Heuristic: Numbers with larger prime factors are "safer"
            # Let's just shuffle to break symmetry or sort by value
            random.shuffle(missing)
            
            for x in missing:
                if time.time() - start_time > limit:
                    return current_S
                idx = bisect.bisect_left(current_S, x)
                new_S = current_S[:idx] + [x] + current_S[idx:]
                if is_valid_fast(new_S):
                    current_S = new_S
                    improved = True
                    break
            
            if not improved:
                # Step 2: Try 1-out, 1-in swaps
                # This allows changing composition without changing length
                candidates_to_remove = current_S[:]
                random.shuffle(candidates_to_remove)
                for y in candidates_to_remove:
                    if time.time() - start_time > limit:
                        return current_S
                    candidates_to_add = [x for x in range(2, 129) if x not in current_S]
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add:
                        new_S = [v for v in current_S if v != y]
                        new_S.append(x)
                        new_S.sort()
                        if is_valid_fast(new_S):
                            current_S = new_S
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Step 3: Try 1-out, 2-in moves to increase length
                if len(current_S) < 127:
                    candidates_to_remove = current_S[:]
                    random.shuffle(candidates_to_remove)
                    for y in candidates_to_remove:
                        if time.time() - start_time > limit:
                            return current_S
                        candidates_to_add = [x for x in range(2, 129) if x not in current_S]
                        if len(candidates_to_add) < 2:
                            continue
                        
                        # Try random pairs
                        found = False
                        for _ in range(20): # Limit trials per removal
                            if time.time() - start_time > limit:
                                return current_S
                            x1, x2 = random.sample(candidates_to_add, 2)
                            new_S = [v for v in current_S if v != y]
                            new_S.extend([x1, x2])
                            new_S.sort()
                            if is_valid_fast(new_S):
                                current_S = new_S
                                improved = True
                                found = True
                                break
                        if found:
                            break

        return current_S

    # Iterated Local Search
    max_iterations = 200
    for i in range(max_iterations):
        if time.time() - start_time > limit:
            break
            
        # Refine current best
        refined_S = local_search(best_S)
        
        if len(refined_S) > best_len:
            best_S = refined_S
            best_len = len(best_S)
        
        # Perturbation: Break the solution to escape local optima
        # Remove k random elements and replace with k random missing ones
        k = random.randint(2, 5)
        if len(best_S) > k:
            elements_to_remove = random.sample(best_S, k)
            new_S = [v for v in best_S if v not in elements_to_remove]
            
            missing = [x for x in range(2, 129) if x not in new_S]
            if len(missing) >= k:
                elements_to_add = random.sample(missing, k)
                new_S.extend(elements_to_add)
                new_S.sort()
                
                # If the perturbed solution is valid, use it as the new starting point
                # If not, we might still try to repair it, but for simplicity, 
                # we only accept valid perturbations or fall back to best_S
                if is_valid_fast(new_S):
                    # Don't update best_S immediately, just use new_S for next local search
                    # Actually, if new_S is valid, it's a good candidate
                    if len(new_S) > best_len:
                        best_S = new_S
                        best_len = len(new_S)
                    else:
                        # Keep best_S as the anchor, but search from new_S?
                        # Let's search from new_S
                        pass 
                else:
                    # Perturbation broke validity, revert to best_S for next iteration
                    # Or try a milder perturbation
                    pass

    return best_S
