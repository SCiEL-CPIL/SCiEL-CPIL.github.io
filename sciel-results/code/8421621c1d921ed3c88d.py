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
    
    # Get all primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)
    
    # Precompute factor vectors for each number 2-128
    factor_vecs = {}
    for x in range(2, 129):
        vec = [0] * nprimes
        for p, e in prime_factors(x).items():
            vec[prime_to_idx[p]] = e
        factor_vecs[x] = vec
    
    def is_valid_fast(seq):
        """
        Check if the sequence has distinct products for all non-empty consecutive blocks.
        Uses prefix factorization vectors for fast block product computation.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Build prefix factorization vectors
        # prefix[i] = sum of factor_vecs[seq[0]], ..., factor_vecs[seq[i-1]]
        prefix = [[0] * nprimes for _ in range(n + 1)]
        for i in range(n):
            fv = factor_vecs[seq[i]]
            for j in range(nprimes):
                prefix[i + 1][j] = prefix[i][j] + fv[j]
        
        products = set()
        for i in range(n):
            # Block starting at i
            # For each j >= i, block [i..j] has factor vec = prefix[j+1] - prefix[i]
            for j in range(i, n):
                # Compute factor vector for block [i..j]
                block_vec = tuple(prefix[j + 1][k] - prefix[i][k] for k in range(nprimes))
                
                if block_vec in products:
                    return False
                products.add(block_vec)
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
                        for _ in range(20):
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
            
            if not improved:
                # Step 4: Try 2-out, 2-in moves (same length, different composition)
                if len(current_S) >= 2:
                    candidates_to_remove = current_S[:]
                    random.shuffle(candidates_to_remove)
                    found = False
                    for i in range(len(candidates_to_remove)):
                        if time.time() - start_time > limit:
                            return current_S
                        y1 = candidates_to_remove[i]
                        for j in range(i + 1, len(candidates_to_remove)):
                            if time.time() - start_time > limit:
                                return current_S
                            y2 = candidates_to_remove[j]
                            candidates_to_add = [x for x in range(2, 129) if x not in current_S]
                            if len(candidates_to_add) < 2:
                                break
                            # Try random pairs to add
                            for _ in range(10):
                                if time.time() - start_time > limit:
                                    return current_S
                                x1, x2 = random.sample(candidates_to_add, 2)
                                new_S = [v for v in current_S if v != y1 and v != y2]
                                new_S.extend([x1, x2])
                                new_S.sort()
                                if is_valid_fast(new_S):
                                    current_S = new_S
                                    improved = True
                                    found = True
                                    break
                            if found:
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
                if is_valid_fast(new_S):
                    if len(new_S) > best_len:
                        best_S = new_S
                        best_len = len(new_S)
                else:
                    # Try a milder perturbation: remove 1, add 1
                    if len(best_S) > 0:
                        y = random.choice(best_S)
                        missing2 = [x for x in range(2, 129) if x not in best_S]
                        if missing2:
                            x = random.choice(missing2)
                            mild_S = [v for v in best_S if v != y]
                            mild_S.append(x)
                            mild_S.sort()
                            if is_valid_fast(mild_S):
                                # Use this as next starting point but don't update best
                                pass

    return best_S
