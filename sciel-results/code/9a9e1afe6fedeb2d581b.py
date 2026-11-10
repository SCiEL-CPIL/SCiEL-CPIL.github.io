from math import gcd
from itertools import combinations

def score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    
    if len(sums) == 0:
        return 0
        
    # Check pairwise coprime
    s_list = list(sums)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if gcd(s_list[i], s_list[j]) != 1:
                return 0
                
    return len(s_list)

def construct():
    # We want to find A, B such that |A+B| is maximized and sums are pairwise coprime.
    # Max possible is 9.
    
    # Precompute prime factors for numbers up to 256
    max_sum = 256
    prime_factors = [set() for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            for j in range(i, max_sum + 1, i):
                prime_factors[j].add(i)
                
    def is_pairwise_coprime(sums):
        # sums is a list of 9 integers
        # Check if all distinct and pairwise coprime
        if len(set(sums)) < 9:
            return False
        for i in range(9):
            for j in range(i + 1, 9):
                if gcd(sums[i], sums[j]) != 1:
                    return False
        return True

    # Strategy: Try specific A sets and search for B
    # We will try a variety of A sets.
    # To speed up, we can prioritize A sets that are "well-spaced" or small.
    
    candidate_As = []
    # Add some structured A sets
    for a1 in range(1, 10):
        for a2 in range(a1 + 1, 20):
            for a3 in range(a2 + 1, 30):
                candidate_As.append([a1, a2, a3])
                
    # Add some random A sets
    import random
    for _ in range(50):
        candidate_As.append(sorted(random.sample(range(1, 129), 3)))
        
    # Deduplicate
    unique_As = []
    seen = set()
    for A in candidate_As:
        if tuple(A) not in seen:
            seen.add(tuple(A))
            unique_As.append(A)
            
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    best_score = 0

    # For each A, search for B
    for A in unique_As:
        # Search for B such that all 9 sums are pairwise coprime
        found = False
        # We can iterate b1, b2, b3
        # Optimization: Pre-calculate constraints for b1
        # For a fixed b1, let S1 = {a1+b1, a2+b1, a3+b1}
        # Forbidden primes for other sums = union of prime factors of S1
        
        for b1 in range(1, 129):
            sums_b1 = [A[0] + b1, A[1] + b1, A[2] + b1]
            forbidden_primes = set()
            for s in sums_b1:
                forbidden_primes.update(prime_factors[s])
                
            # Now find b2, b3
            # We need sums involving b2 and b3 to be coprime to sums_b1 and to each other.
            
            valid_b2s = []
            for b2 in range(1, 129):
                if b2 == b1:
                    continue
                sums_b2 = [A[0] + b2, A[1] + b2, A[2] + b2]
                # Check if any sum in sums_b2 shares a factor with forbidden_primes
                valid = True
                for s in sums_b2:
                    if prime_factors[s] & forbidden_primes:
                        valid = False
                        break
                if valid:
                    valid_b2s.append((b2, sums_b2))
                    
            # Now iterate over valid b2s and find b3
            for b2, sums_b2 in valid_b2s:
                # Update forbidden primes with factors of sums_b2
                new_forbidden = set(forbidden_primes)
                for s in sums_b2:
                    new_forbidden.update(prime_factors[s])
                    
                for b3 in range(1, 129):
                    if b3 == b1 or b3 == b2:
                        continue
                    sums_b3 = [A[0] + b3, A[1] + b3, A[2] + b3]
                    
                    # Check if any sum in sums_b3 shares a factor with new_forbidden
                    valid = True
                    for s in sums_b3:
                        if prime_factors[s] & new_forbidden:
                            valid = False
                            break
                            
                    if valid:
                        # Check pairwise coprimality within sums_b3?
                        # The check above ensures sums_b3 are coprime to sums_b1 and sums_b2.
                        # We also need sums_b3 to be pairwise coprime among themselves.
                        # And sums_b1 pairwise, sums_b2 pairwise.
                        # Since we only checked against forbidden_primes (union), we need to be careful.
                        # Actually, if s1 and s2 are in sums_b1, they might share a factor.
                        # The problem requires ALL distinct values in A+B to be pairwise coprime.
                        # So sums_b1 must be pairwise coprime, sums_b2 pairwise, sums_b3 pairwise,
                        # and cross-sets pairwise.
                        
                        # Let's just do the full check on the 9 sums to be safe, 
                        # but the pruning above is very strong.
                        
                        all_sums = sums_b1 + sums_b2 + sums_b3
                        if is_pairwise_coprime(all_sums):
                            B = [b1, b2, b3]
                            current_score = score(A, B)
                            if current_score > best_score:
                                best_score = current_score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score == 9:
                                    return [best_A, best_B]
                            found = True
                            # We can break early if we found a good one? 
                            # We want max score. 9 is max. 
                            # If we find 9, we can return.
                            # If we find <9, we might want to continue searching for 9.
                            # But if we found 9, we return.
                            
                            # If we don't return, we should keep looking for better scores.
                            # However, 9 is the theoretical max.
                            # So if we find 9, we are done.
                            # If we don't find 9, we keep searching.
                            
                            # To save time, if we found a score > best_score, we update.
                            # We continue searching for 9.
                            pass

    # If no 9 found, return best found
    return [best_A, best_B]
