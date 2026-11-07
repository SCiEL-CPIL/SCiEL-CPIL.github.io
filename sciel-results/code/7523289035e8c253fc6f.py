"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Generate primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [i for i in range(2, P['n'] + 1) if is_prime(i)]
    total_primes = len(primes)
    
    # Precompute prime factors for all numbers up to 256
    prime_factors = [[] for _ in range(P['n'] + 1)]
    for i in range(2, P['n'] + 1):
        if not prime_factors[i]: # if prime
            for j in range(i, P['n'] + 1, i):
                prime_factors[j].append(i)

    # Function to check if a set of composites is valid
    def is_valid_composite_set(A):
        # A is a list of distinct ints
        n = len(A)
        if n < 3:
            return True
        # For each a in A, check if a divides b*c for any distinct b,c in A
        # Optimization: if a is large, it's less likely to divide b*c
        for i in range(n):
            a = A[i]
            for j in range(n):
                if i == j: continue
                b = A[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = A[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Function to find max valid subset of C_out
    def max_valid_subset(C_out):
        if not C_out:
            return []
        # Heuristic: sort descending to pick large numbers first (less restrictive)
        C_sorted = sorted(C_out, reverse=True)
        best_A = []
        current_A = []
        
        # Try greedy forward
        for x in C_sorted:
            new_A = current_A + [x]
            if is_valid_composite_set(new_A):
                current_A = new_A
            else:
                # Try to see if we can swap? Or just skip
                pass
        
        if len(current_A) > len(best_A):
            best_A = current_A
            
        # Try a few random restarts or different orderings if time permits
        # For now, greedy is a good baseline.
        # Let's also try ascending
        C_sorted_asc = sorted(C_out)
        current_A2 = []
        for x in C_sorted_asc:
            new_A = current_A2 + [x]
            if is_valid_composite_set(new_A):
                current_A2 = new_A
        if len(current_A2) > len(best_A):
            best_A = current_A2
            
        return best_A

    best_score = 0
    best_result = primes # Fallback: all primes
    
    # Candidate primes for P_out
    # Only primes that can form composites <= 256
    # p^2 <= 256 => p <= 16
    # 2p <= 256 => p <= 128
    # So primes up to 128 are candidates if combined with 2.
    # But to keep search space small, let's focus on small primes.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127
    # Let's try subsets of the first 10 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    candidate_primes = [p for p in primes if p <= 31]
    
    # Iterate over subsets of candidate_primes
    # To limit time, we can limit subset size or use pruning
    # Max size of P_out?
    # If |P_out| is large, |P_in| is small.
    # Score = |P_in| + |A| = (total_primes - |P_out|) + |A|
    # We want |A| > |P_out|
    
    # We will check all subsets of candidate_primes up to size 5 or so, 
    # and also some random larger subsets.
    
    # First, check empty set (all primes)
    # Already covered by best_result initialization
    
    # Check subsets of size 1
    for i in range(len(candidate_primes)):
        P_out = [candidate_primes[i]]
        P_out_set = set(P_out)
        
        # Generate C_out
        C_out = []
        for x in range(4, P['n'] + 1):
            if not is_prime(x):
                factors = prime_factors[x]
                if all(f in P_out_set for f in factors):
                    C_out.append(x)
        
        A = max_valid_subset(C_out)
        score = (total_primes - len(P_out)) + len(A)
        if score > best_score:
            best_score = score
            # Reconstruct S
            P_in = [p for p in primes if p not in P_out_set]
            best_result = P_in + A
            
        if time.time() > deadline:
            break

    # Check subsets of size 2
    if time.time() < deadline:
        for i in range(len(candidate_primes)):
            for j in range(i + 1, len(candidate_primes)):
                P_out = [candidate_primes[i], candidate_primes[j]]
                P_out_set = set(P_out)
                
                C_out = []
                for x in range(4, P['n'] + 1):
                    if not is_prime(x):
                        factors = prime_factors[x]
                        if all(f in P_out_set for f in factors):
                            C_out.append(x)
                
                A = max_valid_subset(C_out)
                score = (total_primes - len(P_out)) + len(A)
                if score > best_score:
                    best_score = score
                    P_in = [p for p in primes if p not in P_out_set]
                    best_result = P_in + A
                    
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Check subsets of size 3
    if time.time() < deadline:
        for i in range(len(candidate_primes)):
            for j in range(i + 1, len(candidate_primes)):
                for k in range(j + 1, len(candidate_primes)):
                    P_out = [candidate_primes[i], candidate_primes[j], candidate_primes[k]]
                    P_out_set = set(P_out)
                    
                    C_out = []
                    for x in range(4, P['n'] + 1):
                        if not is_prime(x):
                            factors = prime_factors[x]
                            if all(f in P_out_set for f in factors):
                                C_out.append(x)
                    
                    A = max_valid_subset(C_out)
                    score = (total_primes - len(P_out)) + len(A)
                    if score > best_score:
                        best_score = score
                        P_in = [p for p in primes if p not in P_out_set]
                        best_result = P_in + A
                        
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Check subsets of size 4
    if time.time() < deadline:
        for i in range(len(candidate_primes)):
            for j in range(i + 1, len(candidate_primes)):
                for k in range(j + 1, len(candidate_primes)):
                    for l in range(k + 1, len(candidate_primes)):
                        P_out = [candidate_primes[i], candidate_primes[j], candidate_primes[k], candidate_primes[l]]
                        P_out_set = set(P_out)
                        
                        C_out = []
                        for x in range(4, P['n'] + 1):
                            if not is_prime(x):
                                factors = prime_factors[x]
                                if all(f in P_out_set for f in factors):
                                    C_out.append(x)
                        
                        A = max_valid_subset(C_out)
                        score = (total_primes - len(P_out)) + len(A)
                        if score > best_score:
                            best_score = score
                            P_in = [p for p in primes if p not in P_out_set]
                            best_result = P_in + A
                            
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Ensure result is a list of ints
    return list(best_result)

# EVOLVE-BLOCK-END
