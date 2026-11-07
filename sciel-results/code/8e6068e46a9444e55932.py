import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Known best from seed/inspirations
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Primes to consider
    # We want small primes to allow for more factors and potential collisions
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # We will store M -> list of k
    # Using a dictionary
    m_to_ks = defaultdict(list)
    
    # Recursive search
    # idx: index in primes list
    # current_k: current k
    # current_M: current M = g(k)
    def search(idx, current_k, current_M):
        if time.time() > deadline - 10:
            return
        
        # Record the solution
        # We must ensure k <= sqrt(M)
        # g(k) = k * sigma(k). Since sigma(k) >= k+1 for k>1, g(k) > k^2.
        # So k < sqrt(M) is always true for k > 1.
        # For k=1, M=1, sqrt(1)=1, k<=1 holds.
        if current_M <= 10**18:
            m_to_ks[current_M].append(current_k)
        
        if idx == len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents e for prime p
        # We multiply current_M by A(p, e)
        # A(p, e) = p^e * (p^(e+1)-1)/(p-1)
        # We can compute this iteratively
        # Let term = 1
        # A(p, 0) = 1
        # A(p, e) = A(p, e-1) * p * (1 + 1/p + ... + 1/p^e) ? No.
        # A(p, e) = p^e * sigma(p^e)
        # sigma(p^e) = (p^(e+1)-1)/(p-1)
        
        # Let's just compute A(p, e) for e=1, 2, ...
        # A(p, e) = p^e * (p^(e+1)-1)//(p-1)
        
        # Optimization: 
        # current_M * A(p, e) <= 10^18
        # A(p, e) grows very fast.
        
        # We can start with e=1 and keep going until product exceeds limit
        # But we also need to consider that we might not use this prime at all (e=0)
        # That is handled by the recursive call with idx+1 and e=0 effectively (by not multiplying)
        # Wait, the recursion structure:
        # search(idx+1, current_k, current_M) -> e=0 case
        # Then loop e=1, 2, ... for this prime
        
        # Case e=0
        search(idx + 1, current_k, current_M)
        
        # Case e >= 1
        # We need to calculate A(p, e)
        # Let power = 1
        # Let sigma_part = 1
        # For e=1:
        # power = p
        # sigma_part = 1 + p
        # A = p * (1+p)
        
        # Iterative update:
        # A(p, e) = p^e * (1 + p + ... + p^e)
        # Let S_e = 1 + p + ... + p^e
        # S_e = S_{e-1} + p^e
        # A(p, e) = p^e * S_e
        
        # We can maintain p_pow and S
        p_pow = 1
        S = 1
        # e starts at 1
        # We want to find max e such that current_M * p^e * S_e <= 10^18
        # Note: current_M might be large, so we check overflow
        
        max_M = 10**18
        # To avoid overflow in intermediate steps, we can check 
        # p_pow * S <= max_M // current_M
        
        limit = max_M // current_M
        
        while True:
            # Update for next e
            # Previous e was e-1. Now we are at e.
            # p_pow becomes p^e
            # S becomes S_{e-1} + p^e
            
            # Actually, let's just compute A for each e from scratch or incrementally.
            # Incremental:
            # e=1:
            # p_pow = p
            # S = 1 + p
            # A = p * (1+p)
            
            # Let's use a loop variable e
            # We already handled e=0.
            
            # We need to be careful with the limit.
            # If limit < 1, break.
            if limit < 1:
                break
            
            # Compute A(p, e)
            # We can do this by keeping track of p_pow and S
            # But we are in a loop. Let's restart the incremental logic inside the loop?
            # No, we can just calculate A(p, e) directly for small e.
            # e is small (< 60).
            
            # Let's just use a helper to get A(p, e)
            # But calling a function in a tight loop is slow.
            # Let's maintain state.
            
            # State: p_pow = p^e, S = sigma(p^e)
            # Init for e=1:
            # p_pow = p
            # S = 1 + p
            
            # Let's refactor the loop to maintain state.
            # But we called search(idx+1) already.
            # Now we enter the e>=1 loop.
            
            # We need to initialize p_pow and S for e=1.
            # Then update for e=2, etc.
            
            # Wait, the code above is inside the function.
            # I will rewrite the loop properly.
            
            pass # Placeholder for the actual loop logic below

    # The recursive function needs to be defined carefully to avoid overhead.
    # Let's implement the search iteratively or with a tighter recursive structure.
    
    # Re-implementing search
    def search_v2(idx, current_k, current_M):
        if time.time() > deadline - 10:
            return
        
        if current_M <= 10**18:
            m_to_ks[current_M].append(current_k)
        
        if idx == len(primes):
            return
            
        p = primes[idx]
        
        # Option 1: e = 0
        search_v2(idx + 1, current_k, current_M)
        
        # Option 2: e >= 1
        # We will iterate e = 1, 2, ...
        # Maintain p_pow = p^e and S = sigma(p^e)
        
        p_pow = 1
        S = 1
        # e = 1
        # p_pow = p
        # S = 1 + p
        # A = p * (1+p)
        
        # Check if we can even do e=1
        # current_M * p * (1+p) <= 10^18
        
        # To avoid overflow, check p * (1+p) <= 10^18 // current_M
        # But 10^18 // current_M might be 0 if current_M > 10^18 (but we check current_M <= 10^18 before)
        
        # Let's just try to compute and break if exceeds.
        
        # e=1
        p_pow = p
        S = 1 + p
        A = p_pow * S
        if current_M * A <= 10**18:
            search_v2(idx + 1, current_k * p_pow, current_M * A)
            
            # e=2
            p_pow *= p
            S += p_pow
            A = p_pow * S
            if current_M * A <= 10**18:
                search_v2(idx + 1, current_k * p_pow, current_M * A)
                
                # e=3
                p_pow *= p
                S += p_pow
                A = p_pow * S
                if current_M * A <= 10**18:
                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                    
                    # e=4
                    p_pow *= p
                    S += p_pow
                    A = p_pow * S
                    if current_M * A <= 10**18:
                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                        
                        # e=5
                        p_pow *= p
                        S += p_pow
                        A = p_pow * S
                        if current_M * A <= 10**18:
                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                            
                            # e=6
                            p_pow *= p
                            S += p_pow
                            A = p_pow * S
                            if current_M * A <= 10**18:
                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                
                                # e=7
                                p_pow *= p
                                S += p_pow
                                A = p_pow * S
                                if current_M * A <= 10**18:
                                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                                    
                                    # e=8
                                    p_pow *= p
                                    S += p_pow
                                    A = p_pow * S
                                    if current_M * A <= 10**18:
                                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                                        
                                        # e=9
                                        p_pow *= p
                                        S += p_pow
                                        A = p_pow * S
                                        if current_M * A <= 10**18:
                                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                                            
                                            # e=10
                                            p_pow *= p
                                            S += p_pow
                                            A = p_pow * S
                                            if current_M * A <= 10**18:
                                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                
                                                # e=11
                                                p_pow *= p
                                                S += p_pow
                                                A = p_pow * S
                                                if current_M * A <= 10**18:
                                                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                    
                                                    # e=12
                                                    p_pow *= p
                                                    S += p_pow
                                                    A = p_pow * S
                                                    if current_M * A <= 10**18:
                                                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                        
                                                        # e=13
                                                        p_pow *= p
                                                        S += p_pow
                                                        A = p_pow * S
                                                        if current_M * A <= 10**18:
                                                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                                
                                                            # e=14
                                                            p_pow *= p
                                                            S += p_pow
                                                            A = p_pow * S
                                                            if current_M * A <= 10**18:
                                                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                                
    # Start search
    search_v2(0, 1, 1)
    
    # Find best M
    max_count = len(best_ks)
    best_M_found = best_M
    best_ks_found = best_ks
    
    for M, ks in m_to_ks.items():
        # Remove duplicates if any (shouldn't be, but just in case)
        unique_ks = list(set(ks))
        count = len(unique_ks)
        if count > max_count:
            max_count = count
            best_M_found = M
            best_ks_found = unique_ks
            
    return {'M': best_M_found, 'k': best_ks_found}

# To avoid recursion limit issues and overhead, we might need to optimize.
# But 20 primes, max depth 20. Branching factor is small (max e ~ 15 for p=2, less for others).
# Total nodes might be manageable.

# One risk: recursion depth. Python default is 1000. Depth is 20. Safe.
# Another risk: time.
# We will add a timeout check in the main loop if we were iterating, but here it's recursive.
# The check `if time.time() > deadline - 10` is inside search_v2.

# Let's refine the primes list to be slightly smaller to ensure speed, or keep 20.
# 20 primes is fine.

# Final check on types
# M is int, k is list of ints.

# We will wrap the construct function to ensure it returns valid types.

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_M = 333312
    best_ks = [336, 372, 434]
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    m_to_ks = defaultdict(list)
    
    def search_v2(idx, current_k, current_M):
        if time.time() > deadline - 10:
            return
        
        if current_M <= 10**18:
            m_to_ks[current_M].append(current_k)
        
        if idx == len(primes):
            return
            
        p = primes[idx]
        
        search_v2(idx + 1, current_k, current_M)
        
        p_pow = 1
        S = 1
        
        # Unrolled loop for e=1 to 14
        # e=1
        p_pow = p
        S = 1 + p
        A = p_pow * S
        if current_M * A <= 10**18:
            search_v2(idx + 1, current_k * p_pow, current_M * A)
            
            # e=2
            p_pow *= p
            S += p_pow
            A = p_pow * S
            if current_M * A <= 10**18:
                search_v2(idx + 1, current_k * p_pow, current_M * A)
                
                # e=3
                p_pow *= p
                S += p_pow
                A = p_pow * S
                if current_M * A <= 10**18:
                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                    
                    # e=4
                    p_pow *= p
                    S += p_pow
                    A = p_pow * S
                    if current_M * A <= 10**18:
                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                        
                        # e=5
                        p_pow *= p
                        S += p_pow
                        A = p_pow * S
                        if current_M * A <= 10**18:
                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                            
                            # e=6
                            p_pow *= p
                            S += p_pow
                            A = p_pow * S
                            if current_M * A <= 10**18:
                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                
                                # e=7
                                p_pow *= p
                                S += p_pow
                                A = p_pow * S
                                if current_M * A <= 10**18:
                                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                                    
                                    # e=8
                                    p_pow *= p
                                    S += p_pow
                                    A = p_pow * S
                                    if current_M * A <= 10**18:
                                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                                        
                                        # e=9
                                        p_pow *= p
                                        S += p_pow
                                        A = p_pow * S
                                        if current_M * A <= 10**18:
                                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                                            
                                            # e=10
                                            p_pow *= p
                                            S += p_pow
                                            A = p_pow * S
                                            if current_M * A <= 10**18:
                                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                
                                                # e=11
                                                p_pow *= p
                                                S += p_pow
                                                A = p_pow * S
                                                if current_M * A <= 10**18:
                                                    search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                    
                                                    # e=12
                                                    p_pow *= p
                                                    S += p_pow
                                                    A = p_pow * S
                                                    if current_M * A <= 10**18:
                                                        search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                        
                                                        # e=13
                                                        p_pow *= p
                                                        S += p_pow
                                                        A = p_pow * S
                                                        if current_M * A <= 10**18:
                                                            search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                                
                                                            # e=14
                                                            p_pow *= p
                                                            S += p_pow
                                                            A = p_pow * S
                                                            if current_M * A <= 10**18:
                                                                search_v2(idx + 1, current_k * p_pow, current_M * A)
                                                                
    search_v2(0, 1, 1)
    
    max_count = len(best_ks)
    best_M_found = best_M
    best_ks_found = best_ks
    
    for M, ks in m_to_ks.items():
        unique_ks = list(set(ks))
        count = len(unique_ks)
        if count > max_count:
            max_count = count
            best_M_found = M
            best_ks_found = unique_ks
            
    return {'M': int(best_M_found), 'k': [int(k) for k in best_ks_found]}
