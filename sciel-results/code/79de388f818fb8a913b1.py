import random
import time

def construct():
    deadline = time.time() + 50  # 50s limit for safety
    
    # 1. Get primes
    primes = []
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Initial set: all primes
    current_set = set(primes)
    best_set = set(primes)
    best_size = len(best_set)
    
    # Candidates for addition: composites
    composites = [x for x in range(2, 257) if x not in current_set]
    
    def is_valid_set(S):
        # Check if S is valid
        # For every distinct a, b, c in S, a does not divide b*c
        # Optimization: 
        # If a is prime p, p | bc iff p|b or p|c.
        # So if p in S, no two other elements can be multiples of p? 
        # No, if b is multiple of p, c is not, then p|bc is False? 
        # Wait. p | b*c. If p|b, then p|b*c.
        # So if p in S, and b in S is multiple of p, then for ANY c in S, p|bc.
        # Thus, if p in S, there can be AT MOST ONE other element in S that is a multiple of p?
        # No. If b is multiple of p, then p|bc for ALL c.
        # So if p in S, and there exists b in S \ {p} such that p|b, then the set is INVALID 
        # because we can pick any third distinct c.
        # Wait, the condition is "For every THREE DISTINCT members a,b,c".
        # If p|b, then p|bc.
        # So if we have p, b (mult of p), and c (any other), then p|bc.
        # So the set is invalid if it contains p and any multiple of p (other than p).
        # UNLESS the set has size < 3. But size is large.
        # So: IF p is in S, then NO other element in S can be a multiple of p.
        # This is a strong constraint!
        
        # Let's verify.
        # a=p, b=2p, c=3.
        # p | 2p*3 = 6p. Yes.
        # So p, 2p, 3 cannot be together.
        # So if p is in S, all multiples of p (other than p) must be excluded.
        
        # This simplifies things greatly.
        # If we include a prime p, we cannot include any other multiple of p.
        # If we include a composite c, we don't have this "blocking all multiples" property 
        # unless c itself is a prime power?
        # If c = p^2, and b = p, then p^2 | p*q? No, p^2 | pq is false.
        # But if a=p, b=p^2, c=q. p | p^2 q. Yes.
        # So if p is in S, we cannot have p^2.
        # If p^2 is in S, we cannot have p.
        
        # So the rule "If p in S, no other multiple of p" applies to primes.
        # Does it apply to composites?
        # If a = pq, b = p, c = q. pq | pq. Yes.
        # So if pq in S, we cannot have both p and q.
        
        # General rule:
        # For any a in S, and any b, c in S \ {a}, a \nmid bc.
        
        # Let's just implement the direct check for small sets, 
        # or use the prime-blocking heuristic for speed.
        
        # For the local search, we can use a faster check:
        # 1. For each prime p in S, ensure no other element in S is divisible by p.
        # 2. For each composite a in S, check against all pairs b, c.
        
        # Actually, if we stick to the strategy of swapping p->p^2, 
        # we remove p, so the "prime blocking" is gone.
        # Then we can add multiples of p?
        # If p is not in S, we can have multiples of p.
        # But if we have p^2, and b=p, we have conflict.
        # So if we have p^2, we cannot have p.
        
        # Let's write a robust validity checker.
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j: continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # The O(N^3) check is too slow for N=54+ in a loop.
    # We need a faster check or a smarter search.
    
    # Heuristic: 
    # If we maintain the property that "If p in S, no other multiple of p", 
    # and for composites we check carefully, it might work.
    
    # Let's try a different approach:
    # Start with primes.
    # Try to swap p -> p^2 for p in {2,3,5,7,11,13}.
    # Then try to add composites that are multiples of swapped primes.
    
    # Since the O(N^3) check is expensive, I will limit the search space.
    
    # Let's try to find a specific configuration.
    # Known good configuration from literature/hints:
    # Swap 2->4, 3->9, 5->25.
    # Add 6, 10, 15?
    # 6=2*3. 10=2*5. 15=3*5.
    # If we have 4, 9, 25.
    # 4, 9, 25 are in S.
    # 6, 10, 15 are in S.
    # Primes 7, 11, ... are in S.
    
    # Check 4, 6, 10.
    # 4 | 6*10 = 60. Yes. Invalid.
    # So we can't have 4, 6, 10.
    
    # This trial and error is risky.
    # I will implement a simple local search with the O(N^3) check 
    # but only on the changed elements to speed it up?
    # Or just accept that N is small (~60) and N^3 is 216k ops.
    # 216k * 1000 iterations = 216M ops. Might be too slow in Python.
    
    # Optimization:
    # Precompute divisibility?
    
    # Given the risk of timeout, I will return the prime set if search fails or takes too long.
    
    # Let's try a few random swaps and additions.
    
    random.seed(42)
    
    # List of potential swaps: p -> p^2
    swap_candidates = [2, 3, 5, 7, 11, 13]
    
    # List of potential additions: composites
    add_candidates = [x for x in range(2, 257) if x not in current_set]
    
    # Simple hill climbing
    for _ in range(500): # 500 iterations
        if time.time() > deadline - 5:
            break
        
        # Try a random swap
        if swap_candidates and random.random() < 0.5:
            p = random.choice(swap_candidates)
            if p in current_set and p*p <= 256:
                new_set = current_set.copy()
                new_set.remove(p)
                new_set.add(p*p)
                if len(new_set) > len(best_set) or (len(new_set) == len(best_set) and random.random() < 0.1):
                    # Check validity
                    if is_valid_set(new_set):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(best_set)
        
        # Try a random addition
        if add_candidates and random.random() < 0.5:
            c = random.choice(add_candidates)
            new_set = current_set.copy()
            new_set.add(c)
            if len(new_set) > len(best_set):
                if is_valid_set(new_set):
                    current_set = new_set
                    best_set = current_set.copy()
                    best_size = len(best_set)
                    # Remove c from candidates? No, might be useful later with swaps
                    # But if it failed, it's still in candidates.
    
    # Ensure best_set is valid
    if not is_valid_set(best_set):
        return list(primes)
        
    return list(best_set)
