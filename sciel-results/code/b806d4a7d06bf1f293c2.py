"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    
    def is_valid(s):
        """
        Check if set s is valid.
        Forbidden: distinct a,b,c in s such that a+b, a+c, b+c are in s.
        Optimized check:
        For each pair a,b in s with a<b:
            If a+b in s:
                We need to ensure no c in s (c != a, c != b) exists such that 
                a+c in s and b+c in s.
                This is equivalent to: (s - {a}) intersect (s - {b}) intersect s 
                does not contain any element other than potentially a or b?
                Actually, let A = {x in s | x+a in s}
                Let B = {y in s | y+b in s}
                We need A intersect B to be empty, OR to only contain a or b (and we must check if a or b is actually valid c, but c must be distinct).
                Wait, c must be distinct from a and b.
                So if A intersect B is empty, valid.
                If A intersect B is not empty, check if all elements in the intersection are a or b.
                Note: a is in A if 2a in s. a is in B if a+b in s (true).
                b is in A if a+b in s (true). b is in B if 2b in s.
                So a is in A∩B iff 2a in s.
                b is in A∩B iff 2b in s.
                So if A∩B is a subset of {a,b}, it is valid.
        """
        s_list = list(s)
        s_set = s
        
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                ab = a + b
                if ab not in s_set:
                    continue
                
                # Check if there is a c in s, c != a, c != b, such that a+c in s and b+c in s
                # This is c in (s - a) and c in (s - b)
                # Let's compute the intersection of (s-a) and (s-b) restricted to s
                # (s-a) = {x-a | x in s, x-a in s}? No.
                # We want c in s such that c+a in s. Let SA = {c in s | c+a in s}
                # We want c in s such that c+b in s. Let SB = {c in s | c+b in s}
                # We need SA ∩ SB to be empty or subset of {a,b}.
                
                # Efficiently:
                # Iterate through c in s? That's O(N) inside O(N^2) -> O(N^3) again if not careful.
                # But we can break early.
                
                # Let's just check the intersection size.
                # SA = {c for c in s if c+a in s}
                # SB = {c for c in s if c+b in s}
                
                # To avoid O(N) set construction for every pair, we can just iterate c?
                # If N=128, N^3 is 2 million ops. 2 million is fast in Python? ~0.1s.
                # 128 choose 2 is ~8000 pairs. 8000 * 128 = 1 million iterations.
                # This is actually quite fast. Let's try the simple O(N^3) but optimized with early exit.
                
                found_invalid = False
                for c in s_set:
                    if c == a or c == b:
                        continue
                    if (a + c) in s_set and (b + c) in s_set:
                        return False
        return True

    # Better is_valid using set operations
    def is_valid_fast(s):
        s_set = s
        s_list = list(s_set)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                ab = a + b
                if ab not in s_set:
                    continue
                # Check for c
                # We need c in s_set, c != a, c != b
                # such that a+c in s_set and b+c in s_set
                # Let's find candidates for c.
                # c must be in s_set.
                # c must be in s_set - a (meaning c+a in s_set)
                # c must be in s_set - b (meaning c+b in s_set)
                
                # Constructing sets is O(N). Doing it O(N^2) times is O(N^3).
                # Is there a faster way?
                # Maybe just iterate c?
                # If the set is dense, we might find invalid c quickly.
                # If valid, we iterate all c.
                
                for c in s_set:
                    if c == a or c == b:
                        continue
                    if (a + c) in s_set and (b + c) in s_set:
                        return False
        return True

    # Start with the known valid solution: all numbers > n/2
    base_set = set(range(n//2 + 1, n + 1))
    best = list(base_set)
    best_size = len(best)
    best_set = set(base_set)
    
    # Greedy addition
    current = set(base_set)
    for x in range(1, n//2 + 1):
        if time.time() > deadline:
            break
        trial = current | {x}
        if is_valid_fast(trial):
            current = trial
    
    if len(current) > best_size:
        best = list(current)
        best_size = len(current)
        best_set = set(current)
        
    # Local Search with Random Restarts
    # We will try to improve the current best set by random perturbations
    # and greedy re-adding.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Start from a perturbation of the current best
        trial = set(best_set)
        
        # Randomly remove 1-3 elements
        if len(trial) > 3:
            to_remove = random.sample(list(trial), random.randint(1, 3))
            for x in to_remove:
                trial.discard(x)
        
        # Try to add back elements greedily
        # Shuffle candidates to break symmetry
        candidates = [x for x in range(1, n + 1) if x not in trial]
        random.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            t2 = trial | {x}
            if is_valid_fast(t2):
                trial = t2
                improved = True
                
        if len(trial) > len(best_set):
            best_set = trial
            best_size = len(trial)
            best = list(trial)
            
        # If we made progress, continue from this new best
        # If not, we might try a different perturbation strength in the next loop

    # Final check
    if is_valid_fast(set(best)):
        return sorted(best)
    else:
        # Fallback to base
        return sorted(list(base_set))
# EVOLVE-BLOCK-END
