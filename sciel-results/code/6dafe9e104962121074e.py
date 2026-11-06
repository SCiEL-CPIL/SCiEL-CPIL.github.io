import random
import math
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    n = 128

    def _lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute bad pairs: for each x, list of (a,b) pairs such that {x,a,b} is a bad triple
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) != ab:
                    continue
                if _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    # Convert to lists for faster iteration
    bad_pairs_list = [list(s) for s in bad_pairs_for]
    
    # Precompute conflict counts for each element with each other element?
    # Or just use the list.
    
    def can_add(x, current_set):
        for (a, b) in bad_pairs_list[x]:
            if a in current_set and b in current_set:
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def kick(S, k):
        """
        Remove k random elements from S, then greedily refill.
        Returns the new set.
        """
        S = set(S)
        if not S:
            return S
        
        elements_to_remove = random.sample(list(S), k)
        for x in elements_to_remove:
            S.remove(x)
        
        # Greedy refill
        # Try to add elements that were removed first? Or just any missing element?
        # Let's try to add any missing element that fits.
        # To be efficient, we can iterate through all 1..n and add if possible.
        # Since we just removed k elements, the set is "loose".
        
        # Optimization: prioritize elements that were removed?
        # Or just standard greedy on the whole range?
        # Standard greedy on the whole range is O(n * deg).
        # Let's do a randomized greedy refill to introduce more diversity.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        
        return S

    def local_search(S, time_limit):
        """
        Simulated Annealing / Random Restart Local Search.
        """
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        T = 1.0
        alpha = 0.99
        T_min = 0.01
        
        # Initial state is assumed valid (from greedy or previous step)
        current_size = len(S)
        
        while time.time() < time_limit:
            # Kick
            # Choose kick size based on temperature?
            # Small kicks for fine tuning, large kicks for escape.
            if T > 0.5:
                kick_size = random.randint(2, 5)
            else:
                kick_size = random.randint(1, 3)
                
            new_S = kick(S, kick_size)
            new_size = len(new_S)
            
            delta = new_size - current_size
            
            # Acceptance criterion
            if delta > 0:
                # Always accept improvement
                S = new_S
                current_size = new_size
                if new_size > best_size:
                    best_size = new_size
                    best_S = set(S)
            else:
                # Accept worsening with probability exp(delta / T)
                # delta is negative or zero
                if random.random() < math.exp(delta / T):
                    S = new_S
                    current_size = new_size
            
            T = max(T * alpha, T_min)
            
            # Occasionally try a pure greedy refill from scratch if stuck?
            # Or just let the kicks handle it.
            
        return best_S

    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Randomized Greedy Starts
    # We want to find good initial valid sets.
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:  # Leave time for local search
            break
            
        # Generate a random order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Bias towards low conflict degree?
        # Elements with fewer bad pairs are "safer" to add early.
        # Let's try sorting by degree with noise.
        degrees = [len(bad_pairs_list[k]) for k in range(n + 1)]
        
        if i % 2 == 0:
            # Pure random
            pass
        else:
            # Sort by degree + noise
            order.sort(key=lambda x: degrees[x] + random.random() * 10)
            
        S = greedy(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # If we found a very good set, we can stop early?
        # Theoretical max is unknown, but likely around 100-110?
        # If we hit a high bound, we can focus on local search.
        
    # Phase 2: Local Search on the best set found, and maybe a few others
    # To maximize the chance, run local search on the single best set found so far.
    # Or run it on a few top sets?
    # Given time constraints, focusing on the best is usually better.
    
    # Let's keep a list of top few sets and run local search on the best one.
    # Actually, just run on the best one.
    
    if best_set:
        # Allocate remaining time to local search
        # Leave a small buffer for returning
        local_search_deadline = deadline - 2
        best_set = local_search(best_set, local_search_deadline)
        
        if len(best_set) > best_size:
            best_size = len(best_set)
            best_set = best_set

    # Final check: ensure it's not empty
    if not best_set:
        # Fallback: simple greedy
        S = greedy(list(range(1, n + 1)))
        best_set = S

    return list(best_set)
