# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden for the difference check
    # Condition: (a-b) not in F and (b-a) not in F.
    # Let d = (a-b) % m. Then (b-a) % m = (-d) % m.
    # So we need d not in F and (-d)%m not in F.
    # This is equivalent to d not in (F union -F).
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
        
    # Remove 0 just in case, though it shouldn't be in F
    forbidden_sym.discard(0)
    
    def is_valid_addition(S, x):
        """Check if x can be added to set S"""
        for s in S:
            d = (x - s) % m
            if d in forbidden_sym:
                return False
        return True

    def greedy_construct(order):
        """Construct a set greedily based on the given order of candidates"""
        S = []
        S_set = set()
        for x in order:
            if is_valid_addition(S, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def local_search(S, S_set):
        """Improve the set using local search (Add, then Remove+Add cycles)"""
        improved = True
        while improved:
            improved = False
            # Try to add any element
            added = False
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        added = True
            # If no addition was possible, try removing one element and re-filling
            if not added:
                # Try removing each element
                for i in range(len(S)):
                    u = S[i]
                    # Remove u
                    S.remove(u)
                    S_set.remove(u)
                    
                    # Now try to add elements again greedily
                    # We can just run a greedy fill on the remaining candidates
                    # But to be efficient, we can just try adding any x not in S_set
                    # Since the set is small, we can just iterate through all x
                    any_added = False
                    for x in range(m):
                        if x not in S_set:
                            if is_valid_addition(S, x):
                                S.append(x)
                                S_set.add(x)
                                any_added = True
                    
                    # If we ended up with a larger set, keep it
                    # Note: S length changed. We removed 1, added k.
                    # If len(S) > original_len, it's an improvement.
                    # But we modified S in place.
                    # We need to track the original length
                    pass # This logic is getting messy with in-place modification. 
                    # Let's restart local search with a cleaner approach.
        
        # The above loop has a flaw: if we remove u and add k elements, 
        # we might end up with a set that is valid but we don't easily revert 
        # if it's not better. Also, the "improved" flag logic is weak.
        
        # Let's implement a simpler, safer local search:
        # 1. Try all additions.
        # 2. If stuck, try all swaps (remove u, add v) that maintain size? No, size must increase.
        # 3. Try removing u, then greedily adding. If final size > current size, accept.
        
        # Let's rewrite local_search properly.
        pass

    # Better Local Search
    def local_search_robust(S, S_set):
        current_size = len(S)
        improved = True
        while improved:
            improved = False
            # Phase 1: Add any possible element
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S, x):
                        S.append(x)
                        S_set.add(x)
                        current_size += 1
                        improved = True
            
            # Phase 2: If no additions, try removing one element and refilling
            if not improved:
                # Try removing each element u
                for u in list(S): # Iterate over a copy
                    # Temporarily remove u
                    S.remove(u)
                    S_set.remove(u)
                    
                    # Refill greedily
                    # We can just try to add any x not in S_set
                    # To be efficient, we can just call a helper that adds all possible
                    # But since we are in a loop, let's just do a quick pass
                    temp_size = len(S)
                    
                    # Try adding any x
                    for x in range(m):
                        if x not in S_set:
                            if is_valid_addition(S, x):
                                S.append(x)
                                S_set.add(x)
                                temp_size += 1
                                improved = True # We found a better configuration candidate
                                
                    # If the new size is larger, we keep it (S is already modified)
                    # If not, we need to revert? 
                    # Actually, if temp_size <= current_size, we should revert to avoid wasting time 
                    # or getting stuck in a loop of same-size swaps.
                    # But wait, if we don't revert, we might have lost u and gained nothing, 
                    # so size is current_size - 1. That's bad.
                    # So we must revert if no improvement.
                    
                    if temp_size <= current_size:
                        # Revert: remove the newly added ones, put u back
                        # This is complex to track.
                        # Simpler: Keep a copy of S and S_set before removal?
                        # Since m is small, copying is cheap.
                        pass

        return S, S_set

    # Let's use a simpler, more robust strategy:
    # Randomized Greedy + Local Search (Add only, then Remove+Add if stuck)
    # To handle reversion safely, we will work with copies during the "try remove" phase.
    
    def local_search_v2(S, S_set):
        improved = True
        while improved:
            improved = False
            # Try to add
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
            
            # If no add, try remove + add
            if not improved:
                # Try removing each element
                for u in list(S):
                    # Copy current state
                    S_copy = list(S)
                    S_set_copy = set(S_set)
                    
                    # Remove u
                    S_copy.remove(u)
                    S_set_copy.remove(u)
                    
                    # Try to add any x
                    added_any = False
                    for x in range(m):
                        if x not in S_set_copy:
                            if is_valid_addition(S_copy, x):
                                S_copy.append(x)
                                S_set_copy.add(x)
                                added_any = True
                                
                    # If the new set is strictly larger, accept it
                    if len(S_copy) > len(S):
                        S = S_copy
                        S_set = S_set_copy
                        improved = True
                        
        return S, S_set

    best_set = []
    best_set_set = set()
    
    deadline = time.time() + 500 # 500s safety margin
    
    # Number of restarts
    # Estimate: Local search is fast. 
    # Let's do 1000 restarts if time permits.
    
    for i in range(2000):
        if time.time() > deadline:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        S, S_set = greedy_construct(order)
        S, S_set = local_search_v2(S, S_set)
        
        if len(S) > len(best_set):
            best_set = S
            best_set_set = S_set
            # Optional: early exit if we hit a known high bound? 
            # Max possible size is bounded by m / (1 + avg forbidden density)?
            # For now, just keep searching.
            
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
