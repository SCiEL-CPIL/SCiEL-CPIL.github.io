import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit
    
    def get_covered_differences(S):
        """Return the set of d > 0 such that a, a+d, a+2d all in S for some a."""
        S_set = S
        d_set = set()
        for a in S_set:
            for b in S_set:
                if b <= a:
                    continue
                d = b - a
                c = b + d
                if c in S_set:
                    d_set.add(d)
        return d_set

    def get_score(S):
        return len(get_covered_differences(S))

    def greedy_build():
        """Greedy maximum coverage: build a 32-element set maximizing covered differences."""
        # Start with a small seed
        current = set()
        
        # Precompute all possible 3-APs: (a, a+d, a+2d) with 1 <= a, a+2d <= 512
        # For each element x, track which differences it could help cover
        # Actually, let's think differently:
        # For a set S, d is covered if there exists a in S such that a+d in S and a+2d in S.
        # When we add element x to S, which new d's get covered?
        # d is newly covered if:
        #   - There exists a in S (before adding x) such that a+d in S and a+2d == x
        #   - Or there exists b in S (before adding x) such that b+d == x and b+2d in S (i.e., x is the middle)
        #   - Or there exists c in S (before adding x) such that c == x-2d, c+d == x-d, and x-d in S
        # This is getting complex. Let me use a simpler approach.
        
        # Simpler: just greedily add the element that maximizes the new covered differences.
        # For efficiency, track for each potential element x, what d's it would complete.
        
        # Actually, let's just do a direct greedy: at each step, try adding each candidate
        # and pick the one that adds the most new covered d's.
        
        # To make this tractable, limit candidates to values near existing elements.
        
        # Start with a few random elements
        for _ in range(5):
            x = random.randint(1, 512)
            current.add(x)
        
        while len(current) < 32:
            best_x = None
            best_new_d = -1
            best_score = get_score(current)
            
            # Generate candidates: random + neighbors of existing elements
            candidates = set()
            for x in current:
                for delta in range(-20, 21):
                    nx = x + delta
                    if 1 <= nx <= 512 and nx not in current:
                        candidates.add(nx)
            # Add some random candidates
            for _ in range(50):
                nx = random.randint(1, 512)
                if nx not in current:
                    candidates.add(nx)
            
            for x in candidates:
                if x in current:
                    continue
                new_set = set(current)
                new_set.add(x)
                new_score = get_score(new_set)
                if new_score > best_new_d:
                    best_new_d = new_score
                    best_x = x
            
            if best_x is not None:
                current.add(best_x)
            else:
                # Fallback: add random
                x = random.randint(1, 512)
                while x in current:
                    x = random.randint(1, 512)
                current.add(x)
        
        return current

    def targeted_local_search(S, time_limit):
        """Local search that specifically targets missing differences."""
        S_set = set(S)
        current_score = get_score(S_set)
        end_time = start_time + time_limit
        
        while time.time() < end_time:
            covered_d = get_covered_differences(S_set)
            # Find missing d's: d > 0 such that no a, a+d, a+2d all in S
            # We want to find a swap that covers a missing d.
            
            # For each missing d, find if we can complete a 3-AP by swapping one element.
            # A 3-AP with difference d is (a, a+d, a+2d).
            # If we have two of these, we can add the third by removing some other element.
            
            # Collect all partial 3-APs: pairs (a, a+d) or (a+d, a+2d) or (a, a+2d)
            # where one element is missing.
            
            # For efficiency, iterate over small d first (more likely to be completable)
            found_improvement = False
            
            # Try d from 1 to 256 (max possible d for 3-AP in [1,512] is 255)
            for d in range(1, 256):
                if d in covered_d:
                    continue
                
                # Find all pairs (a, a+d) or (a+d, a+2d) in S
                # For each such pair, the missing element is a+2d or a respectively.
                # Check if adding the missing element and removing some other element improves score.
                
                # Find all a such that a in S and a+d in S, but a+2d not in S
                for a in S_set:
                    if a + d in S_set and a + 2 * d not in S_set and a + 2 * d <= 512:
                        # Missing element is a + 2*d
                        missing = a + 2 * d
                        # Try removing each element and adding missing
                        for elem_to_remove in list(S_set):
                            if elem_to_remove == missing:
                                continue
                            new_set = set(S_set)
                            new_set.remove(elem_to_remove)
                            new_set.add(missing)
                            new_score = get_score(new_set)
                            if new_score > current_score:
                                S_set = new_set
                                current_score = new_score
                                found_improvement = True
                                break
                    elif a + 2 * d in S_set and a + d not in S_set and a + d >= 1:
                        # Missing element is a + d
                        missing = a + d
                        for elem_to_remove in list(S_set):
                            if elem_to_remove == missing:
                                continue
                            new_set = set(S_set)
                            new_set.remove(elem_to_remove)
                            new_set.add(missing)
                            new_score = get_score(new_set)
                            if new_score > current_score:
                                S_set = new_set
                                current_score = new_score
                                found_improvement = True
                                break
                    elif a in S_set and a + 2 * d in S_set and a + d not in S_set:
                        # Already handled above
                        pass
                
                if found_improvement:
                    break
            
            # Also try general swaps: remove one, add another, check if score improves
            if not found_improvement:
                # Random swap to escape local optima
                elem_to_remove = random.choice(list(S_set))
                new_val = random.randint(1, 512)
                if new_val not in S_set:
                    new_set = set(S_set)
                    new_set.remove(elem_to_remove)
                    new_set.add(new_val)
                    new_score = get_score(new_set)
                    if new_score >= current_score:
                        S_set = new_set
                        current_score = new_score
        
        return S_set, current_score

    # Phase 1: Greedy build multiple times, keep best
    best_set = None
    best_score = -1
    
    # Generate several greedy builds
    num_greedy = 8
    greedy_time_per = 5.0
    
    for i in range(num_greedy):
        if time.time() >= deadline - 50.0:
            break
        g_set = greedy_build()
        g_score = get_score(g_set)
        if g_score > best_score:
            best_score = g_score
            best_set = set(g_set)
    
    # Phase 2: Targeted local search on best set
    if best_set is not None:
        search_time = min(60.0, deadline - 10.0 - time.time())
        if search_time > 5.0:
            best_set, best_score = targeted_local_search(best_set, search_time)
    
    # Phase 3: Additional random restarts with local search if time permits
    remaining_time = deadline - 10.0 - time.time()
    if remaining_time > 10.0:
        # Do a few more random restarts
        num_restarts = int(remaining_time / 15.0)
        num_restarts = max(1, min(num_restarts, 5))
        
        for r in range(num_restarts):
            if time.time() >= deadline - 10.0:
                break
            
            # Random start
            current_set = set(random.sample(range(1, 513), 32))
            current_score = get_score(current_set)
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            
            # Short targeted local search
            search_time = min(10.0, deadline - 10.0 - time.time())
            if search_time > 2.0:
                current_set, current_score = targeted_local_search(current_set, search_time)
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    return sorted(best_set)
