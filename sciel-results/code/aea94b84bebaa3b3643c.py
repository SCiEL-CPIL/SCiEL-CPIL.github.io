import time
import random

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy:
    1. Start with the full set [1, 100].
    2. Repeatedly find a subset summing to 200 (a 'conflict').
    3. Remove one element from the conflict based on various heuristics.
    4. Repeat until no conflict exists.
    5. Try multiple removal heuristics (remove min, max, random, etc.) and keep the best result.
    """
    deadline = time.time() + 80.0
    TARGET = 200
    N = 100
    ALL_ELEMENTS = set(range(1, N + 1))
    
    def find_conflict_subset(elements):
        """
        Find a subset of 'elements' that sums to TARGET.
        Returns the subset as a set, or None if no such subset exists.
        """
        # We need to track the elements used to form each sum.
        # DP[i] = set of elements used to form sum i? No, that's too much memory/time.
        # DP[i] = bitmask? N=100, bitmask is 100 bits. 201 sums.
        # Or DP[i] = parent sum and element?
        
        # Since N=100 is small, we can use a dictionary: sum -> (prev_sum, element)
        # But we need to distinguish between different ways to reach the same sum.
        # Actually, we just need ONE valid subset.
        
        # Standard DP for subset sum with backtracking:
        # reachable[s] = True if s is reachable.
        # parent[s] = (prev_sum, element) that led to s.
        
        reachable = [False] * (TARGET + 1)
        parent = [None] * (TARGET + 1) # parent[s] = (prev_s, x)
        reachable[0] = True
        parent[0] = (0, None) # Base case
        
        current_elements = sorted(elements)
        
        for x in current_elements:
            if x > TARGET:
                continue
            # Update backwards to avoid using x multiple times
            for s in range(TARGET - x, -1, -1):
                if reachable[s] and not reachable[s + x]:
                    reachable[s + x] = True
                    parent[s + x] = (s, x)
                    
        if not reachable[TARGET]:
            return None
            
        # Backtrack to find the subset
        subset = set()
        current_sum = TARGET
        while current_sum > 0:
            prev_sum, x = parent[current_sum]
            if x is not None:
                subset.add(x)
            current_sum = prev_sum
            
        return subset

    def solve_with_heuristic(elements, heuristic_name):
        """
        Apply the deletion heuristic until no conflicts remain.
        """
        S = set(elements)
        while True:
            if time.time() > deadline:
                return None
            conflict = find_conflict_subset(S)
            if conflict is None:
                return S
            
            if heuristic_name == 'min':
                to_remove = min(conflict)
            elif heuristic_name == 'max':
                to_remove = max(conflict)
            elif heuristic_name == 'median':
                sorted_conf = sorted(conflict)
                to_remove = sorted_conf[len(sorted_conf)//2]
            elif heuristic_name == 'random':
                to_remove = random.choice(list(conflict))
            elif heuristic_name == 'freq_max':
                # Remove element that appears in the most potential conflicts?
                # Approximation: remove the largest element (less flexible)
                to_remove = max(conflict) 
            else:
                to_remove = random.choice(list(conflict))
                
            S.remove(to_remove)
            
    # Heuristics to try
    heuristics = ['min', 'max', 'median', 'random', 'freq_max']
    
    best_set = set()
    
    # Run deterministic heuristics
    for h in ['min', 'max', 'median', 'freq_max']:
        if time.time() > deadline:
            break
        result = solve_with_heuristic(ALL_ELEMENTS, h)
        if result is not None:
            if len(result) > len(best_set):
                best_set = result
                
    # Run random heuristics multiple times
    random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
        result = solve_with_heuristic(ALL_ELEMENTS, 'random')
        if result is not None:
            if len(result) > len(best_set):
                best_set = result
                
    # Try a hybrid: Start with a greedy addition set, then try to improve by adding?
    # Or just stick to the deletion method which starts from 100.
    # The deletion method is generally superior for "avoid sum X" when starting from dense sets.
    
    # One more strategy: Local search on the result of the best heuristic.
    # Try to add back removed elements?
    # We don't know which were removed easily in the loop above, but we can reconstruct.
    # Actually, let's just do a simple local search on the best_set found.
    
    # Local Search: Try to add elements from ALL_ELEMENTS - best_set
    # If adding x creates a conflict, find the conflict C in best_set + {x}.
    # If |C| > 1, we might be able to swap?
    # This is complex. The deletion method already maximizes size for that specific path.
    # Different paths yield different sizes.
    
    # Let's try one more variation: 
    # When finding the conflict, instead of removing one element, 
    # we can try removing *different* elements from the *same* conflict in parallel branches?
    # That's essentially what the 'random' heuristic does over multiple runs.
    
    if not best_set:
        # Fallback: just [1]
        return [1]
        
    return list(best_set)
