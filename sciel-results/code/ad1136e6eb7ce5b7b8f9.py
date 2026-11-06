import time
from fractions import Fraction
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def _elapsed():
        return time.time() - start_time

    def _deadline_hit():
        return _elapsed() > deadline

    # We need a subset S of {1,...,24} such that for every x in S,
    # 1/x is NOT the sum of reciprocals of any subset of S \ {x}.
    #
    # Key insight: use exact rational arithmetic (Fraction) to avoid floating point errors.
    #
    # Strategy:
    # 1. Precompute all "conflict" relationships: for each x, find subsets T of {1,...,24}\{x}
    #    such that sum(1/y for y in T) == 1/x. These are the "forbidden" subsets.
    #    If S contains x and also contains all elements of some forbidden T, then S is invalid.
    # 2. We want to maximize |S| subject to: for every x in S, S does NOT contain any forbidden T for x.
    #    Equivalently: for every x and every forbidden T for x, if x in S then not (T subset of S).
    #    This is: (x in S) => (exists y in T: y not in S), i.e., x in S and T subset of S is forbidden.
    #
    # This is a Maximum Independent Set / Maximum Subset problem with constraints.
    # Since N=24, we can potentially use a branch-and-bound or ILP approach.
    #
    # Let's first enumerate all forbidden subsets.
    # For each x in 1..24, find all subsets T of {1..24}\{x} with sum(1/y) = 1/x.
    # The sum of reciprocals of a subset can be computed exactly with Fraction.
    #
    # Number of subsets of 23 elements is 2^23 = 8 million. For each x, that's too many to enumerate naively.
    # But we can prune: the sum of reciprocals must equal 1/x. Since all terms are positive,
    # we can do a DFS with pruning: if the current sum exceeds 1/x, stop.
    
    forbidden = [set() for _ in range(25)]  # forbidden[x] = set of frozensets T such that sum(1/y for y in T) = 1/x
    
    # For each x, find all subsets T of {1..24}\{x} with sum of reciprocals = 1/x
    for x in range(1, 25):
        target = Fraction(1, x)
        candidates = [y for y in range(1, 25) if y != x]
        candidates.sort(reverse=True)  # Larger y first (smaller reciprocal) for better pruning
        
        found = []
        
        def dfs(idx, current_sum, current_set):
            if current_sum == target:
                if current_set:  # Non-empty subset
                    found.append(frozenset(current_set))
                return
            if current_sum > target:
                return
            if idx >= len(candidates):
                return
            
            # Option 1: skip candidates[idx]
            dfs(idx + 1, current_sum, current_set)
            
            # Option 2: include candidates[idx]
            new_sum = current_sum + Fraction(1, candidates[idx])
            if new_sum <= target:
                current_set.append(candidates[idx])
                dfs(idx + 1, new_sum, current_set)
                current_set.pop()
        
        dfs(0, Fraction(0), [])
        forbidden[x] = set(found)
    
    if _deadline_hit():
        return list(range(13, 25))
    
    # Now we have for each x, the set of forbidden subsets.
    # A set S is valid iff for every x in S, no forbidden subset T for x is a subset of S.
    # Equivalently: for every x in S and every T in forbidden[x], T is not a subset of S.
    # Which means: for every x in S and every T in forbidden[x], there exists y in T such that y not in S.
    
    # We want to maximize |S|.
    # Let's use a greedy + local search approach, or a more systematic search.
    #
    # Since N=24, let's try a branch-and-bound:
    # Process elements from 24 down to 1. At each step, decide to include or exclude.
    # Prune if adding the element would violate a constraint.
    
    def is_valid_incremental(S_set, x):
        """Check if adding x to S_set (where x not in S_set) keeps validity."""
        # Check if x itself causes a violation: is there a forbidden T for x that is subset of S_set ∪ {x}?
        # Since x is being added, T subset of S_set ∪ {x} means T subset of S_set (since x not in T).
        # So check: is there T in forbidden[x] such that T subset of S_set?
        for T in forbidden[x]:
            if T.issubset(S_set):
                return False
        # Also check: does adding x cause any other element y in S_set to have a violation?
        # For each y in S_set, check if there's a forbidden T for y that is now a subset of S_set ∪ {x}.
        # T subset of S_set ∪ {x} means either T subset of S_set (already checked when y was added) 
        # or T contains x and T \ {x} subset of S_set.
        for y in S_set:
            for T in forbidden[y]:
                if x in T and (T - {x}).issubset(S_set):
                    return False
        return True
    
    # Greedy construction from largest to smallest
    S = set()
    for x in range(24, 0, -1):
        if is_valid_incremental(S, x):
            S.add(x)
    
    if _deadline_hit():
        return sorted(S)
    
    # Local search: try to add excluded elements by removing others
    excluded = set(range(1, 25)) - S
    improved = True
    while improved and not _deadline_hit():
        improved = False
        for x in list(excluded):
            # Try to add x. If it conflicts, try to remove conflicting elements.
            if is_valid_incremental(S, x):
                S.add(x)
                excluded.discard(x)
                improved = True
            else:
                # Find which elements need to be removed
                # Check conflict with x's own forbidden sets
                conflicts_for_x = set()
                for T in forbidden[x]:
                    if T.issubset(S):
                        conflicts_for_x.update(T)
                # Check conflict with other elements' forbidden sets
                for y in S:
                    for T in forbidden[y]:
                        if x in T and (T - {x}).issubset(S):
                            conflicts_for_x.update(T - {x})
                
                if not conflicts_for_x:
                    continue
                
                # Try removing each conflict element and see if we can add x
                for c in list(conflicts_for_x):
                    new_S = S - {c}
                    if is_valid_incremental(new_S, x):
                        S = new_S
                        S.add(x)
                        excluded.discard(x)
                        excluded.add(c)
                        improved = True
                        break
    
    if _deadline_hit():
        return sorted(S)
    
    # Try more aggressive local search: swap pairs
    for _ in range(50):
        if _deadline_hit():
            break
        improved = False
        for x in list(excluded):
            for y in list(S):
                new_S = (S - {y}) | {x}
                # Check validity of new_S
                valid = True
                for z in new_S:
                    for T in forbidden[z]:
                        if T.issubset(new_S):
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(new_S) > len(S):
                    S = new_S
                    excluded.discard(x)
                    excluded.add(y)
                    improved = True
                    break
            if improved or _deadline_hit():
                break
    
    return sorted(S)
