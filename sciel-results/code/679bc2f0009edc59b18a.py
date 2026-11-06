"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a Sidon set containing 0 with diameter M such that all differences 1..30 are present.
    Uses DFS with bitmask-based pruning and time limits.
    """
    start_time = time.time()
    MAX_TIME = 115  # seconds, leave margin for the 120s limit

    # Theoretical lower bound on diameter: need at least 30 (for diff 30)
    # Also, with n elements, we have n*(n-1)/2 differences. Need >= 30.
    # n=8 -> 28 < 30, so n >= 9.
    # For M, we need a Sidon set of size >= 9 within [0, M] covering 1..30.
    # Let's start searching from a reasonable lower bound.
    # A known small Sidon set covering 1..30 might have diameter around 40-60.
    # Let's try to find the minimal M.

    best_solution = None
    best_M = float('inf')

    # We will iterate M from a lower bound upwards.
    # Lower bound: M must be at least 30 (to have difference 30).
    # Also, the largest element is M (since 0 is in the set and max element is M, diff M must exist? No, max element is M, so diff M exists if 0 is in set. Yes, 0 is in set, so M-0 = M is a difference. But we only need 1..30. So M can be > 30.
    # However, if M < 30, we can't have difference 30. So M >= 30.
    
    # Let's try a few values of M starting from 30.
    # For each M, we check if there exists a Sidon set S subset [0, M] with 0 in S, |S| <= 32, 
    # such that {y-x : x,y in S, y>x} covers 1..30.
    
    # Since we want to minimize M, we iterate M = 30, 31, 32, ...
    # For each M, we run a DFS to find a valid set.
    
    # To speed up, we can use a heuristic: we must include 0.
    # We also need to cover difference 30, so either 30 is in S (since 0 in S, 30-0=30) or there exist x,y in S such that y-x=30.
    # If 30 is not in S, then some pair (x, x+30) is in S.
    
    # DFS state:
    # current_set: list of elements
    # used_diffs: set of differences already formed
    # missing_diffs: set of differences 1..30 not yet formed
    
    # Pruning:
    # 1. If time > MAX_TIME, abort and return best found so far (or a known valid one).
    # 2. If |current_set| > 32, abort.
    # 3. If max(current_set) > M, abort (but we only add elements <= M).
    # 4. Check if it's possible to cover all missing diffs with remaining slots.
    #    Each new element adds at most (len(current_set)) new diffs with existing elements.
    #    Total remaining diffs needed = len(missing_diffs).
    #    Max new diffs from k more elements = k * len(current_set) + k*(k-1)/2.
    #    If this is < len(missing_diffs), prune.
    
    def dfs(current_set, used_diffs, missing_diffs, M, deadline):
        nonlocal best_solution, best_M
        
        if time.time() > deadline:
            return None
        
        if not missing_diffs:
            # Found a solution
            return sorted(current_set)
        
        if len(current_set) > 32:
            return None
            
        # Pruning: can we cover missing diffs?
        # Remaining capacity
        rem_slots = 32 - len(current_set)
        # Max new diffs: each new element x adds diffs |x - y| for y in current_set, and diffs with other new elements.
        # A rough bound: each new element can add at most len(current_set) + (rem_slots - 1) diffs?
        # Actually, the total number of diffs in a set of size N is N*(N-1)/2.
        # Current diffs = len(used_diffs).
        # Max total diffs if we add rem_slots more elements = (32)*(31)/2 = 496.
        # But we are limited by M.
        # Let's use a tighter bound based on the number of missing diffs.
        # Each new element x can form diffs with existing elements. 
        # If we just add elements, the new diffs are {x-y, y-x for y in current_set} and diffs among new elements.
        # A simple bound: each new element adds at most len(current_set) diffs with old elements.
        # So rem_slots * len(current_set) + rem_slots*(rem_slots-1)/2 should be >= len(missing_diffs).
        max_new_diffs = rem_slots * len(current_set) + rem_slots * (rem_slots - 1) // 2
        if max_new_diffs < len(missing_diffs):
            return None
            
        # Determine candidate next elements to add.
        # We can add any x in 0..M not in current_set.
        # To be efficient, we should prioritize x that cover missing diffs.
        # For each missing diff d, x must be in (current_set + d) or (current_set - d).
        # So candidates are {y+d for y in current_set for d in missing_diffs if y+d <= M} 
        #                 U {y-d for y in current_set for d in missing_diffs if y-d >= 0}
        # Plus, we might need to add elements that don't immediately cover a missing diff but help later?
        # Usually, in such packing problems, it's best to cover as many missing diffs as possible.
        # Let's generate candidates that cover at least one missing diff.
        
        candidates = set()
        for y in current_set:
            for d in missing_diffs:
                x1 = y + d
                if 0 <= x1 <= M and x1 not in current_set:
                    candidates.add(x1)
                x2 = y - d
                if 0 <= x2 <= M and x2 not in current_set:
                    candidates.add(x2)
                    
        # Also, we might need to add 0? No, 0 is always in current_set (we start with [0]).
        # What if candidates is empty? Then we can't cover any missing diff with a single new element.
        # This means we are stuck, unless we could have chosen a different path.
        # So return None.
        
        if not candidates:
            return None
            
        # Sort candidates by how many missing diffs they cover (greedy heuristic)
        def count_covered(x):
            count = 0
            for y in current_set:
                d1 = x - y
                if d1 in missing_diffs:
                    count += 1
                d2 = y - x
                if d2 in missing_diffs:
                    count += 1
            # Also check diffs with other candidates? No, we are adding one at a time.
            return count
            
        sorted_candidates = sorted(candidates, key=count_covered, reverse=True)
        
        for x in sorted_candidates:
            # Check if adding x violates Sidon property
            # New diffs involving x: {x-y, y-x for y in current_set}
            new_diffs = []
            valid = True
            for y in current_set:
                d = abs(x - y)
                if d == 0:
                    continue
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Check if new_diffs have duplicates among themselves?
            # Since x is distinct from y, and y are distinct, the diffs x-y are distinct?
            # Not necessarily. e.g., x=5, y=1,3 -> 4,2. x=5, y=2,4 -> 3,1.
            # Wait, if x=5 and current_set has 1 and 9, diffs are 4 and 4. Duplicate!
            # So we must check if any d in new_diffs appears more than once.
            if len(set(new_diffs)) != len(new_diffs):
                continue
                
            # Also check if new_diffs intersect with used_diffs? Already checked.
            
            # Update state
            new_used_diffs = used_diffs.union(new_diffs)
            new_missing = missing_diffs.difference(new_diffs)
            new_set = current_set + [x]
            
            res = dfs(new_set, new_used_diffs, new_missing, M, deadline)
            if res is not None:
                return res
                
        return None

    # Start with M=30
    for M in range(30, 100):  # Search up to 100, should be enough
        # Initial state
        initial_set = [0]
        initial_used_diffs = set()
        initial_missing = set(range(1, 31))
        
        deadline = start_time + MAX_TIME
        
        # Run DFS
        # We need to be careful: the DFS might be slow for large M if not pruned well.
        # Let's add a time check inside the loop as well.
        
        sol = dfs(initial_set, initial_used_diffs, initial_missing, M, deadline)
        
        if sol is not None:
            best_solution = sol
            best_M = M
            break  # Found minimal M
            
        # If time is up, break
        if time.time() > start_time + MAX_TIME:
            break
            
    # If no solution found within time or M range, return a fallback
    # A known valid Sidon set?
    # Let's construct one greedily as fallback.
    if best_solution is None:
        a = [0]
        def get_diffs(s):
            return set(y - x for i, x in enumerate(s) for y in s[i+1:])
        for d in range(1, 31):
            if d in get_diffs(a):
                continue
            # Try to add x
            found = False
            for x in range(1, 10000):
                trial = sorted(a + [x])
                diffs = get_diffs(trial)
                if len(diffs) == len(trial)*(len(trial)-1)//2:
                    a = trial
                    found = True
                    break
            if not found:
                # Add far away
                x = 2*max(a) + d
                while True:
                    trial = sorted(a + [x])
                    diffs = get_diffs(trial)
                    if len(diffs) == len(trial)*(len(trial)-1)//2:
                        a = trial
                        break
                    x += 1
        best_solution = a

    return best_solution
# EVOLVE-BLOCK-END
