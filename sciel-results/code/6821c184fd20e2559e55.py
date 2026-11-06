import time
import random
import itertools

def construct():
    """
    Find a large subset of [1,256] such that for every distinct a,b in the subset,
    (a*b) % (a+b) != 0.
    
    Strategy:
    1. Start with all odd numbers (128 of them). Verify no conflicts among odds.
       Note: if a and b are both odd, a+b is even. a*b is odd.
       For a*b to be divisible by a+b, a+b must divide a*b.
       Since gcd(a, a+b) = gcd(a, b), and a,b are odd, this is possible but rare.
       Actually, let's just check.
    2. Try to add even numbers one by one, checking compatibility with current set.
    3. Use local search (swap/add/remove) to maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict graph: conflict[i][j] is True if (i*j) % (i+j) == 0
    # We only care about distinct pairs.
    # To save memory and time, we can compute on the fly or use a set of conflicts.
    # Let's use a set of frozensets or just a 2D boolean array.
    # 256x256 is 65536 booleans, which is small.
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def is_valid_set(S):
        """Check if set S is valid (no conflicts)."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if conflict[a][b]:
                    return False
        return True
    
    def count_conflicts(S, x):
        """Count how many elements in S conflict with x."""
        count = 0
        for y in S:
            if y != x and conflict[x][y]:
                count += 1
        return count
    
    # Initial set: all odd numbers
    # Let's verify if all odds are mutually non-conflicting.
    odds = set(range(1, N + 1, 2))
    if not is_valid_set(odds):
        # If odds have conflicts, we need a greedy approach to build a valid set from odds
        current_set = set()
        # Sort odds by number of conflicts with other odds? Or just greedy.
        # Let's just add them if they don't conflict with current set.
        for x in sorted(odds):
            if count_conflicts(current_set, x) == 0:
                current_set.add(x)
    else:
        current_set = set(odds)
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Greedy addition of even numbers
    evens = list(range(2, N + 1, 2))
    # Sort evens by number of conflicts with current set (ascending) to add easy ones first
    # Actually, let's just try adding all evens that fit.
    for x in evens:
        if count_conflicts(current_set, x) == 0:
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(best_set)
    
    # Local Search: 
    # 1. Try to add any remaining number.
    # 2. Try to swap: remove one element, add two? Or remove one, add one if it allows adding more later?
    # Let's do a simple hill climbing:
    # - Try adding any x not in set. If it conflicts, skip.
    # - Try swapping x in set with y not in set. If removing x allows adding y AND z, do it.
    # - Simpler: Try to replace a subset of size k with a subset of size k+1.
    
    # Let's implement a more robust local search.
    # While time remains:
    #   Try to add a random element.
    #   If fails, try to remove one element that conflicts with it, then add it.
    #   If that doesn't increase size, try 2-for-1 swaps.
    
    random.seed(42)
    unselected = set(all_nums) - current_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any unselected element
        # Sort by conflicts to pick the one with fewest conflicts?
        candidates = list(unselected)
        random.shuffle(candidates)
        for x in candidates:
            if count_conflicts(current_set, x) == 0:
                current_set.add(x)
                unselected.remove(x)
                improved = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(best_size)
                break
        
        if time.time() > deadline:
            break
            
        # 2. Try 1-for-1 swap (replace x in set with y not in set)
        # This doesn't increase cardinality, but might enable future additions.
        # So we only do this if it leads to a net gain or if we are stuck.
        # Let's try to find a pair (x in S, y not in S) such that removing x and adding y
        # allows adding at least one more element z.
        
        # This is expensive. Let's try a simpler heuristic:
        # Pick a random x in S. Pick a random y not in S.
        # If conflict[x][y] is False, and removing x reduces the number of conflicts for some other element?
        
        # Alternative: Simulated Annealing style.
        # But let's stick to deterministic improvements first.
        
        # Try to add 2 elements by removing 1.
        # For each x in S:
        #   Temporarily remove x.
        #   Try to add any y in Unselected.
        #   If added, try to add any z in Unselected.
        #   If 2 added, accept.
        
        # This is O(|S| * |U|^2 * Cost). |S|~130, |U|~120. 130 * 14400 = 1.8M checks.
        # Each check is O(1) conflict lookup.
        # 1.8M is fine.
        
        # Let's do this:
        # For each x in S (random order):
        #   S_temp = S - {x}
        #   added = []
        #   For each y in Unselected (sorted by conflicts with S_temp):
        #       if count_conflicts(S_temp, y) == 0:
        #           S_temp.add(y)
        #           added.append(y)
        #           break # Only add one? No, we want to add as many as possible to see if net gain > 0
        #   Actually, let's just greedily add as many as possible from Unselected to S_temp.
        #   If len(S_temp) > len(S):
        #       S = S_temp
        #       break
        
        # Let's implement this "remove one, greedily add many" step.
        
        s_list = list(current_set)
        random.shuffle(s_list)
        
        for x in s_list:
            if time.time() > deadline:
                break
                
            S_temp = set(current_set)
            S_temp.remove(x)
            
            # Greedily add from unselected
            # Sort unselected by number of conflicts with S_temp
            u_list = list(unselected)
            # Heuristic: sort by conflicts
            u_list.sort(key=lambda y: count_conflicts(S_temp, y))
            
            for y in u_list:
                if count_conflicts(S_temp, y) == 0:
                    S_temp.add(y)
                    # Don't break, try to add more
                    # But if we add too many, it might be slow.
                    # Let's just add all that fit.
            
            if len(S_temp) > len(current_set):
                current_set = S_temp
                unselected = set(all_nums) - current_set
                improved = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(best_set)
                break # Restart the loop with new state
    
    # Final check: ensure best_set is valid
    if not is_valid_set(best_set):
        # Fallback to a known valid set: just odds
        best_set = set(range(1, N + 1, 2))
        
    return list(best_set)

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if (a*b) % (a+b) == 0:
                valid = False
                print(f"Conflict: {a}, {b}")
    print(f"Valid: {valid}")
