import time
import sys
from itertools import combinations_with_replacement

# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # 1. Quick Greedy to get a baseline
    def greedy_baseline():
        selected = []
        sums = set()
        # We can speed up greedy by only checking new sums
        for x in range(1, 4097):
            if time.time() > deadline - 10.0:
                break
            # Generate new sums involving x
            new_sums = []
            valid = True
            # Triples with x
            # (x,x,x), (x,x,y), (x,y,y), (x,y,z) where y,z in selected
            # Actually, combinations_with_replacement(selected + [x], 3) is easier but slower.
            # Let's stick to the fast check:
            # New sums are those containing at least one x.
            # 1. x+x+x
            s1 = 3 * x
            if s1 in sums:
                valid = False
            else:
                new_sums.append(s1)
            
            if valid:
                for y in selected:
                    # 2. x+x+y
                    s2 = 2 * x + y
                    if s2 in sums or s2 in new_sums:
                        valid = False
                        break
                    new_sums.append(s2)
            
            if valid:
                for i in range(len(selected)):
                    for j in range(i, len(selected)):
                        y = selected[i]
                        z = selected[j]
                        # 3. x+y+z
                        s3 = x + y + z
                        if s3 in sums or s3 in new_sums:
                            valid = False
                            break
                        new_sums.append(s3)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
                if len(selected) >= 40: # Hard bound
                    break
        return selected

    best_set = greedy_baseline()
    if not best_set:
        best_set = [1] # Fallback
    
    current_size = len(best_set)
    
    # 2. Backtracking search to improve
    # We try to find sets of size current_size + 1, current_size + 2, etc.
    
    def check_candidate(selected, sums, x):
        """
        Returns list of new sums if x is valid, else None.
        selected: list of ints
        sums: set of all triple sums of selected
        x: candidate int
        """
        new_sums = []
        # 1. x+x+x
        s = 3 * x
        if s in sums:
            return None
        new_sums.append(s)
        
        # 2. x+x+y
        for y in selected:
            s = 2 * x + y
            if s in sums or s in new_sums:
                return None
            new_sums.append(s)
        
        # 3. x+y+z
        # y,z from selected with replacement
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                y = selected[i]
                z = selected[j]
                s = x + y + z
                if s in sums or s in new_sums:
                    return None
                new_sums.append(s)
                
        return new_sums

    def dfs(path, sums, start_val, target_size):
        nonlocal best_set, current_size, deadline
        
        if time.time() > deadline:
            return False
        
        if len(path) == target_size:
            # Found a valid set of target_size
            best_set = path[:]
            current_size = target_size
            return True
            
        # Heuristic: Limit the search space
        # Max element for B3 set of size k is roughly k^3.
        # We are building up to target_size.
        # Current path size is k.
        # Next element should not be too large to allow room for remaining elements?
        # Actually, we just need distinct sums.
        # But if x is too large, it might not block anything, but it wastes the "small sum" space?
        # No, large sums are fine as long as they don't collide.
        # However, to find a set of size T, the elements must be small enough.
        # Upper bound for max element in a B3 set of size T is approx T^3.
        # So we can cap the search range.
        max_candidate = min(4096, target_size ** 3 + 100)
        
        for x in range(start_val, max_candidate + 1):
            if time.time() > deadline:
                return False
                
            new_sums = check_candidate(path, sums, x)
            if new_sums is not None:
                # Add x
                path.append(x)
                sums.update(new_sums)
                
                # Recurse
                if dfs(path, sums, x + 1, target_size):
                    # Clean up before returning success (optional if we exit immediately, but good practice)
                    path.pop()
                    sums.difference_update(new_sums)
                    return True
                
                # Backtrack
                path.pop()
                sums.difference_update(new_sums)
                
        return False

    # Iterative deepening on size
    # We know best_set has size current_size.
    # Try to find current_size + 1.
    # To make it faster, we can initialize the search with the best_set as a prefix?
    # No, the best_set might not be a prefix of a larger set.
    # But we can try to extend the greedy set first.
    
    # Let's try extending the current best_set
    # This is a heuristic: assume the greedy prefix is good.
    # If it fails, we do a full search from scratch?
    # Full search from scratch is expensive.
    
    # Strategy: 
    # 1. Try to extend best_set.
    # 2. If that fails, try a full search for target_size = current_size + 1 with a smaller time limit.
    
    # For simplicity and robustness, let's just run a full search for the next size.
    # But full search for size 10+ is huge.
    # We need to prune better.
    
    # Improved DFS:
    # Use the greedy solution as a guide.
    # When choosing the next element, prioritize the greedy choice first.
    
    def dfs_guided(path, sums, start_val, target_size, greedy_next):
        nonlocal best_set, current_size, deadline
        
        if time.time() > deadline:
            return False
        
        if len(path) == target_size:
            best_set = path[:]
            current_size = target_size
            return True
            
        max_candidate = min(4096, target_size ** 3 + 100)
        
        # Create a list of candidates, putting greedy_next first if valid
        # Actually, we can just iterate, but we know the greedy choice is likely valid.
        # Let's just iterate from start_val.
        
        for x in range(start_val, max_candidate + 1):
            if time.time() > deadline:
                return False
                
            new_sums = check_candidate(path, sums, x)
            if new_sums is not None:
                path.append(x)
                sums.update(new_sums)
                
                if dfs_guided(path, sums, x + 1, target_size, None):
                    path.pop()
                    sums.difference_update(new_sums)
                    return True
                
                path.pop()
                sums.difference_update(new_sums)
                
        return False

    # Try to improve
    target_size = current_size + 1
    while target_size <= 40:
        if time.time() > deadline - 5.0:
            break
            
        # Attempt to find a set of this size
        # We start with an empty path to allow for completely different structures
        # But we can also try to extend the current best.
        # Let's try extending first. It's cheaper.
        
        # Check if we can extend best_set
        # This is just one specific branch.
        
        # General search
        found = dfs_guided([], set(), 1, target_size, None)
        
        if found:
            target_size += 1
        else:
            break
            
    return best_set
# EVOLVE-BLOCK-END
