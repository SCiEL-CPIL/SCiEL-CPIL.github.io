import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # Candidates: 1..256
    # Exclude 1 if we aim for size > 1
    # Let's try to find the max size
    
    best_set = []
    
    # Helper to check validity of a set
    def is_valid_set(elements):
        if not elements:
            return True
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            if x in sums or x in new_sums:
                return False
            new_sums.add(x)
            
            # Check divisibility
            # We only need to check interactions involving new_sums
            # But to be safe and simple, we can check all pairs in the final set?
            # No, that's slow.
            # Check new_sums against (sums U new_sums)
            
            all_sums = sums.union(new_sums)
            # Sort all_sums for faster checks?
            # For each s in new_sums, check against all_sums
            for s in new_sums:
                for t in all_sums:
                    if s == t:
                        continue
                    if s > t:
                        if s % t == 0:
                            return False
                    else:
                        if t % s == 0:
                            return False
            sums = all_sums
        return True

    # DFS to find a set of size K
    def dfs(current_elements, current_sums, remaining_candidates, target_size, found):
        if time.time() > deadline:
            return False
        
        if len(current_elements) == target_size:
            found.append(list(current_elements))
            return True
        
        # Pruning: if not enough candidates left
        if len(remaining_candidates) < target_size - len(current_elements):
            return False
            
        for i in range(len(remaining_candidates)):
            x = remaining_candidates[i]
            
            # Try adding x
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + x
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                if x in current_sums or x in new_sums:
                    valid = False
                else:
                    new_sums.add(x)
                    # Check divisibility
                    all_sums = current_sums.union(new_sums)
                    for s in new_sums:
                        for t in all_sums:
                            if s == t:
                                continue
                            if s > t:
                                if s % t == 0:
                                    valid = False
                                    break
                            else:
                                if t % s == 0:
                                    valid = False
                                    break
                        if not valid:
                            break
            
            if valid:
                if dfs(current_elements + [x], all_sums, remaining_candidates[i+1:], target_size, found):
                    return True
            
        return False

    # Try sizes from 10 down to 1
    # We can restrict candidates to larger numbers to increase success rate
    # Let's try full range first, but sorted descending
    
    candidates = list(range(256, 0, -1))
    
    # Optimization: Randomize candidates for different runs?
    # For now, deterministic descending.
    
    found = []
    
    # Try to find size 10
    for k in range(10, 0, -1):
        if time.time() > deadline:
            break
        found = []
        # Use a subset of candidates? 
        # If we use all 256, the branching factor is huge.
        # We should limit the branching factor.
        # Let's pick a random subset of candidates of size 50?
        # Or just take the top 100 largest?
        
        # Let's try with the top 100 largest numbers first, as they are "safer"
        # Actually, mixing might be needed.
        # Let's try a few random permutations of the top 100 candidates.
        
        top_candidates = candidates[:100]
        
        attempts = 0
        while time.time() < deadline and not found:
            random.shuffle(top_candidates)
            # Limit branching: only consider first 20 candidates at each step?
            # This is a heuristic.
            
            # Let's just run DFS on the shuffled list, but cut off if too deep?
            # The DFS above iterates through ALL remaining candidates.
            # This is $O(N^K)$.
            # We need to restrict the candidates considered at each step.
            
            # Modified DFS: Only consider the first `branch_limit` candidates
            branch_limit = 20
            
            def dfs_limited(current_elements, current_sums, remaining_candidates, target_size, found):
                if time.time() > deadline:
                    return False
                
                if len(current_elements) == target_size:
                    found.append(list(current_elements))
                    return True
                
                if len(remaining_candidates) < target_size - len(current_elements):
                    return False
                    
                # Consider only the first `branch_limit` candidates
                limit = min(branch_limit, len(remaining_candidates))
                
                for i in range(limit):
                    x = remaining_candidates[i]
                    
                    new_sums = set()
                    valid = True
                    for s in current_sums:
                        ns = s + x
                        if ns in current_sums or ns in new_sums:
                            valid = False
                            break
                        new_sums.add(ns)
                    
                    if valid:
                        if x in current_sums or x in new_sums:
                            valid = False
                        else:
                            new_sums.add(x)
                            all_sums = current_sums.union(new_sums)
                            for s in new_sums:
                                for t in all_sums:
                                    if s == t:
                                        continue
                                    if s > t:
                                        if s % t == 0:
                                            valid = False
                                            break
                                    else:
                                        if t % s == 0:
                                            valid = False
                                            break
                                if not valid:
                                    break
                    
                    if valid:
                        if dfs_limited(current_elements + [x], all_sums, remaining_candidates[i+1:], target_size, found):
                            return True
                    
                return False
            
            dfs_limited([], set(), top_candidates, k, found)
            attempts += 1
            if attempts > 50: # Don't try too many randomizations for one k
                break
        
        if found:
            best_set = found[0]
            # Verify best_set
            if is_valid_set(best_set):
                return best_set
            else:
                # Fallback
                return [100, 101, 102]

    # If no set found of size > 1, return a small valid set
    # Check if best_set is empty
    if not best_set:
        # Try simple valid sets
        for test_set in [[100, 101], [100, 102], [100, 103], [100], [1]]:
             if is_valid_set(test_set):
                 return test_set
                 
    return [100]

# EVOLVE-BLOCK-END
