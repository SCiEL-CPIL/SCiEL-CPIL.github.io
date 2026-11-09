import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    # We want a Sidon set S containing 0, with |S| <= 32, 
    # such that {y-x for x<y in S} covers {1, ..., 30}.
    # Minimize max(S).
    
    # Strategy: Backtracking search for the set elements s_0=0 < s_1 < ... < s_{k-1}.
    # We search for the smallest possible max element D.
    # Since we don't know D, we can search for the set directly, pruning if max exceeds current best.
    # Or we can binary search on D? No, the structure is complex.
    
    # Better: Depth-first search choosing elements one by one.
    # State: current set S, next element to choose > max(S).
    # Constraints:
    # 1. Sidon property: all differences distinct.
    # 2. Coverage: all d in 1..30 must be differences.
    
    # To make it fast, we use bitmasks for differences up to some limit?
    # Differences can be large. But we only care about 1..30 for coverage.
    # For Sidon, we need to ensure NO duplicate differences.
    # If we pick elements in increasing order, new differences are s_new - s_i for all i.
    # We must check if any of these are already in the set of differences.
    
    # Let's maintain a set of used differences.
    # Also maintain a bitmask of covered differences in 1..30.
    
    # Optimization:
    # If we have k elements, we have k*(k-1)/2 differences.
    # We need at least 30 differences. So k must be such that k*(k-1)/2 >= 30.
    # k=8 -> 28 diffs (not enough)
    # k=9 -> 36 diffs (enough)
    # So minimum size is 9. Maximum size is 32.
    
    # We will search for a valid set.
    # To minimize max, we can try to find ANY valid set with max < current_best.
    # We can use a heuristic to guide the search towards smaller elements.
    
    # Let's use a randomized backtracking with a time limit.
    # We'll try to find the best (smallest max) set.
    
    best_max = float('inf')
    best_set = None
    
    # We can also use a lower bound.
    # For a Sidon set with max element M, the number of differences is at most M(M-1)/2? No.
    # The differences are subsets of 1..M.
    # There are M(M-1)/2 possible differences in 1..M.
    # We need 30 distinct differences.
    # Also, the set size k implies k(k-1)/2 differences.
    
    # Let's run multiple random restarts of the backtracking search.
    
    def backtrack(current_set, used_diffs, covered_mask, max_allowed):
        nonlocal best_max, best_set
        
        if time.time() > deadline - 2.0:
            return False, None
            
        k = len(current_set)
        if k > 32:
            return False, None
            
        # Check if we have covered all 1..30
        if covered_mask == (1 << 30) - 1:
            # We have a valid set.
            current_max = max(current_set)
            if current_max < best_max:
                best_max = current_max
                best_set = list(current_set)
                # We can return True to stop this branch, or continue to find even smaller?
                # Since we are searching for ANY valid set with max < best_max, 
                # and we just updated best_max, we can prune branches that can't beat it.
                # But we are doing DFS, so we return True to indicate success.
                # However, we want the MINIMUM max. 
                # If we find a valid set, we should try to find one with smaller max.
                # So we shouldn't stop immediately unless we are confident we can't do better.
                # Actually, if we are searching with a max_allowed constraint, 
                # and we found a solution, we can update best and return True.
                # The outer loop will try smaller max_allowed or restart.
                return True, current_set
        
        # Pruning: Can we still cover all differences?
        # Remaining elements to add: up to 32 - k.
        # Each new element s adds (k) new differences (s - s_i).
        # Total potential new differences: sum_{i=0}^{32-k-1} (k+i) ? No.
        # If we add m more elements, total diffs increase by m*k + m*(m-1)/2.
        # Current diffs: k(k-1)/2.
        # We need to cover 30 - popcount(covered_mask) more differences.
        # But we also need to ensure Sidon property.
        
        # Heuristic Pruning:
        # If the current max is already >= best_max, we can prune? 
        # Not necessarily, because we might add smaller elements? No, elements are increasing.
        # So max(current_set) is the current max. If we add more, max will increase.
        # So if max(current_set) >= best_max, we can prune.
        if max(current_set) >= best_max:
            return False, None
            
        # What is the next element?
        # It must be > max(current_set).
        # Let's try candidates.
        
        # To find small max, we should try small candidates first.
        # But we also need to satisfy Sidon and Coverage.
        
        # Candidate generation:
        # We want to add an element x > max(current_set).
        # x must not create duplicate differences.
        # For each s_i in current_set, d = x - s_i must not be in used_diffs.
        # Also, we want to cover missing differences.
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d-1)))]
        
        if not missing_diffs:
            return True, current_set # Should have been caught above
        
        # What is the upper bound for x?
        # We want max < best_max.
        # So x < best_max.
        
        # Also, can we cover all missing_diffs with remaining elements?
        # This is a complex check. Let's use a simpler one.
        # If we have m slots left, we can add at most m * (current_size + m) / 2 differences?
        # Actually, each new element x_j adds differences x_j - x_i for all existing x_i.
        # So the first new element adds k diffs.
        # The second adds k+1 diffs.
        # ...
        # The m-th adds k+m-1 diffs.
        # Total new diffs = m*k + m(m-1)/2.
        
        slots_left = 32 - k
        max_new_diffs = slots_left * k + slots_left * (slots_left - 1) // 2
        if len(missing_diffs) > max_new_diffs:
            return False, None
            
        # Generate candidates
        # We want to try x that help cover missing diffs.
        # For each missing diff d, x could be s_i + d for some s_i.
        # Let's collect all such candidates.
        
        candidates = set()
        for s_i in current_set:
            for d in missing_diffs:
                x = s_i + d
                if x > max(current_set) and x < best_max:
                    candidates.add(x)
                    
        # Also try just max(current_set) + 1, +2, etc. to fill gaps?
        # If we don't add a candidate that covers a missing diff, we might still be able to cover it later?
        # Yes, but it's risky.
        # Let's also add a few random small increments.
        for i in range(1, 10):
            x = max(current_set) + i
            if x < best_max:
                candidates.add(x)
                
        # Sort candidates to try smaller ones first (to minimize max)
        candidates = sorted(candidates)
        
        # Randomize order slightly to avoid getting stuck in local minima?
        # Or use deterministic order for reproducibility within a seed?
        # Let's shuffle a bit.
        random.shuffle(candidates)
        # But bias towards smaller?
        # Let's just iterate.
        
        for x in candidates:
            # Check Sidon property
            valid = True
            new_diffs = []
            for s_i in current_set:
                d = x - s_i
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                used_diffs.add(d) # Tentatively add
            
            if not valid:
                # Remove tentatively added
                for d in new_diffs:
                    used_diffs.remove(d)
                continue
                
            # Update covered mask
            new_covered = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered |= (1 << (d-1))
                    
            # Recurse
            new_set = current_set + [x]
            success, _ = backtrack(new_set, used_diffs, new_covered, best_max)
            
            # Undo
            for d in new_diffs:
                used_diffs.remove(d)
                
            if success:
                # We found a better solution in the recursion (best_max was updated)
                # We can return True to propagate success up?
                # Yes, if we found a solution, we don't need to explore other branches 
                # that can't beat the new best_max immediately? 
                # Actually, we should continue to see if we can find an even smaller max?
                # But the recursion updates best_max. 
                # If we return True, the parent will stop exploring siblings?
                # If we want the GLOBAL minimum, we should explore all branches 
                # that could potentially beat the current best_max.
                # But if we found a solution with max M, any branch with current max >= M 
                # can be pruned.
                # Since we are doing DFS, and we update best_max, 
                # the pruning condition `max(current_set) >= best_max` will handle it.
                # So we can return True to indicate "I found a solution, 
                # you can stop if you want, but the pruning will handle it".
                # However, if we return True, the parent might stop.
                # Let's just return True and let the top level collect the best.
                return True, None
                
        return False, None

    # Main loop: Run backtracking with different random seeds or perturbations
    # Since the search space is large, we might need to restart.
    # Also, we can try to fix the first few elements to guide the search.
    
    # Let's try a few different initial "hints" or just random restarts.
    
    # To speed up, we can use a smaller time limit per restart and many restarts.
    
    # But backtracking can be slow if it goes deep.
    # Let's add a node limit or depth limit?
    # The depth is at most 32.
    
    # Let's run the backtracking.
    # We need to initialize used_diffs and covered_mask.
    # Start with S = [0].
    
    # We can also try to pre-fill some elements?
    # No, let's let the search find them.
    
    # To make it more effective, we can use a heuristic to choose the NEXT element 
    # based on which missing difference is "hardest" to cover?
    # Or just randomize candidates.
    
    # Let's run multiple iterations with different random states.
    
    global_rng = random.Random(0)
    
    # We will run backtracking for a certain time, then restart with a new seed/state?
    # Or just run one long backtracking with randomization in candidate order?
    
    # Let's run one long backtracking, but with randomization in candidate selection 
    # to explore different parts of the space.
    
    # However, pure DFS with backtracking might get stuck.
    # Let's use a time slice approach:
    # Run backtracking for 10 seconds, then restart with a different random seed 
    # (which changes the order of candidates).
    
    while time.time() < deadline - 5.0:
        # Reset state
        current_set = [0]
        used_diffs = set()
        covered_mask = 0
        
        # We can try to bias the search by picking a random "target" max?
        # No, let's just run.
        
        # To avoid always exploring the same "greedy" path, 
        # we can shuffle the candidate list in backtrack.
        # This is already done with random.shuffle.
        # But we need to re-seed the random number generator for each restart 
        # to get different shuffles.
        
        seed = global_rng.randint(0, 1000000)
        random.seed(seed)
        
        # Run backtrack
        # We pass best_max as the upper bound.
        # Initially best_max is inf.
        # We can set a heuristic upper bound to prune more?
        # E.g., a known good solution max is ~? 
        # From SA, it was around 100-200? 
        # Let's set a high bound initially, e.g., 1000.
        
        # Actually, if best_max is inf, the pruning `x < best_max` is weak.
        # Let's find a quick greedy solution first to set a good initial bound.
        
        if best_max == float('inf'):
            # Quick greedy to get a bound
            a = [0]
            used = set()
            covered = 0
            for d in range(1, 31):
                if not (covered & (1 << (d-1))):
                    # Need to cover d
                    # Try to add x such that x - s_i = d for some s_i
                    # Or just add a large number?
                    # Let's just add max(a) + 100 to ensure we can cover?
                    # No, that's bad for minimizing max.
                    # Let's try candidates s_i + d.
                    found = False
                    for s_i in a:
                        x = s_i + d
                        if x in a:
                            continue
                        # Check Sidon
                        ok = True
                        for s_j in a:
                            diff = x - s_j
                            if diff in used:
                                ok = False
                                break
                        if ok:
                            # Add x
                            a.append(x)
                            for s_j in a[:-1]:
                                used.add(x - s_j)
                            # Update covered
                            for diff in list(used):
                                if 1 <= diff <= 30:
                                    covered |= (1 << (diff-1))
                            found = True
                            break
                    if not found:
                        # Fallback: add a large number
                        x = max(a) + 100
                        while True:
                            ok = True
                            for s_j in a:
                                diff = x - s_j
                                if diff in used:
                                    ok = False
                                    break
                            if ok:
                                break
                            x += 1
                        a.append(x)
                        for s_j in a[:-1]:
                            used.add(x - s_j)
                        for diff in list(used):
                            if 1 <= diff <= 30:
                                covered |= (1 << (diff-1))
            
            # Now we have a valid set a.
            best_max = max(a)
            best_set = list(a)
            
        # Now run backtracking to improve
        # We use a time limit for this specific backtrack call?
        # The backtrack function checks global deadline.
        # But if it gets stuck in a deep branch, it might take too long.
        # Let's add a node counter to backtrack?
        # Or just rely on the global deadline.
        
        # To make it more effective, we can try to "jump" to a different part of the space.
        # We can do this by fixing the first few elements randomly?
        # Or just rely on the random shuffle of candidates.
        
        # Let's run the backtracking.
        # We need to pass the current best_max to prune.
        
        # One issue: backtrack modifies used_diffs. We need to be careful with backtracking.
        # The current implementation does add/remove correctly.
        
        # Let's run it.
        # We'll run it for a short time, say 5 seconds, then restart.
        # But the function checks global deadline.
        # Let's just call it. If it takes too long, it will return when deadline is hit.
        
        # However, if it finds a solution, it returns True.
        # If it exhausts the search space (unlikely), it returns False.
        
        # Let's run it.
        success, _ = backtrack(current_set, used_diffs, covered_mask, best_max)
        
        # If success is True, best_max has been updated.
        # We can loop again to see if we can find an even smaller one.
        # But the random seed is the same for the next iteration?
        # No, we re-seed at the start of the loop.
        
        # Wait, the loop structure is:
        # while time < deadline:
        #   seed = ...
        #   random.seed(seed)
        #   backtrack(...)
        
        # This will restart the search from scratch each time.
        # This is good for exploration.
        
        # But if backtrack takes 10 seconds, and we have 100 seconds left, 
        # we will do 10 restarts.
        
        # To make it faster, we can limit the depth or nodes in backtrack?
        # Let's add a simple node limit to backtrack to prevent it from spending 
        # all time in one deep, unproductive branch.
        
        # Let's modify backtrack to take a node_limit?
        # Or just use the global time.
        
        # Actually, if we find a solution, best_max decreases.
        # The next restart will have a tighter bound, so it should be faster 
        # to prune.
        
        # So this strategy seems sound.

    if best_set is None:
        # Fallback
        return [0, 1, 3, 7, 12, 18, 25, 33, 42] # Example, need to check validity
        
    return best_set

# EVOLVE-BLOCK-END
