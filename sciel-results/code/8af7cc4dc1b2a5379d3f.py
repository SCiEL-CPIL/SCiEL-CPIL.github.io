import random
import numpy as np

def _score(set_list):
    """Calculate the score: number of distinct d > 0 such that a, a+d, a+2d in set."""
    S = set(set_list)
    diffs = set()
    # To optimize, we can iterate over pairs or use a more clever method.
    # For N=32, O(N^2) is fine.
    # For each pair (a, b) with a < b, let d = b - a.
    # We need a - d = a - (b-a) = 2a - b to be in S.
    # Or, for each d, check if there exists a such that a, a+d, a+2d in S.
    # Max d is (max(S) - min(S)).
    # Let's just iterate d from 1 to max_diff and check.
    # Max possible d is 511.
    for d in range(1, 512):
        found = False
        # a must be such that a + 2d <= max(S)
        # a >= min(S)
        # We can just check all a in S
        for a in S:
            if (a + d) in S and (a + 2 * d) in S:
                found = True
                break
        if found:
            diffs.add(d)
    return len(diffs)

def _improve(set_list, iterations=2000):
    """Hill climbing to improve the set."""
    current = set(set_list)
    current_score = _score(list(current))
    
    for _ in range(iterations):
        # Pick a random element to remove
        if not current:
            break
        elem = random.choice(list(current))
        current.remove(elem)
        
        # Try to add a new element
        # Candidates are numbers in 1..512 not in current
        # To be efficient, we can just pick a random candidate
        # Or, we can try all candidates? 512 * 32 * iterations might be too slow if iterations is high.
        # Let's try a few random candidates per step.
        best_new_elem = None
        best_new_score = current_score - 1 # Removing an element can only decrease or keep score, but we are replacing.
        # Actually, removing an element changes the score. Let's compute score after removal.
        score_after_remove = _score(list(current))
        
        # Try adding random candidates
        # We want to find an x such that score(current U {x}) > score_after_remove
        # Or better than current_score if we want strict improvement, but usually we just want to improve from the state after removal?
        # No, we want to improve the original state.
        # Let's just try to find an x that maximizes score(current U {x}).
        
        # Efficient way:
        # Iterate through all possible x in 1..512 not in current.
        # There are ~480 candidates.
        # _score is O(512 * 32) ~ 16k ops.
        # 480 * 16k = 7.6M ops per swap.
        # 2000 swaps = 15B ops. Too slow in Python.
        
        # Optimization:
        # We can update the score incrementally or use a faster score function.
        # Let's use a bitset or a set for O(1) lookups.
        # The _score function above is O(D * N). D is up to 511. N is 32.
        # We can precompute the set of differences for the current set?
        # If we add x, new differences involving x are:
        # x - y for y in S (if x > y)
        # y - x for y in S (if y > x)
        # But we need 3-APs.
        # A new 3-AP with x can be:
        # 1. x, x+d, x+2d. Need x+d, x+2d in S.
        # 2. x-d, x, x+d. Need x-d, x+d in S.
        # 3. x-2d, x-d, x. Need x-2d, x-d in S.
        
        # Let's implement a faster delta score calculation.
        # Given S, and candidate x.
        # Let S_new = S U {x}.
        # New d's must involve x.
        # If a 3-AP (a, a+d, a+2d) is in S_new but not in S, it must contain x.
        # Cases:
        # 1. a = x. Then x+d, x+2d in S. d > 0.
        #    Check if d was already in S's diff set.
        # 2. a+d = x. Then a = x-d, a+2d = x+d.
        #    Need x-d in S and x+d in S.
        #    d > 0.
        # 3. a+2d = x. Then a+2d = x, a+d = x-d, a = x-2d.
        #    Need x-d in S and x-2d in S.
        #    d > 0.
        
        # We need to know which d's are already present in S.
        # Let's maintain a set of valid d's for the current set.
        
        pass # This incremental approach is complex to implement correctly in a short time without bugs.
        # Given the time constraint, let's stick to a simpler heuristic:
        # Random restarts with greedy construction.
        
        # Let's just do a few random swaps and keep the best one found in a batch.
        # Or, simply return the best set found by random local search.
        
        # Let's revert to a simpler, slower but correct method for the inner loop, 
        # but reduce iterations or use a smarter candidate selection.
        
        # Actually, let's just use a pre-computed best set if we can find one offline?
        # No, we must compute it.
        
        # Let's try a different approach: 
        # Construct a set based on a known good structure.
        # "32 distinct integers in 1..512".
        # A set with many 3-AP differences often looks like a union of arithmetic progressions.
        # Example: S = {1, 2, ..., 16} U {17, 19, 21, ..., 51}?
        # Let's try to generate a few candidate structures and pick the best.
        
        break # Break out of the inefficient loop for now, use the alternative strategy below.

    return list(current)

def construct():
    # Strategy: Generate several candidate sets using different heuristics and pick the best.
    
    candidates = []
    
    # Candidate 1: The current seed (Interval + Outlier)
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 47])
    
    # Candidate 2: Two intervals
    # S = [1..16] U [33..48]
    c2 = list(range(1, 17)) + list(range(33, 49))
    candidates.append(c2)
    
    # Candidate 3: Arithmetic progression with step 2 (odd numbers) + some evens?
    # S = {1, 3, ..., 63} (32 odds).
    # 1, 3, 5, ..., 63.
    # 3-APs: 1, 5, 9 (d=4). 1, 7, 13 (d=6).
    # In a set of odds, d must be even.
    # Let a = 2i-1, b = 2j-1, c = 2k-1.
    # 2i-1, 2j-1, 2k-1 in AP => 2i-1 + 2k-1 = 2(2j-1) => 2(i+k) - 2 = 4j - 2 => i+k = 2j.
    # So i, j, k form an AP.
    # The number of distinct d's is the number of distinct 2*(j-i) = 2*d_i.
    # This is equivalent to finding distinct differences in the set of indices {1..32}.
    # Max distinct differences in {1..32} is 15 (for d=1..15).
    # So d values are 2, 4, ..., 30. Total 15.
    # This is worse than 23.
    
    # Candidate 4: Random search with local improvement
    # Let's implement a simple hill climber that is fast enough.
    # We will use the incremental score update idea, but carefully.
    
    def get_diffs(S):
        S_set = set(S)
        diffs = set()
        for d in range(1, 512):
            for a in S_set:
                if (a + d) in S_set and (a + 2 * d) in S_set:
                    diffs.add(d)
                    break
        return diffs

    def score_from_diffs(diffs):
        return len(diffs)

    # We will run a few iterations of random restarts.
    best_set = candidates[0]
    best_score = _score(best_set)
    
    for restart in range(5):
        # Initialize with a random set or a perturbed version of the best
        if restart == 0:
            current = list(candidates[0])
        else:
            # Random set
            current = random.sample(range(1, 513), 32)
        
        current_set = set(current)
        current_diffs = get_diffs(current)
        current_score = len(current_diffs)
        
        improved = True
        while improved:
            improved = False
            # Try swapping a random element with a random candidate
            for _ in range(50): # Limit swaps per pass to keep it fast
                if not current_set:
                    break
                # Pick element to remove
                elem = random.choice(list(current_set))
                current_set.remove(elem)
                
                # Find best candidate to add
                # We can't try all 511 candidates quickly with full score re-calc.
                # Let's try 10 random candidates.
                candidates_to_try = [x for x in random.sample(range(1, 513), 10) if x not in current_set]
                
                best_candidate = None
                best_candidate_score = current_score # Score of current_set (size 31) is likely lower, but we are comparing against the state BEFORE removal?
                # No, we want to find x such that score(current_set U {x}) > score(original_current_set).
                # Let's store the original score.
                original_score = _score(list(current_set) + [elem]) # This is expensive.
                
                # Let's just compare against a threshold or the previous best.
                # Simpler: Just try to improve the score of the 32-element set.
                
                # Let's re-evaluate the whole set for each candidate?
                # 10 candidates * O(512*32) = 10 * 16k = 160k ops.
                # 50 swaps * 160k = 8M ops.
                # 5 restarts * (some passes) * 8M.
                # This might be too slow in Python if we do many passes.
                
                # Let's limit the search.
                
                for x in candidates_to_try:
                    trial_set = list(current_set) + [x]
                    s = _score(trial_set)
                    if s > current_score: # current_score is the score of the 32-element set BEFORE the swap
                        # We found an improvement
                        current_set.add(x)
                        current = list(current_set)
                        current_score = s
                        improved = True
                        break # Restart the inner loop with the new state
                
                if not improved:
                    # If we didn't improve in this swap attempt, we might have left the set in a 31-element state?
                    # No, we only add if improved. If not improved, we should put elem back?
                    # Actually, the logic above is flawed. If we remove elem, we have 31 elements.
                    # If we don't find a better x, we should revert to the original state (put elem back).
                    # But if we DID find a better x, we have a new 32-element set.
                    # If we didn't, we must put elem back to maintain size 32 for the next iteration?
                    # Or we can just keep the 31-element set? No, must be 32.
                    # So if no improvement, put elem back.
                    current_set.add(elem)
                
                # If we found an improvement, current_set has size 32.
                # If not, current_set has size 32 (elem put back).
                
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                
                # Check if we should break early?
                # If we've been improving, keep going.
        
        # After the while loop, update best
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
        
        if best_score >= 25: # If we found something significantly better, stop
            break

    return best_set
