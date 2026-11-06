"""
Strategy: Combinatorial Design via CSP Backtracking.
Instead of local search on S, we formulate the problem as finding a set S 
that contains at least one arithmetic progression (AP) for each difference 
in a target set D. We use a backtracking CSP solver to construct S.
This allows us to target specific high counts directly.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _solve_csp(target_diffs, n_elements=32, max_val=512, max_steps=50000):
    """
    Attempt to construct a set S of size n_elements in 1..max_val 
    such that for every d in target_diffs, there exists an AP with diff d in S.
    
    Uses randomized backtracking.
    Variables: The elements of S (implicitly, by selecting APs).
    Actually, it's easier to view this as selecting a set of APs.
    Each AP is (a, d). We need to select a set of APs such that their union 
    has size <= n_elements and covers all d in target_diffs.
    But we can also add "free" elements if needed to fill up to n_elements?
    No, the score is just the count of distinct diffs. 
    If we find a set of APs with union size <= 32, we can pad it with random 
    elements to reach exactly 32. Padding might add more diffs, which is a bonus.
    
    So the problem reduces to: 
    Find a collection of APs {(a_i, d_i)} such that:
    1. The set of differences {d_i} includes target_diffs.
    2. The union of all elements in these APs has size <= 32.
    
    We can use a backtracking search on the differences.
    """
    
    # Pre-compute all possible APs for each target diff?
    # That's too many. 
    # Instead, we search for a set of APs.
    
    # State: 
    # - covered_diffs: set of diffs currently covered by selected APs
    # - used_elements: set of elements used by selected APs
    # - selected_aps: list of (a, d)
    
    # Heuristic: 
    # Pick an uncovered diff d.
    # Try all possible APs with difference d that fit in 1..512.
    # Prefer APs that use elements already in 'used_elements' to minimize union size.
    
    def backtrack(covered_diffs, used_elements, selected_aps, depth):
        if depth > max_steps:
            return None
        
        if not target_diffs:
            # All target diffs covered.
            # We have a valid set of APs.
            # Union size is len(used_elements).
            # If len(used_elements) <= n_elements, we can pad.
            return used_elements.copy()
            
        # Pick a diff to cover next.
        # Heuristic: Pick the one with fewest compatible APs given current used_elements?
        # Or just pick the first one.
        # Let's pick the one that is "hardest" to cover? 
        # Actually, smaller diffs have more APs. Larger diffs have fewer.
        # So pick the largest uncovered diff first (most constrained).
        remaining = [d for d in target_diffs if d not in covered_diffs]
        if not remaining:
            return used_elements.copy()
        
        d = max(remaining) # Most constrained
        
        # Generate candidate APs for this d
        # An AP is defined by its start a. a, a+d, a+2d must be in 1..512.
        # a can range from 1 to 512 - 2*d.
        
        # To speed up, we can filter candidates based on overlap with used_elements.
        # But generating all is O(512). 
        # Let's just iterate through possible starts.
        # We want to minimize the number of NEW elements added.
        
        best_next_state = None
        best_new_elements_count = float('inf')
        best_used = None
        best_aps = None
        best_covered = None
        
        # Randomize the order of starts to break symmetry
        starts = list(range(1, 512 - 2 * d + 1))
        random.shuffle(starts)
        
        for a in starts:
            ap_elements = {a, a + d, a + 2 * d}
            new_elements = ap_elements - used_elements
            new_count = len(new_elements)
            
            # If we add this AP, do we exceed the element budget?
            if len(used_elements) + new_count > n_elements:
                continue
            
            # Heuristic: Prefer APs that add fewer new elements.
            # We don't need to explore all if we find one that adds 0 or 1?
            # Not necessarily, but it's a good pruning heuristic.
            # Let's just pick the best one (min new elements) and recurse.
            # This is a greedy choice within the backtracking step.
            # To allow backtracking, we should try a few of the best ones.
            
            if new_count < best_new_elements_count:
                best_new_elements_count = new_count
                best_used = used_elements | ap_elements
                best_aps = selected_aps + [(a, d)]
                best_covered = covered_diffs | {d}
                
                # If we found an AP that adds 0 new elements, it's perfect.
                if new_count == 0:
                    break
        
        if best_used is None:
            return None
        
        # Recurse
        result = backtrack(best_covered, best_used, best_aps, depth + 1)
        return result

    # Initial state
    initial_covered = set()
    initial_used = set()
    initial_aps = []
    
    # Run multiple attempts with different random seeds/shuffles
    for _ in range(20):
        res = backtrack(initial_covered, initial_used, initial_aps, 0)
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_set = None
    best_score = -1
    
    # Baseline: Interval 1..32 has score 15 (diffs 1..15)
    base_set = list(range(1, 33))
    base_score = _count_progression_diffs(base_set)
    best_set = base_set
    best_score = base_score
    
    # Try to construct sets for increasing target diff counts
    # We target the set of diffs {1, 2, ..., K}
    # This is a strong heuristic because small diffs are "dense" and easier to pack.
    
    # What is the max K we can try?
    # Each AP uses 3 elements. If they are disjoint, 32/3 ~ 10 APs.
    # But they can share elements. 
    # A "grid" like {1, 2, 3} U {1+10, 2+10, 3+10} ... 
    # Let's just try K from 16 upwards.
    
    for K in range(16, 40):
        if time.time() > deadline - 5.0:
            break
            
        target_diffs = list(range(1, K + 1))
        
        # Try to solve CSP
        candidate_elements = _solve_csp(target_diffs)
        
        if candidate_elements is not None:
            # We have a set of elements that covers diffs 1..K.
            # Its size is <= 32.
            # Pad it with random elements from 1..512 to reach size 32.
            S = set(candidate_elements)
            while len(S) < 32:
                x = random.randint(1, 512)
                S.add(x)
            
            # Ensure exactly 32 distinct integers
            if len(S) != 32:
                # Should not happen, but just in case
                S_list = list(S)
                if len(S_list) > 32:
                    S = set(S_list[:32])
                else:
                    # Pad with 1..32 if somehow short (shouldn't happen)
                    while len(S) < 32:
                        S.add(random.randint(1, 512))
            
            score = _count_progression_diffs(S)
            
            if score > best_score:
                best_score = score
                best_set = list(S)
                
                # If we hit a high score, we might want to try to improve further
                # or break if we are very close to the theoretical max?
                # Theoretical max is bounded by the number of distinct diffs possible.
                # With 32 elements, max diffs is likely around 50-80?
                # If we get > 50, that's great.
                
                # Try to optimize the padding?
                # The padding is random. Maybe we can do a few SA steps on the padded set?
                # Let's do a quick local search on the best found set.
                
                # Local Search / SA on the current best
                current_S = set(best_set)
                T = 1.0
                for _ in range(200): # Quick SA
                    if time.time() > deadline - 5.0: break
                    
                    # Remove one, add one
                    to_remove = random.choice(list(current_S))
                    new_S = set(current_S)
                    new_S.discard(to_remove)
                    
                    to_add = random.randint(1, 512)
                    while to_add in new_S:
                        to_add = random.randint(1, 512)
                    new_S.add(to_add)
                    
                    new_score = _count_progression_diffs(new_S)
                    delta = new_score - best_score
                    
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        current_S = new_S
                        if new_score > best_score:
                            best_score = new_score
                            best_set = list(current_S)
                    
                    T *= 0.99
                    if T < 0.1: T = 0.1

    # If we didn't improve on baseline, return baseline
    if best_set is None:
        return list(range(1, 33))
        
    return best_set
# EVOLVE-BLOCK-END
