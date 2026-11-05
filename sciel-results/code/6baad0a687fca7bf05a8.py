import time
from fractions import Fraction

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Greedy initialization: add elements in order of decreasing 1/x (i.e., increasing x)
       if they don't create a 3-AP. This is a natural heuristic since 1/x is convex and
       decreasing, so smaller numbers are more valuable.
       
    Actually, wait. The seed program goes from 1 to 256 (increasing order), adding each
    if valid. Since 1/x is decreasing, going from 1 to 256 means we prioritize the
    most valuable elements first. That's a good greedy.
    
    2. Local improvement: try removing one element and re-adding as many as possible
       in order of decreasing value to see if we get a better sum.
       
    3. Simulated annealing or hill climbing with swaps.
    
    Let's think about this more carefully. The greedy approach of adding 1, 2, 3, ...
    in order is good because it prioritizes high-value elements. But it might get stuck
    in a local optimum.
    
    A better approach might be:
    1. Start with greedy (add in order 1, 2, ..., 256 if valid)
    2. Then do local search: for each element in the set, try removing it and see if
       we can add back more/better elements.
    
    Actually, since the constraint is that we can't have 3-APs, and we want to maximize
    sum of 1/x, the key insight is that small numbers are very valuable. So we want to
    keep as many small numbers as possible.
    
    Let me think about the structure of 3-AP-free sets. For [1, n], the largest 3-AP-free
    set has size roughly n / (log n)^c for some c. But we want to maximize reciprocal sum,
    not size.
    
    A better approach:
    1. Greedy: add elements in order 1, 2, ..., 256, adding x if it doesn't form a 3-AP
       with any two existing elements.
    2. Local search: try swapping - for each pair (a in set, b not in set) with 1/a > 1/b
       (i.e., a < b), try removing a and adding b, then greedily fill in more elements
       if possible.
    
    Actually, let me try a different approach. Since n=256 is small enough, I can use
    a more sophisticated greedy with look-ahead, or use a branch and bound with good
    pruning.
    
    Let me try:
    1. Greedy initialization (add 1, 2, ..., 256 in order if valid)
    2. Try to improve by removing one element and re-optimizing the rest
    
    For the re-optimization after removing element r:
    - Start with the current set minus r
    - Try to add back elements (including r and others) in order of decreasing 1/x
    
    Actually, here's a cleaner approach:
    1. Greedy: add 1, 2, ..., 256 in order if valid
    2. For each element x in the set (in order of decreasing 1/x, i.e., increasing x):
       - Remove x
       - Try to add back all elements (including x) in order of decreasing 1/x
       - If the result is better, keep it
    
    Let me implement this.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 60  # Leave safety margin
    
    def is_3ap_free(lst):
        """Check if a list has no non-constant 3-term AP."""
        s = set(lst)
        for i, a in enumerate(lst):
            for b in lst[i+1:]:
                # Check if (a+b)/2 is in s and != a and != b
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if mid != a and mid != b and mid in s:
                        return False
        return True
    
    def can_add(x, s_set):
        """Check if x can be added to s_set without creating a 3-AP."""
        for a in s_set:
            if a == x:
                continue
            # Case 1: a, x, b => b = 2x - a
            b1 = 2 * x - a
            if b1 != x and b1 in s_set:
                return False
            # Case 2: a, b, x => b = (a + x) / 2
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 != x and b2 != a and b2 in s_set:
                    return False
            # Case 3: b, a, x => b = 2a - x
            b3 = 2 * a - x
            if b3 != x and b3 != a and b3 in s_set:
                return False
        return True
    
    def reciprocal_sum(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total
    
    # Step 1: Greedy initialization - add elements in order 1, 2, ..., 256
    selected_set = set()
    for x in range(1, n + 1):
        if can_add(x, selected_set):
            selected_set.add(x)
    
    current_sum = reciprocal_sum(selected_set)
    
    # Step 2: Local improvement
    # Try removing each element and re-optimizing
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() > deadline:
            break
        
        current_set = set(selected_set)
        current_sum = reciprocal_sum(current_set)
        
        # Try removing each element
        for rem in sorted(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {rem}
            
            # Try to add back elements in order of decreasing 1/x (increasing x)
            # But we want to prioritize small numbers, so add in order 1, 2, ..., 256
            # Actually, we should add in order of decreasing value, which is increasing x
            # Wait, 1/x is decreasing, so smaller x has larger 1/x. So we want to add
            # small x first.
            
            new_set = set(trial_set)
            # Try to add all elements that are not in new_set, in order of decreasing 1/x
            # i.e., increasing x
            for cand in range(1, n + 1):
                if cand in new_set:
                    continue
                if can_add(cand, new_set):
                    new_set.add(cand)
            
            new_sum = reciprocal_sum(new_set)
            if new_sum > current_sum:
                selected_set = new_set
                improved = True
                break
    
    # Step 3: Try swap-based improvements
    # For each element a in the set and b not in the set with b > a (so 1/b < 1/a),
    # try removing a and adding b, then re-optimizing.
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(selected_set)
            current_sum = reciprocal_sum(current_set)
            
            # Try swaps: remove a, add b where b > a
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n + 1):
                    if b in current_set:
                        continue
                    # Try swapping: remove a, add b
                    trial_set = current_set - {a}
                    if can_add(b, trial_set):
                        trial_set.add(b)
                        # Now try to add more elements
                        new_set = set(trial_set)
                        for cand in range(1, n + 1):
                            if cand in new_set:
                                continue
                            if can_add(cand, new_set):
                                new_set.add(cand)
                        
                        new_sum = reciprocal_sum(new_set)
                        if new_sum > current_sum:
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
    
    result = sorted(selected_set)
    assert len(result) > 0, "Set must be nonempty"
    assert is_3ap_free(result), "Set must be 3-AP-free"
    
    return result

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 60
    
    def can_add(x, s_set):
        """Check if x can be added to s_set without creating a 3-AP."""
        for a in s_set:
            if a == x:
                continue
            b1 = 2 * x - a
            if b1 != x and b1 in s_set:
                return False
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 != x and b2 != a and b2 in s_set:
                    return False
            b3 = 2 * a - x
            if b3 != x and b3 != a and b3 in s_set:
                return False
        return True
    
    def reciprocal_sum(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total
    
    def is_3ap_free(lst):
        s = set(lst)
        for i, a in enumerate(lst):
            for b in lst[i+1:]:
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if mid != a and mid != b and mid in s:
                        return False
        return True
    
    # Greedy initialization
    selected_set = set()
    for x in range(1, n + 1):
        if can_add(x, selected_set):
            selected_set.add(x)
    
    current_sum = reciprocal_sum(selected_set)
    
    # Local improvement: remove one element, re-optimize
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() > deadline:
            break
        
        current_set = set(selected_set)
        current_sum = reciprocal_sum(current_set)
        
        for rem in sorted(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {rem}
            new_set = set(trial_set)
            
            for cand in range(1, n + 1):
                if cand in new_set:
                    continue
                if can_add(cand, new_set):
                    new_set.add(cand)
            
            new_sum = reciprocal_sum(new_set)
            if new_sum > current_sum:
                selected_set = new_set
                improved = True
                break
    
    # Swap-based improvements
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(selected_set)
            current_sum = reciprocal_sum(current_set)
            
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n + 1):
                    if b in current_set:
                        continue
                    trial_set = current_set - {a}
                    if can_add(b, trial_set):
                        trial_set.add(b)
                        new_set = set(trial_set)
                        for cand in range(1, n + 1):
                            if cand in new_set:
                                continue
                            if can_add(cand, new_set):
                                new_set.add(cand)
                        
                        new_sum = reciprocal_sum(new_set)
                        if new_sum > current_sum:
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
    
    result = sorted(selected_set)
    assert len(result) > 0
    assert is_3ap_free(result)
    
    return result
# EVOLVE-BLOCK-END
