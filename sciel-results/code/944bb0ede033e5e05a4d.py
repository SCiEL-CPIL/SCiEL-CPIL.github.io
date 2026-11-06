import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
    
    # Precompute conflict bitsets
    # conflict_mask[x] is an integer where bit y is 1 if x and y form a 3-AP with some z
    # We only need to check if x and y are in the same 3-AP.
    # Two distinct numbers x, y are in a 3-AP if:
    # 1. (x+y) is even and 1 <= (x+y)//2 <= N (they are endpoints)
    # 2. 2x - y is in [1, N] and 2x - y != x, 2x - y != y (x is middle)
    # 3. 2y - x is in [1, N] and 2y - x != x, 2y - x != y (y is middle)
    
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if x and y are part of a 3-AP
            is_conflict = False
            # Case 1: x, y are endpoints
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    is_conflict = True
            # Case 2: x is middle
            if not is_conflict:
                z = 2 * x - y
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            # Case 3: y is middle
            if not is_conflict:
                z = 2 * y - x
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            
            if is_conflict:
                mask |= (1 << y)
        conflict_mask[x] = mask

    # Order elements by decreasing weight (increasing index)
    # We process 1, 2, ..., N.
    # Since weights are 1/i, smaller i has larger weight.
    # So we process in order 1 to N.
    
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[i] = sum of weights[i] + ... + weights[N]
    # This is the max possible additional score if we could add all remaining elements.
    # We will use a tighter bound: sum of weights of elements j >= i that are NOT blocked.
    # But calculating that dynamically is O(N) per node.
    # A simpler bound: sum of weights of all j >= i.
    # Let's use the simple bound first, it's often strong enough with good ordering.
    # Actually, we can precompute nothing for the bound if we just iterate, but that's slow.
    # Let's precompute the total sum of remaining weights from i to N.
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]

    best_score = Fraction(0)
    best_set = set()
    
    # Recursive DFS with Branch and Bound
    # idx: next element to consider (1-indexed)
    # current_set: set of selected elements
    # current_score: sum of weights in current_set
    # blocked_mask: bitmask of elements that cannot be selected (conflict with current_set)
    # available_mask: bitmask of elements that are still candidates (idx <= j <= N and not blocked)
    # We can just use blocked_mask and check idx.
    
    # To speed up, we'll use an iterative stack or recursion with careful state management.
    # Python recursion depth is limited, but N=256 is small enough if we don't go too deep?
    # Actually, depth is N. 256 is fine.
    
    def dfs(idx, current_set, current_score, blocked_mask):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        # Base case
        if idx > N:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper Bound
        # Max possible score = current_score + sum(weights[j] for j in [idx, N] if j not blocked)
        # We can approximate this by suffix_sum[idx].
        # If current_score + suffix_sum[idx] <= best_score, prune.
        if current_score + suffix_sum[idx] <= best_score:
            return
        
        # Option 1: Exclude idx
        # No change to blocked_mask or current_set
        dfs(idx + 1, current_set, current_score, blocked_mask)
        
        if time.time() > deadline:
            return
        
        # Option 2: Include idx (if not blocked)
        # Check if idx is blocked
        if not (blocked_mask & (1 << idx)):
            # Add idx
            new_set = current_set | {idx}
            new_score = current_score + weights[idx]
            
            # Update blocked_mask: OR with conflict_mask[idx]
            new_blocked_mask = blocked_mask | conflict_mask[idx]
            
            # Prune if new score + remaining bound <= best_score
            # Note: The remaining elements are idx+1 to N.
            # The bound suffix_sum[idx+1] assumes all are available.
            if new_score + suffix_sum[idx + 1] > best_score:
                dfs(idx + 1, new_set, new_score, new_blocked_mask)

    # Start DFS
    # Initial state: idx=1, empty set, score=0, no blocked
    dfs(1, set(), Fraction(0), 0)
    
    return list(best_set)
