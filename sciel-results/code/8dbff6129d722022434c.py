"""
Branch-and-Bound solver for maximum reciprocal weight 3-progression-free set.
"""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import sys

sys.setrecursionlimit(10000)

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Precompute conflicts: for each i, which j > i conflict with i?
    # Conflict exists if there is k < i such that i, j, k form a 3-AP or similar.
    # Actually, let's define the graph properly.
    # Vertices: 1..256
    # Edge (u, v) exists if u, v and some w form a 3-AP.
    # Specifically, if u < v, they conflict if:
    # 1. There exists w < u such that w, u, v is an AP => 2u = w + v => w = 2u - v. 
    #    Valid if w >= 1 and w < u.
    # 2. There exists w > v such that u, v, w is an AP => 2v = u + w => w = 2v - u.
    #    Valid if w <= N and w > v.
    # 3. There exists w between u and v? No, that would be u, w, v. 
    #    2w = u + v. If u+v is even, w = (u+v)//2. If w is an integer and u < w < v, 
    #    then u, w, v is an AP. This means u and v conflict if their midpoint is an integer.
    #    Wait, the definition of "3-progression-free set" means NO three distinct elements 
    #    form an AP. 
    #    So if we pick u and v, we cannot pick any w such that {u, v, w} is an AP.
    #    The possible w's for a pair (u, v) with u < v are:
    #    - w = 2u - v (left of u)
    #    - w = (u + v) / 2 (between u and v, if integer)
    #    - w = 2v - u (right of v)
    #    
    #    So u and v are incompatible if ANY of these w exists in [1, N].
    #    Actually, for the independent set formulation, we just need to know if adding 
    #    a new element x to a set S creates a 3-AP with two existing elements a, b in S.
    #    This is equivalent to: x conflicts with a if there exists b in S such that {x, a, b} is AP.
    #    
    #    For B&B, we process elements in some order. Let's sort by weight (1/x) descending.
    #    Elements: 1, 2, 3, ... 256. Weights: 1, 1/2, 1/3...
    #    So we process 1, 2, 3...
    
    # Precompute for each element i, the set of elements j > i that conflict with i.
    # If we decide to INCLUDE i, we must EXCLUDE all j > i that conflict with i.
    # If we decide to EXCLUDE i, we move to i+1.
    
    conflicts = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Check if u and v conflict
            # 1. Midpoint
            s = u + v
            if s % 2 == 0:
                w = s // 2
                if u < w < v:
                    conflicts[u].add(v)
                    conflicts[v].add(u)
                    continue # Already found a conflict reason
            # 2. Left extension
            w = 2 * u - v
            if 1 <= w < u:
                conflicts[u].add(v)
                conflicts[v].add(u)
                continue
            # 3. Right extension
            w = 2 * v - u
            if v < w <= N:
                conflicts[u].add(v)
                conflicts[v].add(u)
    
    # Weights
    weights = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    # Sort indices by weight descending (which is 1, 2, 3... so index order is already descending weight)
    # But let's be explicit.
    order = list(range(1, N + 1))
    # order is 1..256. Weight of i is 1/i.
    # So 1 is highest weight, 256 is lowest.
    
    best_score = Fraction(0, 1)
    best_set = set()
    
    # B&B State:
    # idx: current index in order (1..N)
    # current_set: set of included elements
    # current_score: sum of weights of included elements
    # available: set of elements still available (not excluded by conflicts with included)
    #    Actually, it's easier to maintain a bitmask or a list of available elements.
    #    Since N=256, a bitset is good. But Python sets are okay if we prune well.
    #    However, checking "is x in available" is O(1) for set.
    #    Updating available: when we include x, we remove x and all conflicts[x] from available.
    
    # To speed up B&B, we need a good upper bound.
    # Upper bound = current_score + sum(weights[i] for i in available)
    # We can maintain a running sum of available weights.
    
    def solve(idx, current_set, current_score, available, avail_sum):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if idx > N:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Upper bound check
        if current_score + avail_sum <= best_score:
            return
            
        # Option 1: Exclude current element (order[idx-1])
        # We don't add it to set, don't remove anything from available
        # Wait, if we exclude it, it's just not in the set. 
        # But we need to advance idx.
        # The "available" set contains elements >= idx that are not conflicting with current_set.
        # If we exclude order[idx-1], it's simply removed from consideration for this branch.
        # It stays "excluded" effectively by moving to next index.
        # But we must ensure that when we recurse, we don't consider it again.
        # Since we iterate idx 1..N, we just move to idx+1.
        # The available set for the next step is the same, minus the element we just passed?
        # No, "available" should only contain elements with index >= idx.
        # Let's define available as a set of element values, not indices.
        # But we process in fixed order 1..N.
        # So at step idx, we consider element e = order[idx-1].
        # If e is not in available, it was already excluded by a conflict with a previous included element.
        # In that case, we just move to idx+1.
        
        e = order[idx - 1]
        
        if e not in available:
            # Already excluded
            solve(idx + 1, current_set, current_score, available, avail_sum)
            return
            
        # Option 1: Exclude e
        # e is removed from available for subsequent steps? 
        # Actually, if we exclude e, it's just not in the set. 
        # But can a later element conflict with e? 
        # No, because e is not in the set. 
        # So we just proceed to idx+1.
        # We need to remove e from available? 
        # If we keep e in available, and a later element f conflicts with e, 
        # does f get excluded? No, because e is not in the set. 
        # The conflict check is "does f conflict with any element in current_set?".
        # Since e is not in current_set, f is not excluded by e.
        # So we don't need to remove e from available based on conflicts.
        # But we must not consider e again. Since we move to idx+1, we won't.
        # However, avail_sum includes weight of e. 
        # If we exclude e, the new avail_sum should subtract weight of e?
        # Yes, because e is no longer a candidate.
        new_avail_sum_excl = avail_sum - weights[e]
        # The available set for the next step is the same set, but we just ignore e?
        # No, we should remove e from the set to keep it clean? 
        # Or just pass the same set and rely on idx?
        # If we pass the same set, and idx increases, we just check `if e not in available` for the NEXT e.
        # This works.
        solve(idx + 1, current_set, current_score, available, new_avail_sum_excl)
        
        # Option 2: Include e
        # Check if e is in available (already done)
        new_set = set(current_set)
        new_set.add(e)
        new_score = current_score + weights[e]
        
        # Update available and avail_sum
        # Remove e and all conflicts of e from available
        to_remove = {e}
        to_remove.update(conflicts[e])
        
        new_avail_sum = new_avail_sum_excl # Start from excluded sum (which is avail_sum - w[e])
        # Subtract weights of other removed elements
        for r in to_remove:
            if r != e:
                if r in available:
                    new_avail_sum -= weights[r]
        
        new_available = available - to_remove
        
        # Pruning check for this branch
        if new_score + new_avail_sum > best_score:
            solve(idx + 1, new_set, new_score, new_available, new_avail_sum)
            
    # Initial call
    initial_available = set(range(1, N + 1))
    initial_avail_sum = sum(weights[1:N+1])
    
    # To improve B&B, we can use a greedy heuristic to get a good initial lower bound.
    # Greedy: pick 1, remove conflicts, pick 2, remove conflicts...
    greedy_set = set()
    greedy_available = set(range(1, N + 1))
    for i in range(1, N + 1):
        if i in greedy_available:
            greedy_set.add(i)
            greedy_available -= conflicts[i]
            greedy_available.discard(i)
            
    if greedy_set:
        best_score = sum(weights[i] for i in greedy_set)
        best_set = set(greedy_set)
        
    solve(1, set(), Fraction(0, 1), initial_available, initial_avail_sum)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
