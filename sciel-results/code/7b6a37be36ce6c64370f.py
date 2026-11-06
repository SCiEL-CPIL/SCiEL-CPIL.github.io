"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Use a safety margin. The evaluator kills at 90s, but we should be safe.
    # Let's aim to finish well before then.
    deadline = start_time + 85.0
    
    n = 256
    
    # Precompute 3-APs and conflict structures
    # conflict[x] is a set of y such that x and y cannot both be in a 3-AP-free set?
    # No, that's for 2-AP. For 3-AP, we need to track pairs.
    # If we have a set S, and we want to add x, we must ensure that for no y, z in S,
    # {x, y, z} is a 3-AP.
    
    # Efficient B&B approach:
    # Order elements by descending reciprocal (i.e., ascending value 1..n).
    # We decide for each element in this order whether to include it or not.
    # State: (index, current_set, current_sum, current_conflicts)
    # current_conflicts can be represented as a bitmask or a set of "blocked" future elements?
    # Actually, standard B&B for independent set on a graph (where edges represent 3-AP conflicts) 
    # is hard because the graph is not just pairwise.
    # A 3-AP is a hyperedge of size 3.
    
    # However, we can reformulate:
    # A set S is 3-AP-free iff for every x in S, and every y in S, the third element z
    # completing the AP (if it exists) is NOT in S.
    
    # Let's use a recursive B&B that processes elements in order 1, 2, ..., n.
    # At step i, we decide if i is in S.
    # If we include i, we must check if it forms an AP with any two previously included elements.
    # If it does, we cannot include i.
    # Also, if we include i, it might "block" future elements?
    # Actually, if we include i, it doesn't block future elements directly in a simple way,
    # unless we consider that future elements j, k might form an AP with i.
    # But since we process in increasing order, if we include i, and later include j,
    # we must check if k exists such that i, j, k is an AP.
    # This depends on the set of *all* previously included elements.
    
    # So the state must include the set of included elements? That's too large.
    # Alternative:
    # The constraint is local in a sense?
    # If we fix the set of included elements less than i, call it S_<i.
    # When considering i, we check if i forms an AP with any pair in S_<i.
    # If yes, i cannot be included.
    # If no, i can be included.
    # If we include i, it becomes part of S_<=i.
    # Future elements j > i will check against S_<=i.
    
    # So the "state" for the bound needs to know S_<i to compute the upper bound for the remaining elements?
    # The upper bound for the remaining sum is sum(1/x for x in i+1..n).
    # This is independent of the choices made so far, EXCEPT that some future elements might be
    # forced to be excluded due to conflicts with current elements?
    # No, conflicts are not "forced exclusion" in a static sense.
    # If we include i, and later we see j, we check if (i, j, k) or (i, k, j) or (k, i, j) is an AP.
    # Since k > j > i or similar, the check is dynamic.
    
    # However, we can prune if:
    # current_sum + sum(1/x for x in i+1..n) <= best_sum_found
    # This is a standard bound. It is very loose because it assumes we can take ALL remaining elements.
    # To tighten, we need to know which remaining elements are "blocked" by the current set.
    # An element x > i is "blocked" if there exist y, z in S_<i such that {x, y, z} is an AP?
    # No, that's not right. x is blocked only if y AND z are in S.
    # Since y, z are already decided, if both are in S, then x CANNOT be in S.
    # So, we can precompute for each step i, given S_<i, the set of "impossible" future elements.
    # Let B_i be the set of x > i such that there exist y, z in S_<i with {x, y, z} being a 3-AP.
    # Then, the upper bound is current_sum + sum(1/x for x in i+1..n if x not in B_i).
    
    # This bound is much tighter.
    
    # Algorithm:
    # 1. Precompute all 3-APs.
    # 2. Recursive function dfs(i, S_current, current_sum, B_current):
    #    - If i > n: update best.
    #    - Calculate UB = current_sum + sum(1/x for x in i+1..n if x not in B_current).
    #    - If UB <= best_sum: return.
    #    - Option 1: Exclude i.
    #      - New S = S_current
    #      - New B = B_current (excluding i? No, i is not in B, i is the current candidate)
    #      - Wait, B_current contains elements > i-1 that are blocked by S_<i.
    #      - If we exclude i, S_<i+1 = S_<i. So B_{i+1} = B_current (but we only care about x > i+1).
    #      - Actually, B_current might contain i? No, B_current is defined for x > i-1 based on S_<i.
    #      - Let's refine definitions.
    #      - Let S be the set of chosen elements from {1, ..., i-1}.
    #      - Let B be the set of elements x >= i such that x cannot be chosen because it forms an AP with two elements in S.
    #      - Note: If x in B, it is definitely not in S_final.
    #      - If we choose to EXCLUDE i:
    #         - S' = S
    #         - B' = B \ {i} (since i is no longer a candidate, and it wasn't in S, it doesn't block anything new? 
    #           Wait, if i was in B, it was already excluded. If i was not in B, it was a candidate.
    #           By excluding i, we don't add any new conflicts. So B' = B \ {i} if i in B, else B.
    #           Actually, B is a set of "forbidden" future elements. i is the current element.
    #           If i is not in S, it doesn't block anything.
    #           So B_next = B (restricted to > i).
    #      - If we choose to INCLUDE i (only if i not in B):
    #         - S' = S ∪ {i}
    #         - New conflicts: i might block future elements.
    #         - For any y in S, if (i, y, z) is an AP, then z is blocked.
    #         - Also, if (y, i, z) is an AP? Since i > y, this is the same.
    #         - Also, if (y, z, i) is an AP? Then y+z=2i. Since y,z < i, this is possible.
    #           But if y, z are in S, and i is added, then {y, z, i} is an AP in S'. This is invalid!
    #           So we can only include i if it does NOT form an AP with any pair in S.
    #           This is the condition "i not in B" if B was defined as "elements forming AP with 2 in S".
    #           Let's verify: B = { x | exists y, z in S, {x, y, z} is AP }.
    #           Yes. So if i in B, we MUST exclude i.
    #         - If we include i, we compute new blocked elements B_add.
    #           For any y in S, find z such that {i, y, z} is AP.
    #           Cases:
    #           1. i is middle: y, i, z => z = 2i - y. If z > i and z <= n, add z to B_add.
    #           2. i is end: y, z, i => z = 2i - y. If z > i? No, z < i. But z is already processed.
    #              Wait, if z < i, and z in S, then {y, z, i} is an AP with y, z in S.
    #              This would mean i is in B. So if we are including i, we know i not in B, so this case doesn't happen.
    #           3. i is start: i, y, z => z = 2y - i. Since y < i, z < y < i.
    #              So z is already processed. If z in S, then {i, y, z} is AP with y, z in S?
    #              No, y is in S, z is in S. So i would be in B.
    #              So again, if i not in B, this case doesn't produce a conflict with *two* existing elements.
    #           
    #           So, when adding i, we only block future elements z > i such that {i, y, z} is an AP for some y in S.
    #           This happens if z = 2i - y (i is middle) or z = 2y - i (i is start, but z < i so not future).
    #           Wait, if i is start, z = 2y - i. Since y < i, z < i. So no future elements blocked this way.
    #           If i is middle, z = 2i - y. Since y < i, z > i. So z is future.
    #           So B_add = { 2i - y | y in S, 2i - y <= n, 2i - y > i }.
    #           Also, what if y > i? No, y in S means y < i.
    #           
    #           So B_new = B ∪ B_add.
    
    # 3. Optimization:
    #    - Use Fraction for sums and bounds.
    #    - Precompute 1/x as Fractions.
    #    - Use a global best_sum.
    #    - To speed up, we can precompute for each i and each possible S? No, S is too big.
    #    - We need to calculate B_add quickly.
    #      B_add depends on S. S is a set of integers.
    #      Iterating y in S to compute 2i-y is O(|S|).
    #      Total complexity might be high, but B&B prunes a lot.
    
    # Let's implement this.
    
    inv = [Fraction(0)] + [Fraction(1, i) for i in range(1, n+1)]
    
    # Precompute suffix sums of reciprocals for the "all remaining" bound
    # suffix_sum[i] = sum(1/x for x in i..n)
    suffix_sum = [Fraction(0)] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + inv[i]
        
    # To calculate the tighter bound, we need sum(1/x for x in i..n if x not in B).
    # This is suffix_sum[i] - sum(1/x for x in B if x >= i).
    # Maintaining sum(1/x for x in B) might be tricky if B changes.
    # But B only grows (elements are added to B, never removed, except we move past them).
    # Actually, B is a set of "blocked" elements.
    # Let's maintain a boolean array is_blocked[1..n].
    # And a running sum of blocked reciprocals?
    # When we add to B, we add to blocked_sum.
    # When we move past i, if i was blocked, we don't subtract it from blocked_sum?
    # The bound is: current_sum + (suffix_sum[i] - sum_{x in B, x>=i} 1/x).
    # If we maintain total_blocked_sum = sum_{x in B} 1/x, and we know which are < i?
    # It's easier to just iterate B and sum? B might be large.
    # Or, maintain a list of blocked elements and sum them?
    
    # Let's try a simpler bound first: current_sum + suffix_sum[i].
    # If that's too slow, tighten.
    # Given n=256, the search space is 2^256, but B&B with good bounds should be fine.
    # The "all remaining" bound is very optimistic.
    # Let's try to implement the tighter bound efficiently.
    
    # State:
    # i: current index
    # S: set of chosen elements (list or set)
    # current_sum: Fraction
    # blocked: set of blocked elements > i? Or all?
    # Let's keep blocked as a set of all blocked elements so far.
    # We need sum(1/x for x in blocked if x >= i).
    # We can maintain a Fenwick tree or just a sum?
    # Since n is small (256), we can just iterate over blocked elements in the bound calculation.
    # |blocked| <= n. So O(n) per bound calc.
    # Depth n. Total O(n^2) per path? No, B&B branches.
    
    best_sum = Fraction(0)
    best_set = []
    
    # Precompute AP triples for fast lookup?
    # We need to check if i is in B.
    # B is constructed dynamically.
    
    # To speed up "is i in B", we can use a boolean array.
    # But B changes with recursion. We need to backtrack.
    # So we should pass a copy or use a global array with undo.
    # Given depth 256, copying a set of size 256 is O(256).
    # 2^256 is too big, but B&B prunes.
    # Let's hope the pruning is effective.
    
    # Improvement: Order of processing.
    # We process 1..n.
    # Elements with large reciprocals (small i) are processed first.
    # This helps finding good solutions early, tightening the bound.
    
    def dfs(i, S, current_sum, blocked):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if i > n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(S)
            return
        
        # Calculate Upper Bound
        # UB = current_sum + sum(1/x for x in i..n if x not in blocked)
        # sum(1/x for x in i..n) is suffix_sum[i]
        # sum(1/x for x in blocked and x >= i)
        blocked_sum = Fraction(0)
        for x in blocked:
            if x >= i:
                blocked_sum += inv[x]
                
        ub = current_sum + suffix_sum[i] - blocked_sum
        
        if ub <= best_sum:
            return
            
        # Option 1: Exclude i
        # i is not added to S.
        # blocked remains the same (i is not added to blocked, it's just skipped).
        # But wait, if i was in blocked, it's already excluded.
        # If i was not in blocked, it was a candidate. By excluding it, it doesn't block anything.
        # So blocked_next = blocked.
        # However, we must ensure we don't consider i in the future.
        # The loop increments i, so we move past it.
        dfs(i+1, S, current_sum, blocked)
        
        # Option 2: Include i
        # Only if i is not in blocked.
        if i not in blocked:
            # Check if adding i creates an AP with two elements in S?
            # This is equivalent to checking if i is in the set of "would-be blocked" elements generated by S.
            # Our `blocked` set contains elements blocked by S.
            # So if i not in blocked, it is safe to add.
            
            new_S = S | {i}
            new_current_sum = current_sum + inv[i]
            
            # Calculate new blocked elements
            # New elements blocked by i and S:
            # For each y in S, z = 2i - y. If z > i and z <= n, z is blocked.
            new_blocked = set(blocked)
            for y in S:
                z = 2*i - y
                if z > i and z <= n:
                    new_blocked.add(z)
            
            dfs(i+1, new_S, new_current_sum, new_blocked)
            
    # Initial call
    # S = empty, current_sum = 0, blocked = empty
    # But wait, the first element 1 cannot be blocked by anything.
    # dfs(1, set(), Fraction(0), set())
    
    # To improve performance, we can sort the decision order?
    # The order 1..n is natural for the "blocked" logic (future elements).
    # If we change order, the "future" logic breaks.
    # So we stick to 1..n.
    
    # One optimization: 
    # If we have time, we can run a local search on the best_set found by B&B.
    # B&B might find the global optimum or a very good one.
    
    dfs(1, set(), Fraction(0), set())
    
    # If B&B didn't finish or found something, we can try to polish best_set with local search.
    # But B&B is exact if it finishes. If it times out, it returns the best found so far.
    # The best found so far is a valid 3-AP-free set.
    
    # Let's add a quick local search on best_set to see if we can squeeze more.
    # This is a "hybrid" approach.
    
    if best_set and time.time() < deadline - 5:
        S = set(best_set)
        current_sum = sum(inv[x] for x in S)
        
        # Precompute conflict pairs for local search
        conflict_pairs = {i: [] for i in range(1, n+1)}
        for b in range(2, n):
            for a in range(1, b):
                c = 2*b - a
                if c <= n:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
                    
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for u in list(S):
                best_v = -1
                best_gain = Fraction(0)
                # Look for v < u not in S
                for v in range(1, u):
                    if v in S:
                        continue
                    # Check if v can be added to S \ {u}
                    valid = True
                    for y, z in conflict_pairs[v]:
                        if y in S and z in S:
                            if y != u and z != u:
                                valid = False
                                break
                    if valid:
                        gain = inv[v] - inv[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                if best_v != -1:
                    S.remove(u)
                    S.add(best_v)
                    current_sum += best_gain
                    improved = True
                    
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(S)
            
    return best_set

# EVOLVE-BLOCK-END
