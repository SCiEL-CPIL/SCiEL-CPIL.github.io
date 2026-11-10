import time
import sys

# Set recursion limit high enough for backtracking
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Leave 5s margin for 90s limit

N = 192
MAX_K = 48

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Candidates in descending order
    # Large numbers are less restrictive as divisors, so they are "safer" to pick first
    candidates = list(range(N, 0, -1))
    
    best_set = []
    best_size = 0
    # We will use a non-recursive or carefully managed recursive approach
    # But Python recursion is slow. Let's try iterative deepening or a simple DFS with pruning.
    
    # State:
    # current_set: list of selected numbers
    # forbidden_residues: dict {a: set of residues mod a that are forbidden for new d}
    #   Actually, for each a in current_set, we store a bitmask of residues r in [0, a) 
    #   such that there exist x, y in current_set \ {a} with (x+y) % a == -r.
    #   Wait, if we add d, we need d % a NOT in forbidden_residues[a].
    
    # Precomputation for speed:
    # For each a in 1..N, we can't precompute the forbidden set as it depends on S.
    # But we can update it incrementally.
    
    # To check if d divides sum of 3 others in S:
    # We need to know if there are x,y,z in S such that x+y+z == k*d.
    # This is hard to maintain incrementally for all d.
    # However, if we process d in descending order, when we consider adding d,
    # all elements in S are > d.
    # So x, y, z > d.
    # x+y+z > 3d.
    # So x+y+z can be 4d, 5d, ...
    # Max sum is 3*192 = 576.
    # If d > 192/3 = 64, then x+y+z < 4d is possible? No, x,y,z > d => sum > 3d.
    # So if d > 64, the only multiple possible is 4d, 5d, etc.
    # Specifically, if d > 192/2 = 96, sum > 3d and sum <= 3*192.
    # If d=100, sum can be 400, 500.
    
    # Let's optimize the "d divides triple sum" check.
    # For a fixed d, and set S (all > d), we can check:
    # Does S contain 3 elements summing to k*d?
    # We can use a set for S for O(1) lookups? 
    # Sum of 3 elements from S to target T:
    # Iterate x in S, y in S, check if T-x-y in S.
    # This is O(|S|^2).
    
    # Let's try a simple backtracking with these checks.
    
    def is_valid_addition(d, S, forbidden_residues):
        # Check 1: d does not divide sum of any 3 distinct elements in S
        n = len(S)
        if n >= 3:
            # Since we add in descending order, all x in S are > d
            # So x+y+z > 3d
            # Max sum is 3*192 = 576
            # Multiples of d to check: 4d, 5d, ... up to 576
            # If d > 144, 4d > 576, so no multiple possible?
            # If d=150, 4d=600 > 576. So no conflict possible from S!
            # This is a huge optimization.
            if 4 * d > 3 * max(S): # max(S) is the first element added, which is the largest
                # Actually max(S) is just S[0] if sorted desc.
                # But S is not necessarily sorted in the list, but we added in desc order.
                # So min(S) > d.
                # Max sum is sum of 3 largest in S.
                # If 4*d > max_possible_sum, then d cannot divide any sum of 3.
                pass
            else:
                # Check specific multiples
                max_sum = S[0] + S[1] + S[2] if n >= 3 else 0
                for k in range(3, max_sum // d + 1):
                    target = k * d
                    # Check if 3 distinct elements in S sum to target
                    # Use 2-pointer or hash set?
                    # S is smallish (<=48).
                    # O(n^2) check
                    found = False
                    # Create a set for O(1) lookup?
                    S_set = set(S)
                    # We need distinct x,y,z
                    for i in range(n):
                        x = S[i]
                        for j in range(i + 1, n):
                            y = S[j]
                            z = target - x - y
                            if z > y and z in S_set:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        else:
            # n < 3, no triple sum possible
            pass
        
        # Check 2: For each a in S, a does not divide d + x + y for any x,y in S\{a}
        # This is equivalent to: d % a NOT in forbidden_residues[a]
        for a in S:
            if d % a in forbidden_residues[a]:
                return False
        
        return True

    def update_forbidden_residues(S, d, forbidden_residues):
        # When adding d, we need to update forbidden_residues for existing a in S.
        # For each a in S, new pairs involving d create new forbidden residues for future elements.
        # Wait, forbidden_residues[a] is the set of residues r such that if we add e, e%a == r is forbidden.
        # This happens if there exist x,y in S\{a} such that x+y+e is divisible by a.
        # i.e., e % a == -(x+y) % a.
        # Currently, forbidden_residues[a] contains residues from pairs in S\{a}.
        # When we add d to S, d becomes part of S.
        # For future additions e, the pairs in S_new\{a} include pairs from S\{a} AND pairs involving d.
        # So we need to add residues corresponding to pairs (d, x) where x in S\{a, d}.
        # And also, does d itself affect the "forbidden" status of existing residues? No.
        # We just add new forbidden residues.
        
        # Also, we need to create an entry for d in forbidden_residues.
        # forbidden_residues[d] will be empty initially?
        # No, forbidden_residues[d] should contain residues r such that e%a == r is forbidden due to pairs in S\{d}.
        # The pairs in S\{d} are the pairs in S.
        # So we need to compute forbidden_residues[d] based on current S.
        
        # Let's restructure:
        # forbidden_residues is a dict {a: set of forbidden residues mod a}.
        # When adding d:
        # 1. Compute forbidden_residues[d] from current S.
        #    For each pair x,y in S, if x!=y, add (-(x+y))%d to forbidden_residues[d].
        # 2. For each a in S, update forbidden_residues[a].
        #    Add residues from pairs (d, x) where x in S, x!=a.
        #    Specifically, for each x in S, if x != a and x != d:
        #       add (-(d+x))%a to forbidden_residues[a].
        #    Note: We don't add pairs (d,d) because distinct.
        
        # This update is O(|S|^2) for step 1 and O(|S|^2) for step 2.
        # Total O(|S|^2) per addition.
        
        # Initialize forbidden_residues[d]
        new_forbidden_d = set()
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            x = S_list[i]
            for j in range(i + 1, n):
                y = S_list[j]
                r = (-(x + y)) % d
                new_forbidden_d.add(r)
        forbidden_residues[d] = new_forbidden_d
        
        # Update existing a
        for a in S:
            if a == d: continue
            # Add residues from pairs (d, x) where x in S, x != a
            for x in S:
                if x == a or x == d:
                    continue
                r = (-(d + x)) % a
                forbidden_residues[a].add(r)
        
        return forbidden_residues

    # Backtracking
    # To avoid deep recursion issues and overhead, we might use a stack or just careful recursion.
    # Given N=192, depth is max 48. Recursion is fine.
    
    def backtrack(idx, S, forbidden_residues, size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if size > best_size:
            best_size = size
            best_set = S.copy()
        
        if idx >= len(candidates):
            return
            
        # Pruning: if remaining candidates + current size <= best_size, prune
        if size + (len(candidates) - idx) <= best_size:
            return
            
        d = candidates[idx]
        
        # Option 1: Include d
        if is_valid_addition(d, S, forbidden_residues):
            # Copy state
            S_new = S + [d]
            # Update forbidden residues
            # To avoid modifying in place, we need to copy the dict and sets.
            # This is expensive. 
            # Alternative: Use a persistent data structure or undo operations.
            # For simplicity and given small depth, we will copy.
            # But copying sets of size up to 192 for each a in S (size 48) is 48 * 192 ops.
            # 48*192 = 9216 ops per node. 2^48 is too big.
            # But we prune heavily.
            
            # Let's try to optimize is_valid_addition and update.
            
            # Actually, copying the entire forbidden_residues dict is O(|S| * avg_set_size).
            # This might be slow.
            
            # Let's try a different representation:
            # Instead of sets, use bitmasks (integers).
            # a <= 192, so bitmask needs 192 bits. Python int handles this.
            # forbidden_residues[a] is an int where bit r is set if r is forbidden.
            
            # Let's refactor to use bitmasks.
            
            pass # Placeholder for refactored logic below
            
        # Option 2: Exclude d
        backtrack(idx + 1, S, forbidden_residues, size)

    # Refactored Backtracking with Bitmasks
    
    def is_valid_addition_bm(d, S, forbidden_residues_bm):
        # Check 1: d does not divide sum of any 3 in S
        n = len(S)
        if n >= 3:
            # Optimization: if 4*d > max_sum(S), return True for this part
            # max_sum(S) is sum of 3 largest in S
            if n >= 3:
                # S is sorted descending? We added in descending order, so S[0] > S[1] ...
                # So S[0], S[1], S[2] are the largest.
                max_sum_3 = S[0] + S[1] + S[2]
                if 4 * d > max_sum_3:
                    # No multiple of d can be formed by 3 elements of S (since min sum > 3d, next is 4d)
                    # Wait, min sum is S[-1]+S[-2]+S[-3].
                    # If min sum > 3d, then sum > 3d.
                    # If max sum < 4d, then sum < 4d.
                    # So sum cannot be a multiple of d (except 3d, but sum > 3d).
                    # So check if min_sum > 3d and max_sum < 4d.
                    min_sum_3 = S[-1] + S[-2] + S[-3]
                    if min_sum_3 > 3 * d and max_sum_3 < 4 * d:
                         # No conflict from triple sum
                         pass
                    else:
                         # Need to check
                         S_set = set(S)
                         found = False
                         # Check multiples k*d in range (min_sum_3, max_sum_3]
                         start_k = (min_sum_3 // d) + 1
                         end_k = max_sum_3 // d
                         for k in range(start_k, end_k + 1):
                             target = k * d
                             # Check 3-sum
                             for i in range(n):
                                 x = S[i]
                                 for j in range(i + 1, n):
                                     y = S[j]
                                     z = target - x - y
                                     if z > y and z in S_set:
                                         found = True
                                         break
                                 if found: break
                             if found: break
                         if found: return False
                else:
                    # 4*d <= max_sum_3, so check all k from 4 to max_sum_3//d
                    S_set = set(S)
                    found = False
                    min_sum_3 = S[-1] + S[-2] + S[-3]
                    start_k = max(4, (min_sum_3 // d) + 1)
                    end_k = max_sum_3 // d
                    for k in range(start_k, end_k + 1):
                        target = k * d
                        for i in range(n):
                            x = S[i]
                            for j in range(i + 1, n):
                                y = S[j]
                                z = target - x - y
                                if z > y and z in S_set:
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: return False
        
        # Check 2: Forbidden residues
        for a in S:
            # d % a must not be in forbidden_residues_bm[a]
            r = d % a
            if (forbidden_residues_bm[a] >> r) & 1:
                return False
        
        return True

    def update_forbidden_residues_bm(S, d, forbidden_residues_bm):
        # Returns new dict
        new_bm = {}
        # Copy existing
        for a in S:
            new_bm[a] = forbidden_residues_bm[a]
        
        # 1. Compute new_bm[d]
        # Pairs in S
        mask_d = 0
        n = len(S)
        for i in range(n):
            x = S[i]
            for j in range(i + 1, n):
                y = S[j]
                r = (-(x + y)) % d
                mask_d |= (1 << r)
        new_bm[d] = mask_d
        
        # 2. Update existing a
        for a in S:
            if a == d: continue
            # Add residues from pairs (d, x) where x in S, x != a
            for x in S:
                if x == a or x == d:
                    continue
                r = (-(d + x)) % a
                new_bm[a] |= (1 << r)
                
        return new_bm

    def backtrack_bm(idx, S, forbidden_residues_bm, size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if size > best_size:
            best_size = size
            best_set = S.copy()
        
        if idx >= len(candidates):
            return
            
        if size + (len(candidates) - idx) <= best_size:
            return
            
        d = candidates[idx]
        
        # Option 1: Include d
        if is_valid_addition_bm(d, S, forbidden_residues_bm):
            S_new = S + [d]
            bm_new = update_forbidden_residues_bm(S, d, forbidden_residues_bm)
            backtrack_bm(idx + 1, S_new, bm_new, size + 1)
            
        # Option 2: Exclude d
        backtrack_bm(idx + 1, S, forbidden_residues_bm, size)

    # Start
    initial_bm = {}
    backtrack_bm(0, [], initial_bm, 0)
    
    return best_set

if __name__ == "__main__":
    construct()
