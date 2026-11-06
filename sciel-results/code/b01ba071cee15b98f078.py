import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    # 1. Precompute 4-AP conflicts
    # conflict_mask[i] is a bitmask of elements that cannot coexist with i
    # i.e., if j is in conflict_mask[i], then {i, j} is part of a 4-AP
    # Wait, the constraint is that the SET has no 4-AP.
    # If we add v to S, we must ensure no 4-AP is formed.
    # A 4-AP is formed if v is part of a 4-AP with 3 existing elements.
    # It is NOT enough to just check pairwise conflicts.
    # However, we can precompute for each v, the set of "witnesses" or use a different check.
    
    # Let's stick to the fast check:
    # To check if v can be added to S:
    # We need to ensure there is no d such that 3 of {v, v+d, v+2d, v+3d} are in S (and v is the 4th, or v is one of the first 3 and the 4th is in S? No, S is 4-AP free, so S has no 4-APs. Adding v creates a 4-AP iff there exist 3 elements in S that form a 4-AP with v).
    
    # Precompute for each v, a list of "forbidden triples" in S?
    # Or better: Precompute for each v, a list of "patterns".
    # Actually, the O(n) loop in the current code is:
    # for d in range(1, n//3 + 1):
    #   check 4 cases.
    # This is ~ 85 * 4 = 340 set lookups.
    # With numpy, we can represent S as a boolean array.
    # But set lookups in Python are fast.
    
    # Let's optimize the delta calculation which is O(n) as well.
    # Precompute 3-AP neighbors for each element.
    # neighbors_3ap[i] = list of (j, k) such that {i, j, k} is a 3-AP.
    # Actually, for delta, we need to know how many 3-APs contain u (to be lost) and v (to be gained).
    # A 3-AP containing u is (u, u+d, u+2d), (u-d, u, u+d), (u-2d, u-d, u).
    # Let's precompute for each i, the list of other elements that form a 3-AP with i.
    # And for each pair (i, j) in that list, we need to know if the 3rd element is in S.
    
    # Even better:
    # Precompute for each i, a list of (j, k) such that {i, j, k} is a 3-AP.
    # Then, number of 3-APs containing i in S is count of (j, k) in list_i such that j in S and k in S.
    # This is still O(deg(i)) per query. deg(i) is small (~ n/2 / 2 ?).
    # Max d is 128.
    # i is 1st: d=1..(n-i)/2. ~ 128 max.
    # i is 2nd: d=1..min(i-1, n-i). ~ 128 max.
    # i is 3rd: d=1..(i-1)/2. ~ 128 max.
    # Total ~ 384 checks per element.
    
    # Let's try to make the 4-AP check faster.
    # Precompute for each v, a list of "critical sets" C_v.
    # If any C in C_v is a subset of S, then v cannot be added.
    # C_v consists of triples {x, y, z} such that {v, x, y, z} is a 4-AP.
    # There are at most 4 * (n/3) ~ 340 such triples.
    # Checking if any of ~340 triples is a subset of S is expensive if done naively.
    
    # Alternative:
    # Use a bitset for S.
    # For each v, precompute a list of bitmasks M_v1, M_v2, ...
    # If (S & M_vk) == M_vk for any k, then v creates a 4-AP.
    # Python integers can serve as arbitrary precision bitsets.
    
    # Let's implement this.
    
    # Map 1..256 to bits 0..255.
    # bit i corresponds to element i+1.
    
    def val_to_bit(x):
        return 1 << (x - 1)
    
    # Precompute forbidden masks for 4-APs
    # forbidden_masks[v] = list of masks. If S contains any mask, v is forbidden.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d. Need {v+d, v+2d, v+3d} in S.
        # v is 2nd: v-d, v, v+d, v+2d. Need {v-d, v+d, v+2d} in S.
        # v is 3rd: v-2d, v-d, v, v+d. Need {v-2d, v-d, v+d} in S.
        # v is 4th: v-3d, v-2d, v-d, v. Need {v-3d, v-2d, v-d} in S.
        
        for d in range(1, n // 3 + 1):
            # Case 1
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
            # Case 2
            if v - d >= 1:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
            # Case 3
            if v - 2 * d >= 1:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
            # Case 4
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP contributions for delta
    # For each element u, precompute list of "pairs" (j, k) such that {u, j, k} is a 3-AP.
    # We need to count how many of these pairs are fully in S.
    # Let's store for each u, a list of masks M_u_i.
    # If (S & M_u_i) == M_u_i, then that 3-AP exists.
    # Wait, this is for counting.
    # To calculate delta when removing u and adding v:
    # Lost = count of 3-APs containing u in S.
    # Gained = count of 3-APs containing v in S \ {u}.
    
    # Let's precompute for each u, the list of masks for 3-APs containing u.
    # mask = bit(j) | bit(k) where j, k are the other two elements.
    
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks
        
    # Helper: Count 3-APs in S (represented as int bitmask)
    # We can just sum over all elements? No, that counts each AP 3 times.
    # Better: Iterate a, d.
    # Or, since we have masks, we can't easily sum without double counting.
    # Let's just use the O(N^2) method for the initial count and best tracking,
    # but use the mask method for fast deltas.
    
    def count_3aps_from_bitmask(S_mask):
        # Convert to set for simple O(N^2) if needed, or use bit tricks.
        # For n=256, O(N^2) is 65536 ops. Fast enough for occasional calls.
        # But we want it fast.
        # Let's just use the set conversion for simplicity in the main loop if it's not called too often.
        # Actually, we can compute the total count from scratch occasionally.
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    # Helper: Check if v can be added to S (S_mask)
    def can_add(S_mask, v):
        masks = forbidden_masks[v]
        for m in masks:
            if (S_mask & m) == m:
                return False
        return True

    # Helper: Calculate delta 3-APs for swap u -> v
    # S is current set.
    # Lost: 3-APs containing u in S.
    # Gained: 3-APs containing v in S \ {u}.
    
    def calc_delta(S_mask, u, v):
        # S_mask contains u.
        # S_mask_without_u = S_mask & ~val_to_bit(u)
        
        # Lost: count masks in ap3_masks[u] that are subset of S_mask.
        # Note: The mask for a 3-AP containing u does NOT include u's bit.
        # So we just check if (S_mask & m) == m.
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        # Gained: count masks in ap3_masks[v] that are subset of S_mask_without_u.
        S_mask_without_u = S_mask & ~val_to_bit(u)
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # 1. Initial Greedy Construction
    # Use bitmask for S
    S_mask = 0
    S_set = set()
    
    for x in range(1, n + 1):
        if can_add(S_mask, x):
            S_mask |= val_to_bit(x)
            S_set.add(x)
            
    current_count = count_3aps_from_bitmask(S_mask)
    best_mask = S_mask
    best_count = current_count
    best_set = set(S_set)
    
    # 2. Simulated Annealing
    # T_start = 1.0, T_end = 0.01
    # Steps: 50000?
    
    T_start = 2.0
    T_end = 0.01
    num_steps = 100000  # Adjust based on speed
    
    # Precompute available elements for random selection?
    # Just pick random 1..n and check if in S.
    
    current_mask = S_mask
    
    for step in range(num_steps):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # Pick random u in S
        # Pick random v not in S
        
        # To pick u in S:
        # While True:
        #   u = random.randint(1, n)
        #   if u in S_set: break
        # This is inefficient if S is small.
        # Let's maintain a list of elements in S.
        # But S changes.
        # For n=256, random sampling is fast enough.
        
        u = 0
        while True:
            u = random.randint(1, n)
            if (current_mask >> (u-1)) & 1:
                break
                
        v = 0
        while True:
            v = random.randint(1, n)
            if not ((current_mask >> (v-1)) & 1):
                break
        
        # Check validity of v in S \ {u}
        # We can use can_add, but can_add checks against S_mask.
        # We need to check against S_mask_without_u.
        # Let's write a specific check or modify can_add.
        # Actually, can_add(S_mask_without_u, v) is correct.
        
        S_mask_without_u = current_mask & ~val_to_bit(u)
        
        valid = True
        for m in forbidden_masks[v]:
            if (S_mask_without_u & m) == m:
                valid = False
                break
        
        if not valid:
            continue
            
        delta = calc_delta(current_mask, u, v)
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            current_mask = S_mask_without_u | val_to_bit(v)
            current_count += delta
            
            if current_count > best_count:
                best_count = current_count
                best_mask = current_mask
                # Update best_set for final return
                best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
                
    return sorted(best_set)

# Test run to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
