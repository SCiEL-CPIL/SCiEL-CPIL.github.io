import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    N = 256
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, b, c) with a < b < c, b-a = c-b => 2b = a+c
    # 4-AP: (a, b, c, d) with a < b < c < d, b-a = c-b = d-c => a, a+d, a+2d, a+3d
    
    # List of 3-APs as indices (0-based)
    ap3_list = []
    ap3_map = [set() for _ in range(N)] # ap3_map[i] holds indices of ap3_list containing i
    
    # List of 4-APs as indices
    ap4_list = []
    ap4_map = [set() for _ in range(N)] # ap4_map[i] holds indices of ap4_list containing i

    # Generate 3-APs
    # a from 0 to N-1, d from 1, a+2d < N
    # b = a+d, c = a+2d
    # But we need distinct a,b,c.
    # Actually, just iterate a, b, c?
    # Faster: iterate a, d.
    for a in range(N):
        for d in range(1, (N - a) // 2):
            b = a + d
            c = a + 2 * d
            if c < N:
                idx = len(ap3_list)
                ap3_list.append((a, b, c))
                ap3_map[a].add(idx)
                ap3_map[b].add(idx)
                ap3_map[c].add(idx)

    # Generate 4-APs
    for a in range(N):
        for d in range(1, (N - a) // 3):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                idx = len(ap4_list)
                ap4_list.append((a, b, c, e))
                ap4_map[a].add(idx)
                ap4_map[b].add(idx)
                ap4_map[c].add(idx)
                ap4_map[e].add(idx)

    # Convert sets to lists for faster iteration if needed, but sets are fine for membership.
    # For counting active APs, we need to check if the other elements are in the current set.
    # To make this O(1) or very fast, we can precompute the "neighbors" in the AP.
    
    # Let's optimize the check:
    # For an AP index k in ap3_map[i], the AP is (u, v, w).
    # If i is u, we need v and w in S.
    # If i is v, we need u and w in S.
    # If i is w, we need u and v in S.
    
    # We can store for each i, and each AP it belongs to, the *other* indices.
    # ap3_other[i] = list of tuples (other1, other2)
    ap3_other = [[] for _ in range(N)]
    for k, (a, b, c) in enumerate(ap3_list):
        ap3_other[a].append((b, c))
        ap3_other[b].append((a, c))
        ap3_other[c].append((a, b))
        
    ap4_other = [[] for _ in range(N)]
    for k, (a, b, c, d) in enumerate(ap4_list):
        ap4_other[a].append((b, c, d))
        ap4_other[b].append((a, c, d))
        ap4_other[c].append((a, b, d))
        ap4_other[d].append((a, b, c))

    def count_active_3aps(i, bits):
        # Count how many 3-APs containing i are fully active (other 2 elements in bits)
        cnt = 0
        for o1, o2 in ap3_other[i]:
            if bits[o1] and bits[o2]:
                cnt += 1
        return cnt

    def count_active_4aps(i, bits):
        cnt = 0
        for o1, o2, o3 in ap4_other[i]:
            if bits[o1] and bits[o2] and bits[o3]:
                cnt += 1
        return cnt

    def get_initial_set(seed_val):
        # Try a few different initialization strategies
        # 1. Greedy: Add x if it doesn't create a 4-AP. (Like seed)
        # 2. Random
        # 3. Residue classes mod 3 or 5
        
        rng = random.Random(seed_val)
        
        # Strategy 1: Greedy (fast)
        if seed_val % 3 == 0:
            bits = np.zeros(N, dtype=bool)
            # Randomize order to break symmetry
            order = list(range(N))
            rng.shuffle(order)
            for x in order:
                # Check if adding x creates a 4-AP
                # x is in ap4_map[x]. Check if any 4-AP containing x has other 3 elements in bits.
                ok = True
                for o1, o2, o3 in ap4_other[x]:
                    if bits[o1] and bits[o2] and bits[o3]:
                        ok = False
                        break
                if ok:
                    bits[x] = True
            return bits
            
        # Strategy 2: Residue mod 5 (avoid 4-APs? Not necessarily, but often sparse)
        # Actually, 4-APs can exist in residue classes. 
        # Example: 1, 2, 3, 4 is a 4-AP. 1, 6, 11, 16 is a 4-AP.
        # So simple residue classes might have 4-APs.
        # Let's stick to Greedy and Random.
        
        # Strategy 2: Random
        if seed_val % 3 == 1:
            bits = np.zeros(N, dtype=bool)
            for x in range(N):
                if rng.random() < 0.5:
                    bits[x] = True
            # Repair: Remove elements that are part of 4-APs
            # Iterate until no 4-APs
            changed = True
            while changed:
                changed = False
                # Find all 4-APs
                for k, (a, b, c, d) in enumerate(ap4_list):
                    if bits[a] and bits[b] and bits[c] and bits[d]:
                        # Remove one, e.g., the last one
                        bits[d] = False
                        changed = True
            return bits

        # Strategy 3: Greedy with random order (already covered by 0)
        # Let's try a different greedy: Maximize 3-APs while avoiding 4-APs?
        # That's the objective.
        # For now, just return the greedy one for other seeds too, with different shuffle.
        bits = np.zeros(N, dtype=bool)
        order = list(range(N))
        rng.shuffle(order)
        for x in order:
            ok = True
            for o1, o2, o3 in ap4_other[x]:
                if bits[o1] and bits[o2] and bits[o3]:
                    ok = False
                    break
            if ok:
                bits[x] = True
        return bits

    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_score = -1
    best_bits = None
    
    num_starts = 50 # Adjust based on time
    start_id = 0
    
    while time.time() < deadline:
        # Check if we have time for another start
        # Roughly estimate time per start.
        # If time is low, break.
        
        bits = get_initial_set(start_id)
        
        # Calculate initial score
        cnt3 = 0
        for k, (a, b, c) in enumerate(ap3_list):
            if bits[a] and bits[b] and bits[c]:
                cnt3 += 1
        
        # Check validity (should be 0 for greedy, might be >0 for random)
        cnt4 = 0
        for k, (a, b, c, d) in enumerate(ap4_list):
            if bits[a] and bits[b] and bits[c] and bits[d]:
                cnt4 += 1
        
        if cnt4 > 0:
            # Repair: remove elements from 4-APs
            # Simple repair: iterate 4-APs, remove last element if all present
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for k, (a, b, c, d) in enumerate(ap4_list):
                    if bits[a] and bits[b] and bits[c] and bits[d]:
                        # Remove d
                        # Update cnt3 and cnt4
                        # This repair is slow if done naively.
                        # For random init, it might be better to just discard if repair is too hard.
                        # But let's try a simple repair.
                        bits[d] = False
                        changed = True
                        # Recalculate counts? Too slow.
                        # Let's just break and recalculate at the end of repair.
                        break
                # If we broke early, loop again
            # Recalculate counts after repair
            cnt3 = 0
            for k, (a, b, c) in enumerate(ap3_list):
                if bits[a] and bits[b] and bits[c]:
                    cnt3 += 1
            cnt4 = 0
            for k, (a, b, c, d) in enumerate(ap4_list):
                if bits[a] and bits[b] and bits[c] and bits[d]:
                    cnt4 += 1
            
            if cnt4 > 0:
                # Still invalid? Discard this start.
                start_id += 1
                continue

        if cnt4 == 0:
            if cnt3 > best_score:
                best_score = cnt3
                best_bits = bits.copy()
        
        # Local Search (Hill Climbing)
        # Try flips and swaps
        # We will do a limited number of steps per start to ensure we finish.
        # Estimate: 1000 steps per start?
        
        improved = True
        steps = 0
        max_steps_per_start = 2000 # Tune this
        
        while improved and steps < max_steps_per_start and time.time() < deadline:
            improved = False
            steps += 1
            
            # 1. Try Single Flips (Add)
            # Find candidates to add
            # We can iterate through all i not in bits.
            # To speed up, we can prioritize i that create many 3-APs.
            # But simple iteration is fine if N=256.
            
            best_delta = 0
            best_action = None # (type, i, j)
            
            # Check Adds
            for i in range(N):
                if not bits[i]:
                    # Check if adding i creates 4-APs
                    # Count active 4-APs containing i
                    active_4 = 0
                    for o1, o2, o3 in ap4_other[i]:
                        if bits[o1] and bits[o2] and bits[o3]:
                            active_4 += 1
                    if active_4 > 0:
                        continue # Would create 4-APs, invalid
                    
                    # Count active 3-APs containing i
                    active_3 = 0
                    for o1, o2 in ap3_other[i]:
                        if bits[o1] and bits[o2]:
                            active_3 += 1
                    
                    if active_3 > best_delta:
                        best_delta = active_3
                        best_action = ('add', i, -1)
            
            # Check Removals (if we want to change size, but usually adding is better for max 3-APs)
            # Removing an element reduces 3-APs. So we only remove if it allows adding more?
            # That's a swap.
            
            # Check Swaps (Remove i, Add j)
            # This is more expensive: O(N^2).
            # N=256, N^2 = 65536.
            # For each swap, we need to check validity and delta.
            # This might be too slow for 2000 steps.
            # 2000 * 65536 = 131 million operations. Too slow in Python.
            
            # Optimization:
            # Only consider swaps if no add is possible.
            # Or, limit swaps to a smaller subset.
            
            if best_action is not None:
                # Perform Add
                i = best_action[1]
                bits[i] = True
                cnt3 += best_delta
                improved = True
            else:
                # No simple add. Try Swaps.
                # To make it faster, we can sample a few swaps.
                # Or, iterate i in bits, and for each i, check j not in bits.
                # Still O(N^2).
                
                # Let's try a random subset of swaps to keep it fast.
                # Or, just do a few random swaps.
                
                # Actually, let's try to be smarter.
                # If we are stuck, maybe we need to remove a "bad" element to add a "good" one.
                
                # Let's just do a limited random search for swaps.
                found_swap = False
                for _ in range(50): # 50 random swap attempts
                    i = rng.randint(0, N-1)
                    if not bits[i]: continue
                    j = rng.randint(0, N-1)
                    if bits[j]: continue
                    
                    # Simulate remove i, add j
                    # Check if removing i breaks 4-APs? No, removing never creates 4-APs.
                    # Check if adding j creates 4-APs given i is removed.
                    
                    # Count active 4-APs for j, assuming i is removed.
                    # An AP is active if all other elements are in bits.
                    # If one of the other elements is i, it's not active (since we remove i).
                    
                    active_4_j = 0
                    for o1, o2, o3 in ap4_other[j]:
                        # Check if o1, o2, o3 are in bits, considering i is removed
                        # If any of them is i, it's effectively 0.
                        # So we check bits[o1] and o1 != i, etc.
                        if (bits[o1] and o1 != i) and (bits[o2] and o2 != i) and (bits[o3] and o3 != i):
                            active_4_j += 1
                    
                    if active_4_j > 0:
                        continue # Invalid swap
                    
                    # Calculate delta 3-APs
                    # Delta = (Active 3-APs for j with i removed) - (Active 3-APs for i currently)
                    
                    active_3_j = 0
                    for o1, o2 in ap3_other[j]:
                        if (bits[o1] and o1 != i) and (bits[o2] and o2 != i):
                            active_3_j += 1
                    
                    active_3_i = 0
                    for o1, o2 in ap3_other[i]:
                        if bits[o1] and bits[o2]:
                            active_3_i += 1
                    
                    delta = active_3_j - active_3_i
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_action = ('swap', i, j)
                        found_swap = True
            
            if found_swap:
                i = best_action[1]
                j = best_action[2]
                bits[i] = False
                bits[j] = True
                cnt3 += best_delta
                improved = True
            else:
                # No improvement found
                break
        
        start_id += 1

    if best_bits is None:
        # Fallback to seed-like greedy
        bits = np.zeros(N, dtype=bool)
        for x in range(N):
            ok = True
            for o1, o2, o3 in ap4_other[x]:
                if bits[o1] and bits[o2] and bits[o3]:
                    ok = False
                    break
            if ok:
                bits[x] = True
        return sorted([i+1 for i in range(N) if bits[i]])

    return sorted([i+1 for i in range(N) if best_bits[i]])

# EVOLVE-BLOCK-END
