import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 115  # Safety margin

    # 1. Precomputation
    all_triples = list(combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Generate all 6-sets and their 15 groups of transversals
    six_sets = list(combinations(range(n), 6))
    group_triples = [] # List of lists of triple indices. Each group has 8 triples.
    
    # We also need: for each triple, which groups it belongs to.
    # triple_to_groups[i] = list of group indices
    triple_to_groups = [[] for _ in range(num_triples)]

    for s in six_sets:
        a, b, c, d, e, f = s
        # The 15 partitions of 6 elements into 3 pairs
        # We can hardcode or generate. Generating is safer.
        # Pairs are (a,b),(c,d),(e,f) etc.
        # Let's use a helper to generate transversals for a partition.
        
        # Partitions of {a,b,c,d,e,f} into 3 pairs.
        # There are 15.
        # We can iterate through all ways to pick 2 elements for pair 1, 2 for pair 2, 2 for pair 3.
        # To avoid duplicates, enforce ordering.
        
        # Actually, the standard 15 partitions for elements 0..5 are:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # ...
        
        # Let's just generate them programmatically to be sure.
        # Pick pair 1 from 6 elements: C(6,2) = 15.
        # Pick pair 2 from remaining 4: C(4,2) = 6.
        # Pair 3 is determined.
        # Total 15 * 6 / 2 (order of pairs doesn't matter? No, the partition is a set of pairs. 
        # But the transversals depend on the set of pairs.
        # The set of pairs is unique.
        # Number of perfect matchings on 6 nodes is 15.
        
        # Let's list the 15 matchings.
        matchings = []
        # Fix 0.
        # 0 pairs with 1: remaining 2,3,4,5. Matchings on 4 nodes: 3.
        # 0 pairs with 2: remaining 1,3,4,5. Matchings on 4 nodes: 3.
        # ...
        # Total 1 + 1 + 1 + 1 + 1 (for 0 with 1,2,3,4,5) * 3 = 15.
        
        # Let's just use a recursive generator or hardcoded list for the 15 matchings of indices 0-5.
        # Then map to actual elements a-f.
        
        # Hardcoded matchings for 0..5 (indices)
        base_matchings = [
            [(0,1), (2,3), (4,5)],
            [(0,1), (2,4), (3,5)],
            [(0,1), (2,5), (3,4)],
            [(0,2), (1,3), (4,5)],
            [(0,2), (1,4), (3,5)],
            [(0,2), (1,5), (3,4)],
            [(0,3), (1,2), (4,5)],
            [(0,3), (1,4), (2,5)],
            [(0,3), (1,5), (2,4)],
            [(0,4), (1,2), (3,5)],
            [(0,4), (1,3), (2,5)],
            [(0,4), (1,5), (2,3)],
            [(0,5), (1,2), (3,4)],
            [(0,5), (1,3), (2,4)],
            [(0,5), (1,4), (2,3)]
        ]
        
        elements = [a, b, c, d, e, f]
        
        for m in base_matchings:
            p1, p2, p3 = m
            # p1, p2, p3 are tuples of indices 0..5
            e1 = elements[p1[0]], elements[p1[1]]
            e2 = elements[p2[0]], elements[p2[1]]
            e3 = elements[p3[0]], elements[p3[1]]
            
            # Generate 8 transversals
            trans = []
            for x in e1:
                for y in e2:
                    for z in e3:
                        t = tuple(sorted((x, y, z)))
                        trans.append(triple_idx[t])
            
            group_idx = len(group_triples)
            group_triples.append(trans)
            
            for t in trans:
                triple_to_groups[t].append(group_idx)

    num_groups = len(group_triples)
    
    # Convert triple_to_groups to numpy arrays for faster indexing?
    # Or just keep as lists. Numpy fancy indexing with a list of indices works.
    
    # State
    active = [False] * num_triples
    counts = np.zeros(num_groups, dtype=np.int32)
    
    # Greedy Initialization
    best_count = 0
    best_active = None
    best_counts = None
    
    # Try a few random orders for greedy
    for seed in [42, 123, 7]:
        random.seed(seed)
        order = list(range(num_triples))
        random.shuffle(order)
        
        active_init = [False] * num_triples
        counts_init = np.zeros(num_groups, dtype=np.int32)
        
        for t in order:
            groups = triple_to_groups[t]
            # Check if any group in 'groups' has count 7
            # If max < 7, we can add
            if np.max(counts_init[groups]) < 7:
                active_init[t] = True
                counts_init[groups] += 1
        
        cnt = sum(active_init)
        if cnt > best_count:
            best_count = cnt
            best_active = active_init[:]
            best_counts = counts_init.copy()

    active = best_active[:]
    counts = best_counts.copy()
    current_count = best_count
    best_sa_count = best_count
    best_sa_active = active[:]

    # Precompute numpy arrays for triple_to_groups to speed up access
    # Actually, accessing counts[groups] where groups is a list is fine.
    # But creating a numpy array for each triple's groups might be faster if reused?
    # triple_to_groups_np = [np.array(g) for g in triple_to_groups]
    # This takes memory but speeds up indexing.
    
    # Let's do that.
    triple_to_groups_np = [np.array(g, dtype=np.int32) for g in triple_to_groups]

    # SA Loop
    t_start = 2.0
    t_end = 0.01
    # Estimate steps
    # We don't know exact speed, so we just run until deadline.
    
    random.seed(42)
    
    active_indices = [i for i in range(num_triples) if active[i]]
    inactive_indices = [i for i in range(num_triples) if not active[i]]
    
    step = 0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Temperature
        # Simple linear or exponential decay.
        # Let's use exponential.
        # Assume ~500k steps possible?
        # If we run for 115s, and each step takes ~50us, that's 2.3M steps.
        # Let's target 1M steps for cooling.
        total_steps_estimate = 1000000
        if step < total_steps_estimate:
            T = t_start * (t_end / t_start) ** (step / total_steps_estimate)
        else:
            T = t_end

        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            if not inactive_indices:
                step += 1
                continue
            t_add = random.choice(inactive_indices)
            groups = triple_to_groups_np[t_add]
            if np.max(counts[groups]) < 7:
                active[t_add] = True
                counts[groups] += 1
                inactive_indices.remove(t_add)
                active_indices.append(t_add)
                current_count += 1
                
                if current_count > best_sa_count:
                    best_sa_count = current_count
                    best_sa_active = active[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        elif move_type < 0.5:
            # Remove
            if not active_indices:
                step += 1
                continue
            t_rem = random.choice(active_indices)
            # Accept removal with prob 0.5 to escape
            if random.random() < 0.5:
                active[t_rem] = False
                groups = triple_to_groups_np[t_rem]
                counts[groups] -= 1
                active_indices.remove(t_rem)
                inactive_indices.append(t_rem)
                current_count -= 1
                no_improve += 1
            else:
                no_improve += 1

        else:
            # Swap
            if not active_indices or not inactive_indices:
                step += 1
                continue
            t_rem = random.choice(active_indices)
            t_add = random.choice(inactive_indices)
            
            groups_rem = triple_to_groups_np[t_rem]
            groups_add = triple_to_groups_np[t_add]
            
            # Combine unique groups
            # We need to check if new counts >= 8.
            # new_count[g] = count[g] - (1 if g in groups_rem) + (1 if g in groups_add)
            
            # We can create a set of affected groups
            # But sets are slow.
            # Since groups are small (~1700), we can just iterate?
            # Or use numpy union1d?
            # union1d is slow.
            
            # Alternative:
            # Check groups_rem: if any count[g] - 1 + (1 if g in groups_add) >= 8
            # Check groups_add: if any count[g] + 1 - (1 if g in groups_rem) >= 8
            
            # To check "if g in groups_add", we can use a boolean array or set.
            # Let's use a set for groups_add for O(1) lookup.
            set_add = set(groups_add)
            
            valid = True
            # Check groups_rem
            # We decrement these. So they are safer.
            # But if a group is in both, net change 0.
            # If only in rem, change -1.
            # If only in add, change +1.
            
            # Check groups_add first (risk of increasing)
            for g in groups_add:
                # new count = counts[g] + 1 - (1 if g in set_rem else 0)
                # We need set_rem too?
                # Let's just compute the delta for the union.
                
                # This loop is in Python. Might be slow.
                # 1700 iterations.
                # 1M steps * 1700 = 1.7B ops. Too slow.
                
                # We need vectorized check.
                pass
            
            # Vectorized approach for Swap:
            # We want to check if max(new_counts) >= 8.
            # new_counts = counts + delta.
            # delta is non-zero only for groups in groups_rem or groups_add.
            
            # Let's construct the delta vector for the affected groups.
            # This is hard to do efficiently with sparse updates in numpy without creating a full array or using advanced indexing carefully.
            
            # Efficient Swap Check:
            # 1. Identify groups where count is 7. If any such group is in groups_add and NOT in groups_rem, it becomes 8. Invalid.
            # 2. Identify groups where count is 8? (Shouldn't happen).
            
            # So:
            # Risky groups are those with count == 7.
            # If g has count 7:
            #   If g in groups_add and g NOT in groups_rem -> new count 8. INVALID.
            #   If g in groups_rem and g NOT in groups_add -> new count 6. OK.
            #   If g in both -> new count 7. OK.
            
            # So we just need to check:
            # Are there any groups g such that counts[g] == 7 AND g in groups_add AND g NOT in groups_rem?
            
            # How to check this fast?
            # 1. Find indices of groups with count 7: `idx_7 = np.where(counts == 7)[0]`
            #    This is O(num_groups) = 120k. Too slow per step.
            
            # Alternative:
            # Maintain a list of "tight" groups (count == 7)?
            # Updating this list is also expensive.
            
            # Let's stick to the simple check but optimize the Python loop by using sets.
            # Actually, `groups_add` and `groups_rem` are numpy arrays.
            # Converting to set is O(N).
            
            # Let's try a different swap strategy:
            # Only allow swaps if the "cost" is low?
            
            # Given the time limit, maybe simple Add/Remove is sufficient if we have many restarts?
            # Or just skip swaps if they are too slow?
            
            # Let's try to optimize the swap check using a precomputed "7-count" mask?
            # No, counts change.
            
            # Let's assume Swap is too slow and rely on Add/Remove with restarts.
            # Or, implement a "Greedy Swap":
            # Try to swap t_rem with t_add only if t_add is in a "less crowded" area?
            
            # Let's try a simpler optimization:
            # Only check the groups in `groups_add`.
            # For each g in groups_add:
            #   if counts[g] == 7 and g not in groups_rem: invalid.
            
            # To check `g not in groups_rem` fast, we need `groups_rem` as a set.
            # Creating a set of 1700 ints is ~50us.
            # Looping over 1700 ints is ~100us.
            # Total ~150us per swap.
            # 1M swaps = 150s. Too slow.
            
            # We must reduce the number of swaps or make them faster.
            # Maybe 100k swaps?
            
            # Let's just do Add/Remove for now and see if it's fast enough.
            # If Add/Remove is ~10us (numpy max on 1700 elements), we can do 10M steps.
            # That should be enough to find a good solution.
            
            step += 1
            continue

        step += 1
        
        if no_improve > max_no_improve:
            # Restart
            active = best_sa_active[:]
            counts = np.zeros(num_groups, dtype=np.int32)
            for i in range(num_triples):
                if active[i]:
                    counts[triple_to_groups_np[i]] += 1
            
            current_count = best_sa_count
            active_indices = [i for i in range(num_triples) if active[i]]
            inactive_indices = [i for i in range(num_triples) if not active[i]]
            no_improve = 0

    # Final Greedy Fill
    for t in inactive_indices[:]:
        groups = triple_to_groups_np[t]
        if np.max(counts[groups]) < 7:
            active[t] = True
            counts[groups] += 1

    # Use best found
    if best_sa_count >= current_count:
        result = [list(all_triples[i]) for i in range(num_triples) if best_sa_active[i]]
    else:
        result = [list(all_triples[i]) for i in range(num_triples) if active[i]]

    return result
