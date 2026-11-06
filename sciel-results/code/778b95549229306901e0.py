import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precomputation
    # neighbors3[x]: list of (y, z) such that x, y, z is a 3-AP? 
    # No, let's use: for each x, list of pairs (a, c) such that a, x, c is a 3-AP.
    # This helps for cnt3[x] which counts how many such pairs have a, c in A.
    
    # Actually, simpler:
    # three_ap_partners[x]: list of (y, z) such that {x, y, z} is a 3-AP.
    # If we add x, gain is number of (y, z) in three_ap_partners[x] such that y, z in A.
    
    # four_ap_partners[x]: list of (y, z, w) such that {x, y, z, w} is a 4-AP.
    # If we add x, valid if no (y, z, w) in four_ap_partners[x] has y, z, w in A.
    
    three_ap_partners = [[] for _ in range(N + 1)]
    four_ap_partners = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            # 3-AP: a, b, c
            three_ap_partners[a].append((b, c))
            three_ap_partners[b].append((a, c))
            three_ap_partners[c].append((a, b))
            
            if a + 3 * d <= N:
                e = a + 3 * d
                # 4-AP: a, b, c, e
                four_ap_partners[a].append((b, c, e))
                four_ap_partners[b].append((a, c, e))
                four_ap_partners[c].append((a, b, e))
                four_ap_partners[e].append((a, b, c))

    # State
    in_set = [False] * (N + 1)
    cnt3 = [0] * (N + 1) # cnt3[x] = number of 3-APs containing x where other 2 are in A
    cnt4 = [0] * (N + 1) # cnt4[x] = number of 4-APs containing x where other 3 are in A
    current_score = 0     # Total number of 3-APs
    
    def add_element(x):
        nonlocal current_score
        if in_set[x]:
            return
        # Gain in 3-APs
        gain = cnt3[x]
        current_score += gain
        in_set[x] = True
        
        # Update cnt3 and cnt4 for neighbors
        # For each 3-AP (x, y, z):
        for y, z in three_ap_partners[x]:
            if in_set[y] and in_set[z]:
                # This 3-AP is now complete? No, cnt3 counts "other 2 in A".
                # If we add x, does it affect cnt3[y]?
                # cnt3[y] counts pairs (a, c) in A such that a, y, c is 3-AP.
                # The pair for y in (x, y, z) is (x, z).
                # If x was just added, and z is in A, then (x, z) is a valid pair for y.
                # So cnt3[y] should increase by 1 if z is in A.
                if in_set[z]:
                    cnt3[y] += 1
                if in_set[x]: # x is now in A
                    # Wait, the pair is (x, z). If z is in A, and x is in A, then the pair is complete.
                    # But cnt3[y] counts pairs in A.
                    # Before adding x, x was not in A. So (x, z) was not a complete pair.
                    # After adding x, if z is in A, (x, z) is a complete pair.
                    # So yes, increment cnt3[y].
                    pass # Handled above? No, logic is:
                    # If z is in A, then the pair (x, z) is now "in A" (both elements).
                    # So cnt3[y] increases by 1.
            
            # Symmetrically for z
            if in_set[y]:
                cnt3[z] += 1
                
        # For each 4-AP (x, y, z, w):
        for y, z, w in four_ap_partners[x]:
            # Check if y, z, w are in A
            if in_set[y] and in_set[z] and in_set[w]:
                # This 4-AP is now "complete" for x?
                # cnt4[x] counts 4-APs where other 3 are in A.
                # So cnt4[x] increases by 1.
                cnt4[x] += 1
                
            # Update cnt4 for y, z, w
            # For y, the other 3 are x, z, w.
            # If x, z, w are in A, then cnt4[y] increases by 1.
            if in_set[x] and in_set[z] and in_set[w]:
                cnt4[y] += 1
            if in_set[x] and in_set[y] and in_set[w]:
                cnt4[z] += 1
            if in_set[x] and in_set[y] and in_set[z]:
                cnt4[w] += 1

    def remove_element(x):
        nonlocal current_score
        if not in_set[x]:
            return
        # Loss in 3-APs
        loss = cnt3[x]
        current_score -= loss
        in_set[x] = False
        
        # Update cnt3 and cnt4 for neighbors
        for y, z in three_ap_partners[x]:
            if in_set[y] and in_set[z]:
                cnt3[y] -= 1
            if in_set[y]:
                cnt3[z] -= 1

        for y, z, w in four_ap_partners[x]:
            if in_set[y] and in_set[z] and in_set[w]:
                cnt4[x] -= 1
            if in_set[z] and in_set[w]: # x is out, so check if x,z,w were the others for y?
                 # For y, others are x, z, w.
                 # If x, z, w were in A, cnt4[y] was incremented.
                 # Now x is out. So decrement if z, w are in A.
                 if in_set[z] and in_set[w]:
                     cnt4[y] -= 1
            if in_set[y] and in_set[w]:
                cnt4[z] -= 1
            if in_set[y] and in_set[z]:
                cnt4[w] -= 1

    # Greedy Initialization
    # Sort elements by potential (number of 3-AP partners)
    elements = list(range(1, N + 1))
    elements.sort(key=lambda x: len(three_ap_partners[x]), reverse=True)
    
    for x in elements:
        if cnt4[x] == 0:
            add_element(x)
            
    best_score = current_score
    best_set = in_set[:]
    
    # Local Search
    # We will perform swaps and flips
    # To speed up, we can precompute a list of elements in set and not in set
    # But maintaining it is costly. We can just iterate 1..N.
    
    while time.time() < deadline:
        improved = False
        
        # Try to find a beneficial move
        # Randomly sample some moves to avoid getting stuck in a specific local minimum
        # Or iterate through all? 256 is small.
        
        # 1. Try adding an element
        # Find x not in set with max gain (cnt3[x]) and cnt4[x]==0
        best_add_gain = -1
        best_add_x = -1
        for x in range(1, N + 1):
            if not in_set[x] and cnt4[x] == 0:
                if cnt3[x] > best_add_gain:
                    best_add_gain = cnt3[x]
                    best_add_x = x
                    
        if best_add_x != -1 and best_add_gain > 0:
            add_element(best_add_x)
            if current_score > best_score:
                best_score = current_score
                best_set = in_set[:]
            improved = True
            continue # Restart search from new state
            
        # 2. Try removing an element
        # Find x in set with min loss (-cnt3[x])? i.e. max cnt3[x] is bad to remove?
        # We want to remove x if it allows adding more?
        # Direct removal decreases score by cnt3[x]. So only remove if cnt3[x] is small?
        # Or if removal enables a large gain elsewhere.
        # For now, try removing element with smallest cnt3[x] (least harmful)
        # And see if it allows an add.
        
        # 3. Try swapping
        # Remove x, Add y
        # Gain = cnt3[y] (in new state) - cnt3[x]
        # This is complex to calculate exactly without doing the move.
        # Approximation: Gain ~ cnt3[y] - cnt3[x]
        # But cnt3[y] might change after removing x.
        
        # Let's do a simple local search:
        # Pick random x in set, random y not in set.
        # If valid, calculate delta.
        
        # To make it faster, let's just do random swaps and accept if better.
        
        # Get random x in set
        # Get random y not in set
        
        # Optimization: Precompute list of in_set elements?
        # It's 256, so iterating is fine.
        
        x_rem = -1
        for i in range(1, N + 1):
            if in_set[i]:
                x_rem = i
                break
        
        if x_rem == -1:
            break # Empty set? Should not happen
            
        y_add = -1
        for i in range(1, N + 1):
            if not in_set[i]:
                y_add = i
                break
                
        if y_add == -1:
            break # Full set?
            
        # Check validity of swap:
        # Remove x_rem, Add y_add
        # 1. Check if y_add is valid in (A \ {x_rem})
        #    This requires checking 4-APs for y_add.
        #    We can temporarily remove x_rem, check cnt4[y_add], then restore?
        #    Or just calculate:
        #    cnt4[y_add] in (A \ {x_rem})
        #    Original cnt4[y_add] counts 4-APs with other 3 in A.
        #    If x_rem is one of the "other 3", and the other 2 are in A, then cnt4 decreases by 1.
        
        # Let's just do the move and rollback if bad.
        # This is safe and simple.
        
        # Save state
        old_score = current_score
        old_in_set = in_set[:]
        old_cnt3 = cnt3[:]
        old_cnt4 = cnt4[:]
        
        remove_element(x_rem)
        
        # Now check if y_add is valid
        if cnt4[y_add] == 0:
            add_element(y_add)
            if current_score > best_score:
                best_score = current_score
                best_set = in_set[:]
                improved = True
            else:
                # Rollback
                # This is expensive.
                # Better to calculate delta beforehand?
                pass
        else:
            # Rollback
            pass
        
        # This rollback strategy is too slow.
        # Let's stick to the "best add" and "best swap" logic.
        
        # Improved Local Search:
        # 1. Find best add (max gain, valid)
        # 2. If none, find best swap (max net gain)
        
        # We already did best add.
        
        # Best Swap:
        # Iterate over all pairs (x in A, y not in A)? 256*256 = 65k pairs.
        # For each pair, calculate delta.
        # Delta = (gain of adding y to A\{x}) - (loss of removing x from A)
        # Loss of removing x = cnt3[x]
        # Gain of adding y to A\{x} = cnt3[y] in (A\{x})
        
        # To calculate cnt3[y] in (A\{x}):
        # cnt3[y] counts pairs (a, c) in A such that a, y, c is 3-AP.
        # If x is one of a or c, and the other is in A, then this pair is lost.
        # So cnt3[y]_new = cnt3[y] - (1 if x is in a 3-AP with y and the other in A else 0)
        
        # Similarly for validity of y in A\{x}:
        # cnt4[y]_new = cnt4[y] - (1 if x is in a 4-AP with y and other 2 in A else 0)
        
        best_swap_delta = -1
        best_swap_x = -1
        best_swap_y = -1
        
        # To speed this up, we can limit the search to promising candidates.
        # Or just do it for all if time permits.
        
        # Let's do a limited random search for swaps to avoid O(N^2) per step.
        for _ in range(100):
            if time.time() > deadline:
                break
            x_rand = random.choice([i for i in range(1, N+1) if in_set[i]])
            y_rand = random.choice([i for i in range(1, N+1) if not in_set[i]])
            
            # Calculate delta
            # Loss
            loss = cnt3[x_rand]
            
            # Gain
            # Check if x_rand affects cnt3[y_rand]
            # x_rand is in 3-AP with y_rand?
            # Check three_ap_partners[y_rand] for (x_rand, z) or (z, x_rand)
            # Actually, three_ap_partners[y_rand] contains pairs (a, c) such that a, y_rand, c is 3-AP.
            # So if (x_rand, z) is in three_ap_partners[y_rand], and z is in A, then loss of x_rand reduces gain by 1.
            
            reduction_gain = 0
            for a, c in three_ap_partners[y_rand]:
                if a == x_rand and in_set[c]:
                    reduction_gain += 1
                elif c == x_rand and in_set[a]:
                    reduction_gain += 1
                    
            gain = cnt3[y_rand] - reduction_gain
            delta = gain - loss
            
            # Check validity
            # cnt4[y_rand] in A\{x_rand}
            reduction_validity = 0
            for b, c, d in four_ap_partners[y_rand]:
                if b == x_rand and in_set[c] and in_set[d]:
                    reduction_validity += 1
                elif c == x_rand and in_set[b] and in_set[d]:
                    reduction_validity += 1
                elif d == x_rand and in_set[b] and in_set[c]:
                    reduction_validity += 1
                    
            new_cnt4_y = cnt4[y_rand] - reduction_validity
            
            if new_cnt4_y == 0 and delta > best_swap_delta:
                best_swap_delta = delta
                best_swap_x = x_rand
                best_swap_y = y_rand
                
        if best_swap_x != -1 and best_swap_delta > 0:
            remove_element(best_swap_x)
            add_element(best_swap_y)
            if current_score > best_score:
                best_score = current_score
                best_set = in_set[:]
            improved = True
            continue
            
        if not improved:
            # Stuck? Try a random move to escape?
            # Or just break and restart?
            # For now, break and restart with a different greedy seed?
            # We only have one greedy seed.
            # Let's add a random restart mechanism.
            break

    # Convert best_set to list
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
