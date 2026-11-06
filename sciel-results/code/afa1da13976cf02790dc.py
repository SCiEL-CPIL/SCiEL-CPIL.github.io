import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    aps3 = []  # (a, b, c, mask)
    aps4 = []  # (a, b, c, d, mask)
    
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                aps3.append((a, b, c, mask))
                ap_id += 1
    
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
                aps4.append((a, b, c, e, mask))
                ap_id += 1
    
    num_aps3 = len(aps3)
    num_aps4 = len(aps4)
    
    # Build conflict graph on 3-APs: two 3-APs conflict if they share a 4-AP
    # i.e., if there exists a 4-AP such that both 3-APs are subsets of it
    # A 4-AP (a, a+d, a+2d, a+3d) contains two 3-APs: (a,a+d,a+2d) and (a+d,a+2d,a+3d)
    
    # For each 4-AP, find the two 3-APs it contains, and add an edge between them
    # We need to map from (a,b,c) to ap_id
    ap3_to_id = {}
    for i, (a, b, c, mask) in enumerate(aps3):
        ap3_to_id[(a, b, c)] = i
    
    # Conflict graph: adjacency list for each 3-AP
    conflicts = [[] for _ in range(num_aps3)]
    
    for a4, b4, c4, d4, mask4 in aps4:
        # First 3-AP: (a4, b4, c4)
        # Second 3-AP: (b4, c4, d4)
        ap1 = (a4, b4, c4)
        ap2 = (b4, c4, d4)
        
        if ap1 in ap3_to_id and ap2 in ap3_to_id:
            i1 = ap3_to_id[ap1]
            i2 = ap3_to_id[ap2]
            if i1 not in conflicts[i2]:
                conflicts[i1].append(i2)
                conflicts[i2].append(i1)
    
    # Also, two 3-APs conflict if they share an element? No, the constraint is only about 4-APs.
    # Two 3-APs can coexist as long as they don't form a 4-AP together.
    # Actually, the constraint is: no 4 elements form a 4-AP. So if we select a set of 3-APs,
    # the union of their elements must not contain any 4-AP.
    # Two 3-APs conflict if their union contains a 4-AP. This happens if they share 2 elements
    # and the 4th element completes a 4-AP, or if they are disjoint but together with other selected
    # 3-APs form a 4-AP. For simplicity, we'll use the pairwise conflict: two 3-APs conflict
    # if they are the two 3-APs contained in some 4-AP. This is a necessary but not sufficient
    # condition for the union to contain a 4-AP. But for a greedy approach, this is a good start.
    
    # Actually, let's be more careful. Two 3-APs A and B conflict if there exists a 4-AP 
    # that contains elements from both A and B. The simplest case is when A and B are the 
    # two consecutive 3-APs in a 4-AP. But there could be other cases. For now, we'll use
    # the simple conflict: A and B conflict if they are the two 3-APs in a 4-AP.
    
    # Greedy construction on the AP-level graph
    def run_greedy_ap_level(seed):
        rng = random.Random(seed)
        # Order 3-APs by some heuristic: prefer those with fewer conflicts
        deg = [len(conflicts[i]) for i in range(num_aps3)]
        order = list(range(num_aps3))
        rng.shuffle(order)
        
        selected = [False] * num_aps3
        A_mask = 0
        A_set = set()
        
        # Process in order of increasing degree (greedy independent set)
        order.sort(key=lambda x: deg[x])
        
        for i in order:
            # Check if i conflicts with any selected AP
            valid = True
            for j in conflicts[i]:
                if selected[j]:
                    valid = False
                    break
            if valid:
                selected[i] = True
                a, b, c, mask = aps3[i]
                A_mask |= mask
                A_set.add(a)
                A_set.add(b)
                A_set.add(c)
        
        return selected, A_mask, A_set
    
    def count_3aps_from_mask(A_mask):
        count = 0
        for _, _, _, mask in aps3:
            if (A_mask & mask) == mask:
                count += 1
        return count
    
    def count_4aps(A_mask):
        count = 0
        for _, _, _, _, mask in aps4:
            if (A_mask & mask) == mask:
                count += 1
        return count
    
    def is_valid_add_ap(selected, i, A_mask):
        # Check if adding AP i creates a 4-AP
        a, b, c, mask = aps3[i]
        # The new elements are those in mask but not in A_mask
        new_elements = mask & ~A_mask
        # For each new element, check if it forms a 4-AP with existing elements
        # Simpler: check if the new A_mask (A_mask | mask) contains any 4-AP
        new_A_mask = A_mask | mask
        return count_4aps(new_A_mask) == 0
    
    def local_search_ap_level(selected, A_mask, A_set, time_limit):
        # Local search: try adding a 3-AP, removing a 3-AP, or swapping
        # Score = number of 3-APs in the union (not just selected ones)
        best_score = count_3aps_from_mask(A_mask)
        
        while time.time() < time_limit:
            best_delta = 0
            best_action = None  # ('add', i) or ('remove', i) or ('swap', i_remove, i_add)
            
            # Try adding each unselected AP
            for i in range(num_aps3):
                if selected[i]:
                    continue
                if not is_valid_add_ap(selected, i, A_mask):
                    continue
                new_score = count_3aps_from_mask(A_mask | aps3[i][3])
                delta = new_score - best_score
                if delta > best_delta:
                    best_delta = delta
                    best_action = ('add', i)
            
            # Try removing each selected AP
            for i in range(num_aps3):
                if not selected[i]:
                    continue
                new_A_mask = A_mask & ~aps3[i][3]
                # Need to recompute which 3-APs are still present
                # This is expensive; skip for now or do a quick estimate
                # For simplicity, skip removal in this version
                pass
            
            # Try swapping: remove one, add another
            # This is O(n^2) which is too slow; skip for now
            
            if best_action is None:
                break
            
            if best_action[0] == 'add':
                i = best_action[1]
                selected[i] = True
                a, b, c, mask = aps3[i]
                A_mask |= mask
                A_set.add(a)
                A_set.add(b)
                A_set.add(c)
                best_score += best_delta
            
            if time.time() > time_limit:
                break
        
        return selected, A_mask, A_set, best_score
    
    # Run multiple greedy initializations and local search
    best_mask = 0
    best_set = set()
    best_score = 0
    
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        selected, A_mask, A_set = run_greedy_ap_level(seed)
        
        # Ensure maximality: try adding any remaining valid element
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in A_set:
                    bit_x = 1 << (x-1)
                    new_mask = A_mask | bit_x
                    if count_4aps(new_mask) == 0:
                        A_mask = new_mask
                        A_set.add(x)
                        changed = True
        
        score = count_3aps_from_mask(A_mask)
        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = set(A_set)
    
    # Local search from best
    if time.time() < deadline - 2.0:
        # Reconstruct selected from best_mask
        selected = [False] * num_aps3
        for i, (a, b, c, mask) in enumerate(aps3):
            if (best_mask & mask) == mask:
                selected[i] = True
        
        selected, best_mask, best_set, best_score = local_search_ap_level(selected, best_mask, best_set, deadline - 2.0)
    
    # Final validation
    if count_4aps(best_mask) != 0:
        # Fallback
        selected, best_mask, best_set = run_greedy_ap_level(0)
        best_score = count_3aps_from_mask(best_mask)
    
    return sorted(best_set)
