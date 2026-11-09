import time
import random
import sys

N = 256
DEADLINE_MARGIN = 5.0

def _precompute():
    """
    Precompute 3-AP and 4-AP structures for each element.
    three_ap_pairs[x] = list of (y, z) such that {x, y, z} is a 3-AP.
    four_ap_triples[x] = list of (y, z, w) such that {x, y, z, w} is a 4-AP.
    """
    three_ap_pairs = [[] for _ in range(N + 1)]
    four_ap_triples = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, b, c) with a < b < c and b-a = c-b
    # For each x, we need to find all y, z such that {x, y, z} is a 3-AP.
    # x can be the first, second, or third term.
    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if z <= N:
                three_ap_pairs[x].append((y, z))
        
        # x as second: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))
        
        # x as third: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))
                
    # 4-APs: (a, b, c, d) with a < b < c < d and b-a = c-b = d-c
    # For each x, we need to find all y, z, w such that {x, y, z, w} is a 4-AP.
    for x in range(1, N + 1):
        # x as 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            y = x + d
            z = x + 2 * d
            w = x + 3 * d
            if w <= N:
                four_ap_triples[x].append((y, z, w))
        
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            y = x - d
            z = x + d
            w = x + 2 * d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            y = x - 2 * d
            z = x - d
            w = x + d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            y = x - 3 * d
            z = x - 2 * d
            w = x - d
            if y >= 1:
                four_ap_triples[x].append((y, z, w))
                
    return three_ap_pairs, four_ap_triples

THREE_AP_PAIRS, FOUR_AP_TRIPLES = _precompute()

def _count_3aps(mask):
    """
    Count distinct 3-APs in the set represented by mask.
    Each 3-AP {a, b, c} is counted 3 times in the sum (once for each element).
    So we divide by 3.
    """
    count = 0
    # Iterate over all x in mask
    # To iterate bits in a mask efficiently in Python without external libs:
    # We can just loop 1..N, it's fast enough for N=256.
    for x in range(1, N + 1):
        if (mask >> x) & 1:
            for y, z in THREE_AP_PAIRS[x]:
                if ((mask >> y) & 1) and ((mask >> z) & 1):
                    count += 1
    return count // 3

def _creates_4ap(x, mask):
    """
    Check if adding x to mask creates a 4-AP.
    x is NOT in mask.
    """
    for y, z, w in FOUR_AP_TRIPLES[x]:
        if ((mask >> y) & 1) and ((mask >> z) & 1) and ((mask >> w) & 1):
            return True
    return False

def _gain_3ap(x, mask):
    """
    Count how many 3-APs are completed by adding x to mask.
    x is NOT in mask.
    """
    gain = 0
    for y, z in THREE_AP_PAIRS[x]:
        if ((mask >> y) & 1) and ((mask >> z) & 1):
            gain += 1
    return gain

def _greedy_construct(deadline, conflict_weight=0.1, noise=0.0):
    """
    Greedy construction with conflict penalty.
    """
    mask = 0
    remaining = set(range(1, N + 1))
    
    # Precompute conflict degrees? 
    # Conflict degree of x with respect to remaining set:
    # Number of elements y in remaining such that x and y are in a common 4-AP.
    # This is expensive to update dynamically. 
    # Approximation: Count how many of the neighbors in 4-AP graph are in remaining.
    
    while remaining and time.time() < deadline - 2:
        best_x = -1
        best_score = -1e18
        
        for x in remaining:
            if _creates_4ap(x, mask):
                continue
            
            gain = _gain_3ap(x, mask)
            
            # Calculate conflict count
            conflict_count = 0
            for y, z, w in FOUR_AP_TRIPLES[x]:
                # If y, z, w are in remaining, they might form 4-APs with x later?
                # Actually, the conflict is: if we add x, we might block adding y, z, w later 
                # because they would form a 4-AP with x.
                # But we only care if they are *already* in mask (checked above).
                # The "future penalty" is: how many potential 4-APs do we create with future elements?
                # If y is in remaining, and we add x, then adding y later will be forbidden 
                # IF z and w are also in mask or remaining?
                # Actually, adding x forbids adding y ONLY IF z and w are already in mask.
                # If z and w are in remaining, then adding x doesn't immediately forbid y, 
                # but it creates a "threat".
                
                # A better proxy: How many elements in `remaining` are "close" to x in the 4-AP graph?
                # Let's just count how many of {y, z, w} are in `remaining`.
                if y in remaining:
                    conflict_count += 1
                if z in remaining:
                    conflict_count += 1
                if w in remaining:
                    conflict_count += 1
            
            score = gain * 10.0 - conflict_count * conflict_weight
            if noise > 0:
                score += random.gauss(0, noise)
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x == -1:
            break
            
        mask |= (1 << best_x)
        remaining.remove(best_x)
        
    return mask

def _local_search(mask, deadline):
    """
    Local search to improve 3-AP count.
    Moves:
    1. Swap: Remove i, Add j.
    2. Add: Add j (if valid).
    3. Remove: Remove i (to allow better future adds? Maybe not directly, but can unblock).
    """
    current_count = _count_3aps(mask)
    in_set = set(x for x in range(1, N + 1) if (mask >> x) & 1)
    not_in_set = set(range(1, N + 1)) - in_set
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. Try Swaps
        # To make this fast, we can iterate over a subset of swaps.
        # Full swap search is O(|A| * |Complement|).
        # |A| ~ 100, |Comp| ~ 150. 15000 swaps.
        # Each swap check is O(1) for validity (if we precompute?) 
        # No, validity check is O(deg_4ap).
        
        # Optimization: Precompute loss/gain for each element?
        # Loss of removing i: count of 3-APs containing i.
        # Gain of adding j: count of 3-APs completed by j.
        
        # Let's compute loss for all i in A
        losses = {}
        for i in in_set:
            mask_no_i = mask & ~(1 << i)
            # Count 3-APs in mask_no_i
            # This is O(N * deg). Too slow for all i?
            # 100 * 256 * 50 = 1.2M ops. Might be okay in Python if optimized?
            # Better: Calculate loss directly.
            # loss(i) = number of 3-APs containing i.
            loss = 0
            for y, z in THREE_AP_PAIRS[i]:
                if ((mask >> y) & 1) and ((mask >> z) & 1):
                    loss += 1
            losses[i] = loss
            
        # Let's compute gain for all j not in A
        gains = {}
        for j in not_in_set:
            if _creates_4ap(j, mask):
                gains[j] = -1 # Invalid
            else:
                gain = 0
                for y, z in THREE_AP_PAIRS[j]:
                    if ((mask >> y) & 1) and ((mask >> z) & 1):
                        gain += 1
                gains[j] = gain
                
        # Sort candidates
        # We want max (gain_j - loss_i)
        # But validity of (mask - i + j) depends on interaction.
        # Removing i might make j valid (if j was invalid due to i being in mask).
        # Adding j might make some other element invalid? No, we only check if j creates 4-AP with mask-i.
        
        # Simple hill climbing:
        # Try best few swaps.
        # Sort in_set by loss (ascending) - we want to remove low-loss elements.
        # Sort not_in_set by gain (descending) - we want to add high-gain elements.
        
        sorted_in = sorted(in_set, key=lambda x: losses[x])
        sorted_not_in = sorted(not_in_set, key=lambda x: gains[x], reverse=True)
        
        # Try top K swaps
        K = 20
        found_improvement = False
        for i in sorted_in[:K]:
            if found_improvement:
                break
            if time.time() > deadline - 2:
                break
                
            mask_no_i = mask & ~(1 << i)
            
            for j in sorted_not_in[:K]:
                if found_improvement:
                    break
                if time.time() > deadline - 2:
                    break
                    
                # Check if j is valid in mask_no_i
                if _creates_4ap(j, mask_no_i):
                    continue
                    
                # Calculate new count
                # new_count = current_count - loss(i) + gain(j in mask_no_i)
                # Note: gain(j) was calculated with respect to `mask`, not `mask_no_i`.
                # If i was part of a 3-AP with j and another element, gain(j) might be overestimated.
                # However, for a quick heuristic, it's okay.
                # For exactness, we should recalculate gain for j in mask_no_i.
                
                gain_j_in_mask_no_i = 0
                for y, z in THREE_AP_PAIRS[j]:
                    if ((mask_no_i >> y) & 1) and ((mask_no_i >> z) & 1):
                        gain_j_in_mask_no_i += 1
                    
                new_count = current_count - losses[i] + gain_j_in_mask_no_i
                
                if new_count > current_count:
                    # Perform swap
                    mask = mask_no_i | (1 << j)
                    in_set.remove(i)
                    in_set.add(j)
                    not_in_set.remove(j)
                    not_in_set.add(i)
                    current_count = new_count
                    improved = True
                    found_improvement = True
                    break
                    
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 - DEADLINE_MARGIN
    
    # Run greedy with different parameters and keep the best
    best_mask = 0
    best_count = -1
    
    # Multiple restarts with different conflict weights and noise
    params = [
        (0.1, 0.0),
        (0.05, 0.0),
        (0.2, 0.0),
        (0.1, 0.5),
        (0.1, 1.0),
    ]
    
    for cw, noise in params:
        if time.time() > deadline - 5:
            break
            
        mask = _greedy_construct(deadline, cw, noise)
        
        # Local search
        mask = _local_search(mask, deadline)
        
        count = _count_3aps(mask)
        
        if count > best_count:
            best_count = count
            best_mask = mask
            
    # Convert mask to list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> x) & 1:
            result.append(x)
            
    # Final validity check
    A_set = set(result)
    for a in A_set:
        for d in range(1, N + 1):
            if (a+d) in A_set and (a+2*d) in A_set and (a+3*d) in A_set:
                # Fallback
                return [1]
                
    return result

# Entry point
if __name__ == "__main__":
    construct()
