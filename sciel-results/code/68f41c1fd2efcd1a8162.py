import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, 10s buffer
    
    best_set = []
    
    # Strategy 1: Greedy Largest First
    s1 = greedy_construct(256, 1, -1, deadline)
    if len(s1) > len(best_set):
        best_set = s1
        
    # Strategy 2: Greedy Smallest First
    s2 = greedy_construct(1, 256, 1, deadline)
    if len(s2) > len(best_set):
        best_set = s2
        
    # Strategy 3: Randomized Greedy
    for _ in range(1000): # Adjust based on time
        if time.time() > deadline:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        s3 = greedy_construct_from_list(candidates, deadline)
        if len(s3) > len(best_set):
            best_set = s3
            
    return best_set

def greedy_construct(start, end, step, deadline):
    current_set = []
    sums_by_card = [set([0])]
    
    for x in range(start, end + (1 if step > 0 else -1), step):
        if time.time() > deadline:
            break
        if can_add(x, current_set, sums_by_card):
            add_element(x, current_set, sums_by_card)
            
    return current_set

def greedy_construct_from_list(candidates, deadline):
    current_set = []
    sums_by_card = [set([0])]
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if can_add(x, current_set, sums_by_card):
            add_element(x, current_set, sums_by_card)
            
    return current_set

def can_add(x, current_set, sums_by_card):
    # Check if x is already in set (should be handled by caller or here)
    if x in current_set:
        return False
        
    n = len(current_set)
    # New sums will be formed for cardinalities 1 to n+1
    # Specifically, new sums of size k+1 are {s + x | s in sums_by_card[k]}
    
    # We need to check if any new sum collides with existing sums of DIFFERENT cardinality.
    # Let new_sums_k be the set of new sums of size k.
    # Collision if new_sums_k intersects with sums_by_card[j] for j != k.
    
    # Optimization: Pre-compute the union of all existing sums?
    # No, because we need to distinguish which cardinality they belong to for the collision check?
    # Actually, if a sum value V is in S_j and also in new_sums_k (where k != j), it's a collision.
    # So we just need to check if (new_sums_k) intersects with (Union of all S_j for j != k).
    
    # Let's compute the set of all "forbidden" values for each new cardinality k.
    # Forbidden for new S_k = Union(S_j for j != k).
    
    # We can compute the total union of all sums: TotalUnion = Union(S_0 ... S_n)
    # Then Forbidden for new S_k = TotalUnion - S_k?
    # No, S_k is being updated.
    # Let's just iterate.
    
    for k in range(n, -1, -1):
        # New sums of size k+1
        new_sums = set()
        for s in sums_by_card[k]:
            new_sums.add(s + x)
            
        # Check collision with other cardinalities
        # We need to check against all S_j where j != k+1
        for j in range(n + 1):
            if j == k + 1:
                continue
            # Check intersection
            # Optimization: if len(new_sums) is large, iterate smaller set
            if len(new_sums) < len(sums_by_card[j]):
                for v in new_sums:
                    if v in sums_by_card[j]:
                        return False
            else:
                for v in sums_by_card[j]:
                    if v in new_sums:
                        return False
                        
    return True

def add_element(x, current_set, sums_by_card):
    current_set.append(x)
    n = len(current_set) - 1 # Old size
    
    # Update sums
    # We need to update from high cardinality to low to avoid overwriting
    # New S_{k+1} = Old S_{k+1} U {s+x | s in Old S_k}
    
    # Create a copy or update carefully
    # Since we are adding to the end, we can append to the list of sets
    sums_by_card.append(set())
    
    for k in range(n, -1, -1):
        for s in sums_by_card[k]:
            sums_by_card[k+1].add(s + x)
            
    # Note: sums_by_card[k] for k<=n remains unchanged?
    # No, sums of size k do not change when adding an element?
    # Wait. S_k is the set of sums of subsets of size k.
    # If we add x to the set of available elements, the subsets of size k can either include x or not.
    # If they don't include x, the sum is from the old set.
    # If they include x, the sum is s+x where s is a sum of size k-1 from the old set.
    # So New S_k = Old S_k U {s+x | s in Old S_{k-1}}.
    
    # My previous logic in can_add was correct for checking.
    # In add_element, I need to update ALL S_k, not just append.
    
    # Let's re-implement add_element correctly.
    # Old sums_by_card has size n+1 (indices 0..n).
    # New sums_by_card will have size n+2 (indices 0..n+1).
    
    # We need to compute New S_k for all k.
    # New S_k = Old S_k U {s+x | s in Old S_{k-1}}
    
    # It's safer to compute new sets into a new list.
    new_sums_by_card = []
    for k in range(n + 2):
        new_set = set()
        # 1. Subsets not containing x: from Old S_k (if k <= n)
        if k <= n:
            new_set.update(sums_by_card[k])
        # 2. Subsets containing x: from Old S_{k-1} + x (if k-1 >= 0 and k-1 <= n)
        if k > 0 and k-1 <= n:
            for s in sums_by_card[k-1]:
                new_set.add(s + x)
        new_sums_by_card.append(new_set)
        
    sums_by_card[:] = new_sums_by_card
