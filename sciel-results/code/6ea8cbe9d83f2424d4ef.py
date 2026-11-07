import sys

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Approach: 
    1. Greedy initialization to find a good lower bound.
    2. Branch and Bound to search for a better solution.
    """
    
    n = 256
    best_set = []
    best_size = 0
    node_count = 0
    max_nodes = 50000  # Limit to ensure speed
    
    def can_add(current_sums_by_card, x, max_k):
        new_sums = [set() for _ in range(max_k + 2)]
        new_sums[0] = {0}
        
        for k in range(1, max_k + 2):
            if k <= max_k:
                new_sums[k].update(current_sums_by_card[k])
            if k - 1 <= max_k:
                for s in current_sums_by_card[k - 1]:
                    new_sums[k].add(s + x)
        
        for k in range(max_k + 2):
            for j in range(k + 1, max_k + 2):
                if new_sums[k] & new_sums[j]:
                    return False, None
        
        return True, new_sums

    # 1. Greedy Initialization
    sums_by_card = [set() for _ in range(61)]
    sums_by_card[0] = {0}
    greedy_set = []
    max_k = 0
    for x in range(1, n + 1):
        if max_k >= 60:
            break
        ok, new_sums = can_add(sums_by_card, x, max_k)
        if ok:
            greedy_set.append(x)
            sums_by_card = new_sums
            max_k += 1
            
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    # 2. Branch and Bound
    def search(idx, current_set, sums_by_card, max_k):
        nonlocal best_set, best_size, node_count
        
        if node_count > max_nodes:
            return
            
        node_count += 1
        
        remaining = n - idx + 1
        if len(current_set) + remaining <= best_size:
            return
            
        if idx > n:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        x = idx
        
        # Option 1: Add x
        ok, new_sums = can_add(sums_by_card, x, max_k)
        if ok:
            current_set.append(x)
            search(idx + 1, current_set, new_sums, max_k + 1)
            current_set.pop()
            
        # Option 2: Skip x
        if len(current_set) + remaining - 1 > best_size:
            search(idx + 1, current_set, sums_by_card, max_k)

    # Start B&B from the greedy state? No, start from scratch to explore all, 
    # but use the greedy result as the initial best bound.
    # To save time, we can start B&B from index 1.
    initial_sums = [set() for _ in range(61)]
    initial_sums[0] = {0}
    
    search(1, [], initial_sums, 0)
    
    return [int(x) for x in best_set]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Set: {result}")
