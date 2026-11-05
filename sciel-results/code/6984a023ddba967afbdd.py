import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 512
    
    # 1. Precompute all 4-APs and mapping from element to AP indices
    ap_list = []
    elem_to_aps = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(ap_idx)
            elem_to_aps[b].append(ap_idx)
            elem_to_aps[c].append(ap_idx)
            elem_to_aps[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # State variables for the current search
    in_set = set()
    count_in_ap = [0] * num_aps
    
    def add_element(x):
        """
        Adds x to the set if it doesn't complete any 4-AP.
        Returns True if added, False otherwise.
        """
        for ap_idx in elem_to_aps[x]:
            if count_in_ap[ap_idx] == 3:
                return False
        in_set.add(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] += 1
        return True

    def remove_element(x):
        """
        Removes x from the set.
        """
        in_set.discard(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] -= 1

    def reset_state():
        """
        Clears the current set and counters.
        """
        in_set.clear()
        for i in range(num_aps):
            count_in_ap[i] = 0

    def get_base3_set_in_range(lo, hi):
        """
        Returns the set of integers in [lo, hi] whose base-3 representation
        contains only digits 0 and 1.
        """
        s = set()
        # 3^5 = 243, 3^6 = 729 > 512, so 6 bits suffice
        for mask in range(1, 1 << 6):
            val = 0
            for i in range(6):
                if mask & (1 << i):
                    val += (3 ** i)
            if lo <= val <= hi:
                s.add(val)
        return s

    def block_based_init(block_size=81, overlap=20):
        """
        Density-incremental construction: partition [1,n] into overlapping blocks,
        apply base-3 construction to each block, then merge via conflict-aware union.
        """
        candidates = set()
        block_starts = []
        pos = 1
        while pos <= n:
            block_starts.append(pos)
            pos += (block_size - overlap)
        
        # Ensure last block covers n
        if block_starts[-1] + block_size - 1 < n:
            block_starts.append(n - block_size + 1)
        
        for start in block_starts:
            end = min(start + block_size - 1, n)
            block_seed = get_base3_set_in_range(start, end)
            candidates.update(block_seed)
        
        # Convert to list and add remaining elements in random order
        remaining = [x for x in range(1, n + 1) if x not in candidates]
        random.shuffle(remaining)
        return list(candidates) + remaining

    def local_search_improve(max_steps=3000):
        """
        Improved local search: prioritize adding elements, then targeted swaps.
        Returns the final set size.
        """
        current_size = len(in_set)
        no_improvement = 0
        steps = 0
        
        # Build list of non-members for efficient sampling
        non_members = [x for x in range(1, n + 1) if x not in in_set]
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            
            if not non_members:
                break
            
            # Try to add a random non-member
            idx = random.randrange(len(non_members))
            x = non_members[idx]
            
            if add_element(x):
                current_size += 1
                non_members.pop(idx)
                no_improvement = 0
                continue
            
            # Find conflicting APs
            conflicting_aps = []
            for ap_idx in elem_to_aps[x]:
                if count_in_ap[ap_idx] == 3:
                    conflicting_aps.append(ap_idx)
            
            if not conflicting_aps:
                continue
            
            # Try swaps: remove one element from a conflicting AP, try to add x
            swap_success = False
            for _ in range(min(3, len(conflicting_aps))):
                ap_idx = random.choice(conflicting_aps)
                ap = ap_list[ap_idx]
                y = random.choice([t for t in ap if t != x])
                
                remove_element(y)
                if add_element(x):
                    # Update non_members
                    if x in non_members:
                        non_members.remove(x)
                    if y not in non_members:
                        non_members.append(y)
                    swap_success = True
                    break
                else:
                    # Restore y
                    add_element(y)
            
            if swap_success:
                no_improvement += 1
            else:
                no_improvement += 1
            
            if no_improvement > 800:
                break
        
        return current_size

    best_size = 0
    best_set = set()
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        reset_state()
        
        # Choose initialization strategy
        if restart_count % 3 == 0:
            # Pure random greedy
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        elif restart_count % 3 == 1:
            # Block-based initialization
            candidates = block_based_init(block_size=81, overlap=20)
        else:
            # Different block sizes for diversity
            candidates = block_based_init(block_size=64, overlap=16)
        
        # Greedy Phase
        for x in candidates:
            if x not in in_set:
                add_element(x)
        
        current_size = len(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(in_set)
        
        # Local Search Phase
        current_size = local_search_improve(max_steps=3000)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(in_set)
        
        # Check if we have enough time for another restart
        if time.time() > deadline - 10:
            break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
