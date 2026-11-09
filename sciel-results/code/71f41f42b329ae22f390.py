import time
import random

N = 16

def _apply_comparator(state, i, j):
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi < bj:
        state |= (1 << i)
        state &= ~(1 << j)
    return state

def _verify(network):
    for mask in range(1 << N):
        state = mask
        for i, j in network:
            state = _apply_comparator(state, i, j)
        # Check if sorted: all 0s then 1s
        # If state is sorted, it must be of form 0...01...1
        # This is equivalent to: state == 0 or state == ((1 << N) - 1) or 
        # state has a contiguous block of 1s at the end.
        # A faster check: 
        # If state is sorted, then for all k, if bit k is 0, all bits < k are 0? No.
        # Sorted ascending: x0 <= x1 <= ... <= x15.
        # In binary mask, 0 is low, 1 is high.
        # So if x_k is 1, all x_j for j>k must be 1? No, x_k <= x_{k+1}.
        # So if x_k is 1, x_{k+1} must be 1.
        # If x_k is 0, x_{k-1} must be 0.
        # So the 1s must be at the top (high indices).
        # Check: 
        # Let k be the number of 1s.
        # The state must be equal to ((1 << k) - 1) << (N - k).
        # Or simpler: 
        # state == 0: True
        # state == (1 << N) - 1: True
        # Else: 
        # Find lowest set bit.
        # low = (state & -state).bit_length() - 1
        # The bits from low to N-1 must be 1.
        # mask = ((1 << N) - 1) ^ ((1 << low) - 1)
        # return state == mask
        
        # Even faster:
        # A number is sorted if it has no "10" pattern in its binary representation 
        # when read from LSB to MSB? 
        # x0 x1 ... x15.
        # x_i <= x_{i+1}.
        # If x_i=1 and x_{i+1}=0, it's invalid.
        # So we check if `state & (state >> 1)` has any bits set?
        # If state has a 1 at i and 0 at i+1, then bit i of state is 1, bit i+1 is 0.
        # state >> 1 has bit i-1 = state bit i.
        # This logic is tricky.
        
        # Let's stick to the low-bit check, it's fast enough.
        if state == 0:
            continue
        if state == (1 << N) - 1:
            continue
        low = (state & -state).bit_length() - 1
        mask = ((1 << N) - 1) ^ ((1 << low) - 1)
        if state != mask:
            return False
    return True

def _build_oms(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    k = n // 2
    left = _build_oms(k)
    right = _build_oms(k)
    comps = left + [(i + k, j + k) for i, j in right]
    for i in range(k):
        comps.append((i, i + k))
    
    even_pos = list(range(0, n, 2))
    odd_pos = list(range(1, n, 2))
    
    even_sub = _build_oms(k)
    odd_sub = _build_oms(k)
    
    for i, j in even_sub:
        comps.append((even_pos[i], even_pos[j]))
    for i, j in odd_sub:
        comps.append((odd_pos[i], odd_pos[j]))
    return comps

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Start with a known valid network. 
    # The 80-comparator one from the prompt is good. 
    # Let's use it as the initial state.
    initial_network = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    ]
    
    # Verify initial
    if not _verify(initial_network):
        # Fallback to OMS if initial is broken
        return _build_oms(N)
        
    best_network = initial_network[:]
    best_size = len(best_network)
    
    # Local Search: Random Removal
    # We try to remove comparators one by one.
    # If stuck, we try to swap two comparators to change the structure, 
    # then try removals again.
    
    current_network = initial_network[:]
    
    while time.time() < deadline:
        # Try to remove a random comparator
        if len(current_network) == 0:
            break
            
        # Pick a random index
        idx = random.randint(0, len(current_network) - 1)
        comp_to_remove = current_network[idx]
        
        # Remove it
        new_network = current_network[:idx] + current_network[idx+1:]
        
        # Verify
        if _verify(new_network):
            current_network = new_network
            if len(current_network) < best_size:
                best_size = len(current_network)
                best_network = current_network[:]
        else:
            # Rejection. 
            # To escape local minima, occasionally perform a "swap" move
            # that keeps the network valid but changes the order/structure.
            # This allows different comparators to become redundant.
            if random.random() < 0.1:
                # Swap two random comparators
                if len(current_network) > 1:
                    i1 = random.randint(0, len(current_network) - 1)
                    i2 = random.randint(0, len(current_network) - 1)
                    if i1 != i2:
                        current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
                        # Check if still valid (swapping disjoint comparators is always valid? 
                        # No, comparators interact. But swapping two comparators 
                        # in a valid network doesn't necessarily keep it valid 
                        # if they are not independent. 
                        # Actually, if the network was valid, swapping two comparators 
                        # might break it. We must verify.)
                        if not _verify(current_network):
                            current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
                            # Revert
                        else:
                            # Accept swap, continue trying to remove
                            pass
                            
    return best_network
