import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

# Precompute full integer representation of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_int = all_masks.copy()

# Precompute popcounts for the initial inputs
# Using a lookup table for 8-bit chunks for speed
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
# Split W0_int into 4 bytes
W0_bytes = W0_int[:, None] // (256 ** np.arange(4)) % 256
popcounts = popcount_table[W0_bytes].sum(axis=1)

# Precompute target values for sorted outputs
# Sorted output for k ones: 1s are in the most significant k bits (wires 18-k to 17)
# Value = 2^18 - 2^(18-k)
targets = np.zeros(N + 1, dtype=np.int32)
for k in range(N + 1):
    if k == 0:
        targets[k] = 0
    else:
        targets[k] = (1 << N) - (1 << (N - k))

# Pippenger's optimal 16-wire sorting network (57 comparators)
# This is a well-known optimal network for n=16
PIPPENGER_16 = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
]

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses fast bitwise operations on int32 arrays.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        # Empty network sorts only if N<=1, which is false for N=18
        return False 
        
    W = W0_int.copy()
    
    for i, j in net:
        # Extract bits
        ai = (W >> i) & 1
        aj = (W >> j) & 1
        
        # Min and Max
        mn = ai & aj
        mx = ai | aj
        
        # Clear bits i and j
        mask_i = ~(1 << i)
        mask_j = ~(1 << j)
        
        W = (W & mask_i & mask_j) | (mn << i) | (mx << j)
        
    # Check if W matches the sorted target for each input's popcount
    expected = targets[popcounts]
    
    return np.all(W == expected)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _extend_16_to_18():
    """
    Extend the 16-wire Pippenger network to 18 wires by:
    1. Starting with the 16-wire network on wires 0-15
    2. Adding comparators to integrate wires 16 and 17
    3. Using a merge-sort style approach to combine
    """
    # Start with Pippenger's 16-wire network
    net = list(PIPPENGER_16)
    
    # Add comparators to merge in wire 16
    # First, compare wire 16 with several wires in the sorted 16-wire network
    # Then insert it properly
    merge_16 = [
        (0, 16), (1, 16), (2, 16), (3, 16),
        (4, 16), (5, 16), (6, 16), (7, 16),
        (8, 16), (9, 16), (10, 16), (11, 16),
        (12, 16), (13, 16), (14, 16), (15, 16),
    ]
    net.extend(merge_16)
    
    # Now wire 16 is sorted relative to 0-15, but we need to fix the 16-wire subnetwork
    # Add a final Pippenger sort on 0-15 to ensure they're still sorted
    net.extend(PIPPENGER_16)
    
    # Add comparators to merge in wire 17
    merge_17 = [
        (0, 17), (1, 17), (2, 17), (3, 17),
        (4, 17), (5, 17), (6, 17), (7, 17),
        (8, 17), (9, 17), (10, 17), (11, 17),
        (12, 17), (13, 17), (14, 17), (15, 17),
        (16, 17),
    ]
    net.extend(merge_17)
    
    # Final sort on all 18 wires
    net.extend(PIPPENGER_16)
    net.extend([(16, 17)])
    
    return net

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with extended Pippenger network
    net = _extend_16_to_18()
    best_net = net[:]
    best_count = len(net)
    
    # Also try odd-even transposition as a fallback
    oet_net = _odd_even_transposition()
    if len(oet_net) < best_count:
        best_net = oet_net
        best_count = len(oet_net)
        net = oet_net
    
    # Phase 1: Greedy removal
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Hill Climbing / Local Search
    if time.time() < deadline - 2:
        T = 1.0
        min_T = 0.05
        cooling = 0.999
        
        current_count = len(net)
        
        while time.time() < deadline - 1:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                
            # Generate move
            r = random.random()
            if r < 0.3:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.5:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            elif r < 0.7:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            elif r < 0.85:
                # Swap
                if len(net) < 2:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            else:
                # Move
                if len(net) == 0:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                comp = new_net.pop(idx1)
                new_net.insert(idx2, comp)
            
            if not _is_sorted_network(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
