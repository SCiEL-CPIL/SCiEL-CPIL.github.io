import numpy as np
import time
import random
import sys

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Initial odd-even transposition network: 120 comparators
    initial_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    # Target: sorted 0s then 1s.
    # If k ones, target is ((1 << k) - 1) << (N - k)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)
    # Note: if k=0, (1<<0)-1 = 0. 0 << 16 = 0. Correct.
    # if k=16, (1<<16)-1 = 65535. 65535 << 0 = 65535. Correct.

    # Precompute bit masks for each wire position
    bit_masks = [1 << i for i in range(N)]
    
    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = bit_masks[i]
            bit_j = bit_masks[j]
            # Condition: bit i is 1, bit j is 0
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    def verify_prefix(net, prefix_len):
        """Verify if the first prefix_len comparators of net are valid."""
        masks = inputs.copy()
        for i, j in net[:prefix_len]:
            bit_i = bit_masks[i]
            bit_j = bit_masks[j]
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    # Verify initial
    if not verify(initial_net):
        # Fallback to seed if something went wrong (should not happen)
        return initial_net

    current_net = list(initial_net)
    best_net = list(initial_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 58.0 # Safety margin
    
    random.seed(42) # Reproducibility
    
    # Greedy removal: try to remove each comparator in order, keeping the network valid
    # This is more systematic than random removal
    i = 0
    while i < len(current_net):
        if time.time() - start_time > deadline:
            break
        
        # Try removing comparator at position i
        candidate = current_net[:i] + current_net[i+1:]
        if verify(candidate):
            current_net = candidate
            if len(current_net) < best_len:
                best_net = current_net
                best_len = len(current_net)
            # Don't increment i since the next comparator shifted into position i
        else:
            i += 1
    
    # Now do random local search to escape local minima
    while time.time() - start_time < deadline:
        # Strategy: Random removal with restart
        if len(current_net) > 1:
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
        
        # Occasionally try swapping to escape local minima
        if len(current_net) > 1 and random.random() < 0.3:
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                candidate = current_net.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify(candidate):
                    current_net = candidate
        
        # Occasionally try inserting a new comparator to enable future removals
        if len(current_net) > 1 and random.random() < 0.1:
            # Pick a random position to insert
            insert_pos = random.randint(0, len(current_net))
            # Pick a random comparator to insert
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i < new_j:
                candidate = current_net.copy()
                candidate.insert(insert_pos, (new_i, new_j))
                if verify(candidate):
                    current_net = candidate
                    # Try to remove the newly inserted comparator or others
                    for _ in range(5):
                        if time.time() - start_time > deadline:
                            break
                        if len(current_net) > 1:
                            idx = random.randint(0, len(current_net) - 1)
                            candidate2 = current_net[:idx] + current_net[idx+1:]
                            if verify(candidate2):
                                current_net = candidate2
                                if len(current_net) < best_len:
                                    best_net = current_net
                                    best_len = len(current_net)
        
        # Occasionally try a small perturbation: swap adjacent comparators
        if len(current_net) > 2 and random.random() < 0.2:
            i1 = random.randint(0, len(current_net) - 2)
            candidate = current_net.copy()
            candidate[i1], candidate[i1+1] = candidate[i1+1], candidate[i1]
            if verify(candidate):
                current_net = candidate
        
        # Occasionally try to move a comparator to a different position
        if len(current_net) > 3 and random.random() < 0.1:
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                candidate = current_net.copy()
                comp = candidate.pop(i1)
                candidate.insert(i2, comp)
                if verify(candidate):
                    current_net = candidate
        
        # Occasionally restart from best_net to explore different regions
        if random.random() < 0.05:
            current_net = best_net.copy()
    
    return best_net
# EVOLVE-BLOCK-END
