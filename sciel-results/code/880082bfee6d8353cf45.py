import numpy as np
import time
import random

N = 17

def construct():
    # Precompute masks and initial bits for verification
    # We will use a helper function for fast verification
    
    def verify(network):
        # Fast verification using NumPy
        # 131,072 inputs
        masks = np.arange(1 << N, dtype=np.uint32)
        bits = (masks[:, None] >> np.arange(N, dtype=np.uint8)) & 1
        
        for i, j in network:
            bi = bits[:, i]
            bj = bits[:, j]
            # Swap if bi=1 and bj=0
            swap_mask = (bi == 1) & (bj == 0)
            if np.any(swap_mask):
                bits[swap_mask, i] = 0
                bits[swap_mask, j] = 1
        
        # Check if sorted: 0...01...1
        # This means no 1 followed by 0
        # Check diffs
        # If any diff is -1, it's not sorted
        diffs = bits[:, 1:].astype(np.int8) - bits[:, :-1].astype(np.int8)
        return not np.any(diffs == -1)

    # Initial network: Odd-Even Transposition (Bubble Sort Network)
    # This has 136 comparators
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    best_network = list(network)
    best_len = len(network)
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s for safety
    
    while time.time() - start_time < timeout:
        current_network = list(best_network)
        
        # Try to remove comparators greedily in random order
        # We will iterate through indices in random order
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            # Try to remove current_network[idx]
            # Note: indices might be invalid if we removed previous ones, 
            # but we are iterating over a fixed list of indices from the start of this pass.
            # We need to be careful: if we remove one, the list shrinks.
            # So we should check if idx is still valid.
            if idx >= len(current_network):
                continue
                
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if verify(candidate):
                current_network = candidate
                removed_any = True
                # If we removed one, the indices for subsequent items in this pass are shifted.
                # It's safer to restart the pass or just continue and let invalid indices be skipped.
                # Actually, if we remove item at idx, all items after idx shift left by 1.
                # The `indices` list contains the original indices.
                # So if we removed item 5, item 6 is now at 5.
                # The next index in `indices` might be 6, which now points to the old item 7.
                # This is a valid strategy (randomized greedy).
        
        if len(current_network) < best_len:
            best_network = current_network
            best_len = len(current_network)
            
        # If we didn't remove anything, we are stuck in a local minimum.
        # Try shuffling the best network to see if a different order allows removals.
        if not removed_any:
            # Shuffle the best network
            shuffled = list(best_network)
            random.shuffle(shuffled)
            # Try to remove from the shuffled version
            # This might find a different local minimum
            temp_network = list(shuffled)
            temp_indices = list(range(len(temp_network)))
            random.shuffle(temp_indices)
            
            for idx in temp_indices:
                if time.time() - start_time > timeout:
                    break
                if idx >= len(temp_network):
                    continue
                candidate = temp_network[:idx] + temp_network[idx+1:]
                if verify(candidate):
                    temp_network = candidate
            
            if len(temp_network) < best_len:
                best_network = temp_network
                best_len = len(temp_network)
            else:
                # If shuffling didn't help, we might be close to the limit for this seed.
                # We could try a more aggressive restart, but we don't have a better seed.
                # We will just continue the loop, but it will likely keep failing.
                # To avoid infinite loop of failures, we can break if we've tried several shuffles.
                # For now, the time limit will stop us.
                pass

    return best_network
