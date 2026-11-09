import time
import random
import numpy as np

N = 512
TIME_LIMIT = 55  # seconds, safety margin before 60s limit

def construct():
    start_time = time.time()
    
    # Precompute all 4-term arithmetic progressions in [1, 512]
    # For each element x, store the other 3 elements in each 4-AP containing x
    aps_containing = {i: [] for i in range(1, N + 1)}
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))
    
    # Convert to numpy arrays for faster vectorized checking
    aps_array = {}
    for x in range(1, N + 1):
        if aps_containing[x]:
            aps_array[x] = np.array(aps_containing[x], dtype=np.int32)
        else:
            aps_array[x] = np.empty((0, 3), dtype=np.int32)
    
    def can_add_vectorized(x, current_bits):
        """Check if adding x creates a 4-AP using numpy vectorization"""
        aps = aps_array[x]
        if len(aps) == 0:
            return True
        bits_a = current_bits[aps[:, 0]]
        bits_b = current_bits[aps[:, 1]]
        bits_c = current_bits[aps[:, 2]]
        if np.any(bits_a & bits_b & bits_c):
            return False
        return True
    
    def greedy_randomized():
        """Greedy with random ordering, returns numpy bool array"""
        current_bits = np.zeros(N + 1, dtype=bool)
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if can_add_vectorized(x, current_bits):
                current_bits[x] = True
        return current_bits
    
    def local_search(current_bits):
        """
        Optimized local search. Uses NumPy for candidate selection.
        Avoids expensive full refills; uses targeted swaps/additions.
        """
        current_bits = current_bits.copy()
        current_size = int(np.sum(current_bits))
        best_bits = current_bits.copy()
        best_size = current_size
        
        # Precompute indices for faster random selection
        all_indices = np.arange(1, N + 1)
        
        while time.time() - start_time < TIME_LIMIT - 5:
            # Try to add a random element
            # Get indices of elements not in set
            not_in_set = np.where(~current_bits[1:])[0] + 1
            if len(not_in_set) == 0:
                break # Set is full (unlikely for 4-AP free)
            
            x = int(random.choice(not_in_set))
            
            if can_add_vectorized(x, current_bits):
                current_bits[x] = True
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_bits = current_bits.copy()
            else:
                # If we can't add x, try to swap: remove one blocker and add x
                aps = aps_array[x]
                if len(aps) > 0:
                    bits_a = current_bits[aps[:, 0]]
                    bits_b = current_bits[aps[:, 1]]
                    bits_c = current_bits[aps[:, 2]]
                    blocking = bits_a & bits_b & bits_c
                    if np.any(blocking):
                        blocking_idx = np.where(blocking)[0]
                        idx = random.choice(blocking_idx)
                        # The AP is (aps[idx,0], aps[idx,1], aps[idx,2], x) or some permutation
                        # Actually aps_containing[x] stores the OTHER 3.
                        # So the AP is {x, aps[idx,0], aps[idx,1], aps[idx,2]}
                        remove_choices = [int(aps[idx, 0]), int(aps[idx, 1]), int(aps[idx, 2])]
                        remove_x = random.choice(remove_choices)
                        
                        if current_bits[remove_x]:
                            trial_bits = current_bits.copy()
                            trial_bits[remove_x] = False
                            if can_add_vectorized(x, trial_bits):
                                trial_bits[x] = True
                                # Size remains same
                                # Accept swap with some probability to escape local optima
                                if random.random() < 0.5:
                                    current_bits = trial_bits
                                    # Size is same, so best_size doesn't change unless we found a better one later
                                    # But we might have freed up space for future additions
                                    
            # Occasionally, try a "kick" by removing a random element to allow different structure
            if random.random() < 0.1 and current_size > 0:
                in_set = np.where(current_bits[1:])[0] + 1
                if len(in_set) > 0:
                    remove_x = int(random.choice(in_set))
                    current_bits[remove_x] = False
                    current_size -= 1
                    # Try to add several random elements
                    for _ in range(10):
                        not_in_set = np.where(~current_bits[1:])[0] + 1
                        if len(not_in_set) == 0: break
                        y = int(random.choice(not_in_set))
                        if can_add_vectorized(y, current_bits):
                            current_bits[y] = True
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_bits = current_bits.copy()
        
        return best_bits
    
    best_bits = None
    best_size = 0
    
    # Run multiple restarts
    for i in range(300):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        bits = greedy_randomized()
        bits = local_search(bits)
        size = int(np.sum(bits))
        
        if size > best_size:
            best_size = size
            best_bits = bits.copy()
    
    # Also try starting from odd numbers
    if time.time() - start_time < TIME_LIMIT - 10:
        odd_bits = np.zeros(N + 1, dtype=bool)
        odd_bits[1::2] = True
        # Remove 4-APs from odd set
        for d in range(1, (N - 1) // 3 + 1):
            for a in range(1, N - 3 * d + 1, 2):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if odd_bits[a] and odd_bits[b] and odd_bits[c] and odd_bits[e]:
                    odd_bits[e] = False
        bits = local_search(odd_bits)
        size = int(np.sum(bits))
        if size > best_size:
            best_size = size
            best_bits = bits.copy()
    
    # Convert to list
    if best_bits is None:
        return [1]
    return [int(x) for x in range(1, N + 1) if best_bits[x]]
