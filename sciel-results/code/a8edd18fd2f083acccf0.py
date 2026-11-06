import numpy as np
import time

N = 17

def _generate_batcher_odd_merge(n):
    """
    Generates Batcher's Odd-Merge Sort network for n wires (n must be power of 2).
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    
    comps = []
    # Recursive structure
    # 1. Sort first half and second half
    comps.extend(_generate_batcher_odd_merge(n // 2))
    for i in range(n // 2, n):
        # Shift indices for second half
        # The recursive call above generates indices 0..n/2-1.
        # We need to add n/2 to these indices for the second half.
        pass # This simple recursion doesn't easily handle offsets.
        
    # Better: Implement the merge step explicitly.
    # Batcher's Odd-Merge:
    # 1. Recursively sort two halves.
    # 2. Odd-Even Merge the two halves.
    
    def odd_even_merge(left, right):
        """
        Merges two sorted lists of equal length represented by wire indices.
        left: list of indices for the first half (sorted)
        right: list of indices for the second half (sorted)
        Returns list of comparators to merge them.
        """
        n = len(left)
        if n == 1:
            return [(left[0], right[0])]
        
        comps = []
        # Interleave odd and even elements
        # Odd elements: left[0], right[0], left[2], right[2]... ?
        # Actually, Batcher's Odd-Merge:
        # 1. Compare left[0] with right[0], left[1] with right[1]? No.
        # The merge step:
        # 1. Compare (left[0], right[0]), (left[1], right[1]) ... (left[n/2-1], right[n/2-1])?
        # No, that's not it.
        
        # Standard Odd-Merge Step:
        # 1. Split into odd and even indexed elements.
        # Odd: left[0], right[1], left[2], right[3]... ?
        # Actually, the definition is:
        # Merge(L, R) where L and R are sorted.
        # 1. Compare L[0] with R[0], L[1] with R[1], etc? No.
        
        # Let's use the standard recursive definition:
        # Merge(n):
        # 1. Split L into L_odd, L_even. Split R into R_odd, R_even.
        # 2. Recursively Merge(L_odd, R_odd) and Merge(L_even, R_even).
        # 3. Compare (L_odd[0], L_even[0]), (L_odd[1], L_even[1]) ... ?
        
        # This is getting complicated to implement correctly from memory.
        # Let's use a simpler, known-correct network: **Batcher's Bitonic Sort**.
        # Although Bitonic is 160 for N=16, it is very easy to implement.
        # But 160 is worse than 136.
        
        # Let's stick to the **Search** approach starting from a simple seed.
        # Seed: Odd-Even Transposition (136).
        # We will use the search to improve it.
        
        return []

    return []

# We will not use the broken Batcher implementation above.
# We will use the search approach starting from the seed.

def construct():
    # Seed: Odd-Even Transposition
    seed_net = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Optimization: Local Search
    # Check function using NumPy
    def is_valid(net):
        # Generate all 2^N boolean inputs
        # 2^17 = 131072. N=17.
        # Matrix shape: (131072, 17)
        # Memory: 131072 * 17 * 1 byte (if uint8) = ~2.2 MB.
        
        # Create inputs
        # Using bit shifts
        inputs = (np.arange(2**N, dtype=np.uint8)[:, None] >> np.arange(N, dtype=np.uint8)[None, :]) & 1
        
        # Apply network
        for i, j in net:
            a = inputs[:, i]
            b = inputs[:, j]
            # min to i, max to j
            inputs[:, i] = np.minimum(a, b)
            inputs[:, j] = np.maximum(a, b)
            
        # Check if sorted
        # A sequence is sorted if diff >= 0
        diffs = np.diff(inputs, axis=1)
        return np.all(diffs >= 0)

    current_net = seed_net[:]
    best_net = current_net[:]
    best_len = len(current_net)
    
    start_time = time.time()
    timeout = 50.0 # Leave 10s buffer
    
    # Simple Hill Climbing
    # 1. Try to remove a comparator
    # 2. Try to swap two comparators
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try removal
        if len(current_net) > 0:
            idx = np.random.randint(len(current_net))
            test_net = current_net[:]
            test_net.pop(idx)
            if is_valid(test_net):
                current_net = test_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
                improved = True
        
        # Try swap
        if len(current_net) >= 2:
            idx1 = np.random.randint(len(current_net))
            idx2 = np.random.randint(len(current_net))
            if idx1 != idx2:
                test_net = current_net[:]
                test_net[idx1], test_net[idx2] = test_net[idx2], test_net[idx1]
                if is_valid(test_net):
                    current_net = test_net
                    improved = True
        
        # If no improvement, try a random mutation (replace a comparator with a random valid one)
        if not improved:
            idx = np.random.randint(len(current_net))
            i, j = current_net[idx]
            # Pick a new random pair
            new_i = np.random.randint(N)
            new_j = np.random.randint(N)
            if new_i < new_j:
                test_net = current_net[:]
                test_net[idx] = (new_i, new_j)
                if is_valid(test_net):
                    current_net = test_net
                    if len(current_net) < best_len:
                        best_net = current_net[:]
                        best_len = len(current_net)

    return best_net

# EVOLVE-BLOCK-START
def construct():
    # The logic above is inside the function to keep the interface clean.
    # But we need to ensure it runs fast.
    # The is_valid check is O(2^N * N * K).
    # 131072 * 17 * 136 ~ 300M operations.
    # In NumPy, this is vectorized, so it's fast.
    # Each check might take ~10-20ms.
    # In 50s, we can do ~2500-5000 checks.
    # This should allow some improvement.
    
    import numpy as np
    import time
    
    N = 17
    seed_net = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    def is_valid(net):
        inputs = (np.arange(2**N, dtype=np.uint8)[:, None] >> np.arange(N, dtype=np.uint8)[None, :]) & 1
        for i, j in net:
            a = inputs[:, i]
            b = inputs[:, j]
            inputs[:, i] = np.minimum(a, b)
            inputs[:, j] = np.maximum(a, b)
        diffs = np.diff(inputs, axis=1)
        return np.all(diffs >= 0)

    current_net = seed_net[:]
    best_net = current_net[:]
    best_len = len(current_net)
    
    start_time = time.time()
    timeout = 50.0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try removal
        if len(current_net) > 0:
            idx = np.random.randint(len(current_net))
            test_net = current_net[:]
            test_net.pop(idx)
            if is_valid(test_net):
                current_net = test_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
                improved = True
        
        # Try swap
        if len(current_net) >= 2:
            idx1 = np.random.randint(len(current_net))
            idx2 = np.random.randint(len(current_net))
            if idx1 != idx2:
                test_net = current_net[:]
                test_net[idx1], test_net[idx2] = test_net[idx2], test_net[idx1]
                if is_valid(test_net):
                    current_net = test_net
                    improved = True
        
        if not improved:
            idx = np.random.randint(len(current_net))
            new_i = np.random.randint(N)
            new_j = np.random.randint(N)
            if new_i < new_j:
                test_net = current_net[:]
                test_net[idx] = (new_i, new_j)
                if is_valid(test_net):
                    current_net = test_net
                    if len(current_net) < best_len:
                        best_net = current_net[:]
                        best_len = len(current_net)

    return best_net
# EVOLVE-BLOCK-END
