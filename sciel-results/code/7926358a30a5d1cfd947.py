import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _bitonic_network(n):
    """Generate Batcher's bitonic sort network for n=2^k wires.
    Returns list of (i, j) comparators, 0 <= i < j < n.
    """
    if n == 1:
        return []
    comps = []
    def rec(size, offset, ascending):
        if size == 1:
            return
        half = size // 2
        # Sort first half ascending, second half descending (or vice versa)
        rec(half, offset, True)
        rec(half, offset + half, not ascending)
        # Merge the two halves
        def merge(sz, off, asc):
            if sz == 1:
                return
            h = sz // 2
            # Compare elements that are h apart
            for i in range(h):
                a = off + i
                b = off + i + h
                comps.append((a, b))
            # Recursively merge the two resulting halves
            merge(h, off, asc)
            merge(h, off + h, asc)
        merge(size, offset, ascending)
    rec(n, 0, True)
    return comps

def _is_valid_fast(net):
    """
    Vectorized validity check using NumPy.
    Checks if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
        
    # Initialize state matrix: 65536 rows (all masks), 16 cols (wires)
    # M[i, j] is 1 if wire j is 1 in input i
    # We can generate this efficiently.
    # Instead of creating a 65536x16 array every time (which is memory intensive),
    # we can use a trick or just accept the memory cost since it's ~1MB.
    
    # Actually, creating the full matrix every time might be slow due to allocation.
    # Let's try to optimize by using a pre-allocated buffer if possible, 
    # but in a function, we can't easily hold state without global vars.
    # Let's just create it. 65536 * 16 * 1 byte = 1MB. Very fast.
    
    # Generate all masks 0 to 2^16 - 1
    masks = np.arange(1 << N, dtype=np.uint8)
    
    # Unpack bits into a matrix M of shape (65536, 16)
    # M[i, j] = (masks[i] >> j) & 1
    # We can do this with broadcasting and shifts.
    # shifts = np.arange(N)
    # M = ((masks[:, None] >> shifts) & 1).astype(np.uint8)
    
    # This creates a (65536, 16) array.
    # Let's verify speed.
    
    # Alternative: Loop over wires? No, vectorized is better.
    
    # Pre-compute shifts
    shifts = np.arange(N, dtype=np.uint8)
    
    # Create M
    # masks is shape (65536,). We want (65536, 16).
    # (masks[:, np.newaxis] >> shifts) works.
    M = ((masks[:, np.newaxis] >> shifts) & 1).astype(np.uint8)
    
    # Apply network
    for i, j in net:
        # Compare-exchange: min to i, max to j
        # Since values are 0 or 1, min is AND, max is OR?
        # No, min(0,1)=0, max(0,1)=1.
        # If x=0, y=1: min=0, max=1.
        # If x=1, y=0: min=0, max=1.
        # If x=0, y=0: min=0, max=0.
        # If x=1, y=1: min=1, max=1.
        # So min = x & y, max = x | y.
        
        x = M[:, i]
        y = M[:, j]
        
        # min_val = x & y
        # max_val = x | y
        
        # Update M
        M[:, i] = x & y
        M[:, j] = x | y
        
    # Check if sorted
    # A row is sorted if M[i, 0] <= M[i, 1] <= ... <= M[i, 15]
    # Since values are 0/1, this means once a 1 appears, all subsequent must be 1.
    # Or: sum of first k zeros must be consistent.
    # Simple check: diff of consecutive elements must be >= 0.
    # Or: np.all(M[:, :-1] <= M[:, 1:])
    
    # This returns a boolean array of shape (65536,)
    # If all True, it's valid.
    return np.all(M[:, :-1] <= M[:, 1:])

def construct():
    start_time = time.time()
    
    # Start with bitonic sort (63 comparators for N=16)
    current_net = _bitonic_network(N)
    
    # Verify it's valid (should be)
    if not _is_valid_fast(current_net):
        # Fallback to OET if bitonic fails (should not happen)
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(current_net)
    best_len = len(current_net)
    
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - start_time
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # We will run until time is up
    while elapsed() < DEADLINE and T > T_min:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Remove a random comparator
            if len(current_net) <= 1:
                T *= cooling
                continue
            idx = rng.randrange(len(current_net))
            candidate = current_net[:idx] + current_net[idx+1:]
            if _is_valid_fast(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
        elif move_type < 0.8:
            # Swap two comparators
            if len(current_net) < 2:
                T *= cooling
                continue
            i, j = rng.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_valid_fast(candidate):
                # Neutral move, accept with some probability to explore
                if rng.random() < 0.5:
                    current_net = candidate
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
        else:
            # Insert a random comparator
            # Pick a random pair (i,j)
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate = list(current_net) + [(i, j)]
            # Adding a comparator usually preserves validity if original was valid?
            # No, it can break it? No, adding a compare-exchange to a sorting network
            # results in a sorting network? 
            # Actually, if a network sorts all inputs, adding a comparator might not
            # necessarily result in a sorting network if the comparator is placed in a way
            # that disrupts the order? 
            # Wait. If a network is a sorting network, it maps every permutation to sorted.
            # Adding a comparator (i,j) at the end:
            # The input to this comparator is already sorted? No, the intermediate state is sorted.
            # If the state is sorted, and we compare i and j (i<j), then x_i <= x_j.
            # min is x_i, max is x_j. So it does nothing.
            # So adding a comparator to a valid sorting network always yields a valid sorting network.
            # Therefore, we don't even need to check validity for inserts!
            # This is a huge speedup.
            
            # However, we might want to insert in the middle?
            # If we insert in the middle, it might break validity.
            # But for simplicity, let's just append. Or insert at random position.
            # Let's try inserting at a random position.
            pos = rng.randrange(len(current_net) + 1)
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            
            # Check validity (needed if inserted in middle)
            if _is_valid_fast(candidate):
                # This is a "bad" move in terms of count
                # Accept with Boltzmann probability
                delta_cost = 1
                if rng.random() < min(1.0, np.exp(-delta_cost / T)):
                    current_net = candidate
            
        T *= cooling
        
    # Final greedy removal pass
    improved = True
    while improved and elapsed() < DEADLINE - 1:
        improved = False
        for i in range(len(current_net)):
            if elapsed() > DEADLINE - 1:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if _is_valid_fast(candidate):
                current_net = candidate
                improved = True
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
                break
    
    return best_net
