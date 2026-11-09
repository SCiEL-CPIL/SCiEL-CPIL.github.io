import time
import random
import numpy as np

N = 16

def _generate_bitonic_network(n):
    """Generates a Batcher's Bitonic Sorting Network for n inputs."""
    comparators = []
    
    def bitonic_sort(start, step, size):
        if size == 2:
            comparators.append((start, start + step))
            return
        
        half = size // 2
        bitonic_sort(start, step, half)
        bitonic_sort(start + step * half, step, half)
        bitonic_merge(start, step, size)
        
    def bitonic_merge(start, step, size):
        # size is the total number of wires in this merge stage
        # step is the distance between corresponding wires in the two halves
        # The two halves are sorted in opposite directions
        k = size // 2
        while k > 0:
            for i in range(k):
                # Compare wires that are k apart
                # Indices: start + i*step and start + (i+k)*step
                i1 = start + i * step
                i2 = start + (i + k) * step
                comparators.append((i1, i2))
            k //= 2
            
    bitonic_sort(0, 1, n)
    return comparators

def _verify_network(comparators, n):
    """
    Verifies if the network sorts all 2^n inputs using the Zero-One Principle.
    Uses NumPy for speed.
    Returns True if valid, False otherwise.
    """
    if not comparators:
        return n <= 1
        
    # Generate all 2^n inputs
    # Shape: (2^n, n)
    inputs = np.array(list(np.ndindex(*([2] * n))), dtype=np.uint8)
    # This creates a grid of all binary vectors.
    # np.ndindex yields tuples. We need to convert to array.
    # More efficient:
    inputs = np.zeros((2**n, n), dtype=np.uint8)
    for i in range(n):
        inputs[:, i] = (np.arange(2**n) >> i) & 1
        
    current = inputs.copy()
    
    for i, j in comparators:
        # Compare and exchange
        # min goes to i, max goes to j
        # Since inputs are 0/1, min is 0 if any is 0, max is 1 if any is 1.
        # Actually, for 0/1:
        # If current[:, i] == 1 and current[:, j] == 0, swap.
        # Or simply:
        # new_i = min(a, b) = a & b
        # new_j = max(a, b) = a | b
        a = current[:, i]
        b = current[:, j]
        current[:, i] = a & b
        current[:, j] = a | b
        
    # Check if sorted
    # A sequence is sorted if x[0] <= x[1] <= ... <= x[n-1]
    # For 0/1, this means all 0s come before all 1s.
    # So, for each row, once we see a 1, all subsequent must be 1.
    # Or simply: diff >= 0?
    # Check: current[:, 1:] >= current[:, :-1]
    diffs = current[:, 1:] - current[:, :-1]
    if np.any(diffs < 0):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # 1. Generate initial network
    # Bitonic sort for N=16 uses 80 comparators
    network = _generate_bitonic_network(N)
    
    # 2. Verify initial network (should be true)
    if not _verify_network(network, N):
        # Fallback to odd-even transposition if bitonic generation failed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    best_network = network
    best_count = len(network)
    
    # 3. Greedy Removal Search
    # We try to remove comparators one by one.
    # To avoid getting stuck in a local minimum, we can shuffle the order of attempts.
    # However, simple sequential removal is often effective.
    # We will loop through the network and try to remove each one.
    # If successful, we continue with the shorter network.
    
    # To be more effective, we can do multiple passes or randomize.
    # Given the time budget, we'll do a randomized greedy removal.
    
    # Create a list of indices to try
    # We'll maintain the current list of comparators.
    current_net = list(network)
    
    # We can also try to find redundant comparators more efficiently.
    # But simple removal is robust.
    
    # Randomize the order of comparators to try first
    indices = list(range(len(current_net)))
    
    # Loop until time runs out or no more removals possible
    # We'll use a "shuffled" approach: pick a random comparator, try to remove.
    # If it fails, put it back.
    # If it succeeds, keep it removed.
    # To avoid infinite loops on the same failing comparator, we can track failures.
    # Or just shuffle the list of comparators at each "pass".
    
    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        # Shuffle the current network indices to try different orders
        # Actually, we should try to remove from the *current* network.
        # Let's iterate through the current network in a random order.
        
        current_indices = list(range(len(current_net)))
        random.shuffle(current_indices)
        
        removed_any = False
        for idx in current_indices:
            if time.time() > deadline:
                break
                
            # Try removing current_net[idx]
            # Create a new network without this comparator
            test_net = current_net[:idx] + current_net[idx+1:]
            
            # Verify
            if _verify_network(test_net, N):
                # Success!
                current_net = test_net
                best_network = current_net
                best_count = len(current_net)
                removed_any = True
                # Since we removed one, the indices shift.
                # We break the inner loop to re-shuffle/re-index?
                # Or we can just continue, but indices are now invalid.
                # Safer to break and restart the pass with the new network.
                break
                
        if not removed_any:
            # No more single removals possible
            # We can try local search (swap) if time permits, 
            # but for now, we stop if we can't remove any.
            # To allow for more exploration, we could try swapping two comparators.
            # But swapping doesn't reduce count. We only care about count.
            # So if we can't remove, we are stuck.
            break
            
    return best_network

# Quick self-test if run directly
if __name__ == "__main__":
    net = construct()
    print(f"Comparators: {len(net)}")
    print(f"Valid: {_verify_network(net, N)}")
