import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Pre-compute nothing, just run many random greedy trials
    # Each trial is O(N^2) or O(N * bit_ops)
    # N=100, so very fast.
    
    # To ensure we don't miss the seed's deterministic behavior, 
    # include the reverse order as one of the trials.
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101))
    ]
    
    # Add some random shuffles
    # We can run as many as time permits
    # Estimate: 1 trial ~ 100 * 200 bit ops ~ negligible.
    # We can easily run 100,000 trials in < 1 second.
    
    # Let's run 10,000 random trials plus the deterministic ones.
    # If time remains, we can do local search or more trials.
    
    def try_order(order):
        current = []
        # Use a bitset for reachable sums up to 200
        # Python ints are arbitrary precision, so we can just use an int mask.
        # Bit i is 1 if sum i is reachable.
        reachable = 1  # bit 0 set
        
        for x in order:
            # Check if adding x creates a sum of 200
            # This happens if (200 - x) is currently reachable
            target = 200 - x
            if target < 0:
                # x > 200, always safe (but max is 100)
                pass
            elif (reachable >> target) & 1:
                # Conflict, skip
                continue
            else:
                # Safe to add
                current.append(x)
                # Update reachable sums
                # New sums are old sums + x
                # We only care about sums <= 200
                new_reachable = reachable | (reachable << x)
                # Mask to keep only bits 0..200
                # Mask with (1 << 201) - 1
                reachable = new_reachable & ((1 << 201) - 1)
        
        return current

    # 1. Deterministic checks
    for order in orders:
        s = try_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 2. Randomized checks
    # Run until time limit or a fixed high number of iterations
    # Since each is fast, we run many.
    # 100,000 iterations should take < 1s.
    # Let's do 50,000 to be safe, then check time.
    
    rng = random.Random(42)
    
    # We can generate shuffles on the fly
    base_list = list(range(1, 101))
    
    # Run in batches to check time
    batch_size = 1000
    max_iterations = 200000
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        rng.shuffle(base_list)
        s = try_order(base_list)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        # Optional: Simple local search improvement?
        # If we find a new best, we could try to improve it.
        # But randomized greedy is usually sufficient for this size.
        
    return best_set
