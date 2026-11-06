import time

N = 17
TIME_LIMIT = 85.0 # Safety margin

def construct():
    start_time = time.time()
    
    # 1. Generate Seed: Odd-Even Transposition
    # Phases: 0, 1, ..., N-1
    # Phase k: compare (i, i+1) for i = k%2, k%2+2, ...
    seed_net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_net.append((i, i + 1))
    
    # 2. Precompute Targets
    # target[m] is the sorted bitmask for input m
    # popcount is number of 1s.
    # Sorted state has popcount 1s at the top (high indices).
    # So target = ((1 << popcount) - 1) << (N - popcount)
    targets = [0] * (1 << N)
    for m in range(1 << N):
        pc = bin(m).count('1')
        targets[m] = ((1 << pc) - 1) << (N - pc)
        
    # 3. Validity Checker
    def is_valid(net):
        # net is a list of (i, j)
        for m in range(1 << N):
            state = m
            target = targets[m]
            # If already sorted, skip
            if state == target:
                continue
            for i, j in net:
                # Check if swap needed
                # Bit i is 1, Bit j is 0
                if (state >> i) & 1 > (state >> j) & 1:
                    state ^= (1 << i) | (1 << j)
                    # Early exit if sorted
                    if state == target:
                        break
            if state != target:
                return False
        return True

    # 4. Greedy Removal
    current_net = seed_net[:]
    
    # We will iterate multiple times
    # Each pass tries to remove one comparator
    # We shuffle to be random
    
    import random
    random.seed(42) # Deterministic for reproducibility? Or just use system time?
    # Let's use a fixed seed for stability in the "island" context if needed, 
    # but random is fine.
    
    while True:
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
            
        removed_any = False
        # Shuffle indices to avoid always checking from 0
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
                
            # Try removing current_net[idx]
            # Create new net without idx
            # Optimization: Instead of copying list, we can just pass a flag?
            # But is_valid iterates over the list.
            # Copying a list of 100 items is fast.
            
            test_net = current_net[:idx] + current_net[idx+1:]
            
            if is_valid(test_net):
                # Accept removal
                current_net = test_net
                removed_any = True
                # Restart the loop (or break inner and let outer continue)
                # Breaking inner and continuing outer is fine, 
                # but the next outer iteration will shuffle again.
                # Actually, if we remove one, the indices shift.
                # So we should break the inner loop and restart the outer.
                break
        
        if not removed_any:
            # No more removals possible with this strategy
            # We could try adding/swapping, but removal is primary.
            # If we are stuck, we might try a "kick" (add a random comparator then remove)
            # But let's stick to simple removal first.
            break

    return current_net
