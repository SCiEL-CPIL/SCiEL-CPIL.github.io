import time
import random

N = 12

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    # Precompute transition masks for each state for both letters
    # mask_a[i] is the bitmask of the destination of state i under letter a
    # mask_b[i] is the bitmask of the destination of state i under letter b
    # Since it's a function, mask is just 1 << trans[i]
    
    # To make it faster, we can precompute the effect of applying a or b to any subset.
    # However, the number of subsets is 2^12 = 4096.
    # We can precompute next_subset[subset][letter] for all subsets?
    # That would be 4096 * 2 entries. Each entry is an integer.
    # This is very fast to compute and makes the BFS very fast.
    
    # Let's do it.
    # subset is an integer from 0 to (1<<N) - 1
    
    # Precompute transition for each subset for letter a and b
    # trans_subset_a[subset] = result of applying a to subset
    # trans_subset_b[subset] = result of applying b to subset
    
    # We can compute these on the fly or precompute.
    # Precomputation cost: 4096 subsets * 2 letters * (popcount operations)
    # popcount is fast.
    
    # Actually, we can compute trans_subset[subset] as follows:
    # For each state i, if i is in subset, then trans_a[i] is in the new subset.
    # new_subset = OR of (1 << trans_a[i]) for all i in subset.
    
    # Let's precompute arrays trans_a_sub and trans_b_sub of size 1<<N.
    size = 1 << N
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    for subset in range(size):
        # If subset is 0, result is 0
        if subset == 0:
            continue
        sa = 0
        sb = 0
        # Iterate through bits
        # We can use bit operations to iterate set bits
        temp = subset
        while temp:
            # Get least significant set bit
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1: # Only state 0
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            # Apply a
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            # Apply b
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 50 # Safety margin
    
    # Initial candidate: A known good structure or random
    # Let's try a structure that tends to have long reset words.
    # Cycle for a, and b collapses slowly.
    
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    # This specific one might not be optimal. Let's initialize with a random one that synchronizes.
    
    # Better initialization:
    # Try a few random initializations and pick the best.
    # Or use a heuristic:
    # a: cycle
    # b: 0 for most, but b(0)=1, b(1)=0? No.
    
    # Let's run a short random search to find a good starting point.
    start_time = time.time()
    
    # Initialize with a "star" automaton which is known to be decent?
    # Or just random.
    
    current_a = [(i + 1) % N for i in range(N)]
    current_b = [0] * N
    
    # Ensure it synchronizes
    # The cycle + sink 0 synchronizes. Length is N-1?
    # 0->1->...->11->0 (a)
    # all->0 (b)
    # Word: b^1? No, b maps everyone to 0. So length 1.
    # We need a better start.
    
    # Let's try:
    # a: cycle
    # b: i -> i+1 for i < N-1, N-1 -> 0?
    # This is just another cycle.
    
    # A good heuristic for high reset length is to have one letter act as a "rotation" and the other as a "shift" that overlaps slowly.
    
    # Let's use a random search for the first 10 seconds to find a local max, then continue refining.
    
    best_len = -1
    best_a = None
    best_b = None
    
    # Try a few specific patterns
    candidates = []
    
    # Pattern 1: Cycle + random sink
    for _ in range(10):
        a = [(i + 1) % N for i in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        # Ensure synchronization by forcing b[0]=0 or similar?
        # Actually, random binary automata are synchronizing with high probability.
        candidates.append((a, b))
        
    # Pattern 2: Known good structure?
    # Let's just rely on local search from random starts.
    
    for a, b in candidates:
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    # If we didn't find a good one, just use the first valid one.
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = compute_reset_length(best_a, best_b)
        
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Local Search (Hill Climbing with Restarts)
    # To escape local maxima, we will occasionally do a "kick" (random perturbations)
    
    no_improve_count = 0
    kick_threshold = 500
    
    while time.time() < deadline:
        # Choose a transition to mutate
        if random.random() < 0.5:
            i = random.randint(0, N-1)
            old_val = current_a[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_a[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1: # Not synchronizing
                current_a[i] = old_val
                no_improve_count += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve_count = 0
            else:
                current_a[i] = old_val
                no_improve_count += 1
        else:
            i = random.randint(0, N-1)
            old_val = current_b[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_b[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                current_b[i] = old_val
                no_improve_count += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve_count = 0
            else:
                current_b[i] = old_val
                no_improve_count += 1
                
        # Kick if stuck
        if no_improve_count > kick_threshold:
            # Randomize current state partially
            for _ in range(3):
                if random.random() < 0.5:
                    current_a[random.randint(0, N-1)] = random.randint(0, N-1)
                else:
                    current_b[random.randint(0, N-1)] = random.randint(0, N-1)
            # Check if it's still valid (synchronizing)
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                # Revert to best
                current_a = best_a[:]
                current_b = best_b[:]
                current_len = best_len
            else:
                current_len = new_len
            no_improve_count = 0
            
    return [best_a, best_b]
