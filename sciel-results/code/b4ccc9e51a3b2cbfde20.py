import time
import random

N = 16

def _verify_network(comps):
    """
    Verifies if the network sorts all 2^N binary inputs.
    Uses bitmask simulation for speed.
    """
    for mask in range(1 << N):
        state = mask
        for i, j in comps:
            # Check if i has 1 and j has 0
            # i < j
            if (state >> i) & 1 and not ((state >> j) & 1):
                # Swap bits i and j
                state ^= (1 << i) | (1 << j)
        
        # Check if sorted: all 0s then 1s
        # Count 1s
        k = bin(state).count('1')
        if k > 0:
            # The lowest N-k bits must be 0
            if state & ((1 << (N - k)) - 1):
                return False
    return True

def construct():
    # Start with Odd-Even Transposition Network (120 comparators)
    # Phases 0 to 15
    # Phase p: comparators (i, i+1) for i in range(p%2, N-1, 2)
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    
    # Initial check (should be true)
    # _verify_network(comps)
    
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Random Deletion Heuristic
    # We try to remove random comparators.
    # To be more efficient, we can shuffle the list and try removing from the end?
    # Or just pick random indices.
    
    # To avoid getting stuck in a local minimum, we can restart if no progress?
    # But 120 -> 60 is a long way.
    # Let's just keep deleting as long as valid.
    
    # Optimization: 
    # If we remove a comparator, we don't need to re-verify the whole thing if we can prove it?
    # No, verification is the only way.
    
    # We can speed up verification by caching results?
    # No, state space is too big.
    
    # Let's try to delete from the end of the list first, as later comparators are often redundant
    # in simple networks? Or random?
    # Random is safer.
    
    # We will use a random shuffle approach on the indices to try.
    # Actually, just picking a random index from the current list is fine.
    
    # To improve success rate, we can try multiple restarts if we get stuck?
    # But time is limited.
    
    # Let's just iterate.
    while time.time() < deadline:
        if not comps:
            break
            
        # Pick a random comparator to remove
        idx = random.randint(0, len(comps) - 1)
        candidate = comps[idx]
        
        # Remove it
        comps.pop(idx)
        
        # Verify
        if _verify_network(comps):
            # Keep it removed
            pass
        else:
            # Restore it
            comps.insert(idx, candidate)
            
    return comps
