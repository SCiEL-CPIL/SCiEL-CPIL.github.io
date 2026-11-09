"""
Optimized sorting network minimizer using bitmask-based vectorized verification.

Strategy:
1. Represent the state of all 2^N binary inputs for each wire as a single large integer (bitmask).
   Bit k of the mask for wire w indicates if input k has a 1 on wire w.
2. A compare-exchange (i, j) on all inputs simultaneously is:
   new_mask_i = mask_i & mask_j
   new_mask_j = mask_i | mask_j
   This is O(1) in Python (C-level big int ops) instead of O(2^N) in a Python loop.
3. Use this fast verifier to drive an aggressive greedy removal search.
"""
import time
import random

N = 17
# Safety margin: The evaluator kills at 90s. We target 80s to be safe.
_DEADLINE = 80.0

# Precompute initial masks for each wire
# For input x (0 to 2^N-1), the bit at position w is (x >> w) & 1.
# So mask_w has bit k set if (k >> w) & 1 is 1.
# This creates a repeating pattern of 0s and 1s.
_INITIAL_MASKS = []

def _precompute_initial_masks():
    """Precompute the initial bitmask for each wire."""
    global _INITIAL_MASKS
    if _INITIAL_MASKS:
        return
    for w in range(N):
        mask = 0
        # Efficient construction using pattern replication
        # Pattern for bit w: 2^w zeros, then 2^w ones, repeated.
        # Total bits: 2^N.
        # We can build this by iterating, but for N=17, a simple loop is fast enough for precomputation.
        # 17 * 131072 iterations ~ 2M ops, takes < 0.1s in Python.
        for k in range(1 << N):
            if (k >> w) & 1:
                mask |= (1 << k)
        _INITIAL_MASKS.append(mask)

def _verify_network_fast(network):
    """
    Verify if the network sorts all 2^N binary inputs using bitmask simulation.
    Returns True if valid.
    """
    if not network:
        return N <= 1
    
    # Copy initial masks
    # List of N integers, each 2^N bits wide
    masks = _INITIAL_MASKS[:]
    
    # Apply comparators
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        # min goes to i, max goes to j
        # min(1,0)=0, min(0,1)=0, min(1,1)=1, min(0,0)=0 => AND
        # max(1,0)=1, max(0,1)=1, max(1,1)=1, max(0,0)=0 => OR
        masks[i] = mi & mj
        masks[j] = mi | mj
        
    # Check if sorted:
    # For the output to be sorted, for every wire w, if there is a 1 on wire w,
    # there must be a 1 on all subsequent wires w+1...N-1.
    # This means mask[w] must be a subset of mask[w+1].
    # i.e., mask[w] & mask[w+1] == mask[w]
    # Or equivalently: mask[w] & ~mask[w+1] == 0
    for w in range(N - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove(network, idx):
    """Try removing comparator at index idx. Returns new network if valid, else None."""
    candidate = network[:idx] + network[idx+1:]
    if _verify_network_fast(candidate):
        return candidate
    return None

def construct():
    start_time = time.time()
    
    # Ensure masks are ready
    _precompute_initial_masks()
    
    # Start with seed network
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed (should be true)
    if not _verify_network_fast(network):
        return best

    # Helper to check time
    def time_left():
        return _DEADLINE - (time.time() - start_time)

    # Phase 1: Aggressive Forward Greedy Removal
    # With fast verification, we can iterate many times.
    # We will do multiple passes until no more removals are possible or time is up.
    
    # Pass 1: Forward scan
    i = 0
    while i < len(network) and time_left() > 5:
        candidate = _try_remove(network, i)
        if candidate is not None:
            network = candidate
            best = list(candidate)
            best_len = len(best)
            # Do not increment i, re-check the same index as network changed
        else:
            i += 1

    # Pass 2: Reverse scan
    if time_left() > 5:
        i = len(network) - 1
        while i >= 0 and time_left() > 5:
            candidate = _try_remove(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
                # Reset to end because indices shifted
                i = len(network) - 1
            else:
                i -= 1

    # Pass 3: Forward scan again
    if time_left() > 5:
        i = 0
        while i < len(network) and time_left() > 5:
            candidate = _try_remove(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1

    # Phase 2: Randomized Greedy Removal
    # Sometimes local minima are found in sequential scans.
    # Randomly picking an index might find a removable comparator that sequential scan missed
    # due to interaction with neighbors? (Actually, removal is independent of neighbors in terms of validity,
    # but the *order* of removal matters for finding the global minimum via greedy.
    # Randomizing the order of *checks* doesn't change the set of removable comparators,
    # but it might lead to a different sequence of removals if we were doing swaps.
    # For pure removal, the order of checking doesn't matter for *which* ones are removable,
    # but it matters for the *sequence* of states we reach.
    # However, if a comparator is removable in state S, it remains removable in state S' (subset of S)
    # only if the removal doesn't affect the others. 
    # Actually, if we remove C1, C2 might become removable. 
    # Sequential scan finds these chains. Random scan also finds them but less systematically.
    # Let's stick to sequential for reliability, but add a "swap" heuristic if time allows.
    
    # Phase 3: Swap/Move Heuristic (Structural Improvement)
    # Try swapping adjacent comparators to see if it allows further removals.
    # This is more expensive (verify twice) but can escape local minima.
    
    if time_left() > 10:
        # Try swapping adjacent pairs
        # Limit iterations to avoid timeout
        max_swap_iters = 200
        iters = 0
        random.seed(42)
        
        while time_left() > 5 and iters < max_swap_iters:
            iters += 1
            # Pick a random adjacent pair
            if len(network) < 2:
                break
            idx = random.randint(0, len(network) - 2)
            
            # Swap network[idx] and network[idx+1]
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            
            if _verify_network_fast(candidate):
                # Check if this swap allows immediate removal of one of them or others
                # Just accept the swap if it's valid, then try to remove
                network = candidate
                best = list(candidate)
                best_len = len(best)
                
                # Try to remove one of the swapped comparators
                # Try removing idx
                c1 = network[:idx] + network[idx+1:]
                if _verify_network_fast(c1):
                    network = c1
                    best = list(network)
                    best_len = len(best)
                else:
                    # Try removing idx+1
                    c2 = network[:idx+1] + network[idx+2:]
                    if _verify_network_fast(c2):
                        network = c2
                        best = list(network)
                        best_len = len(best)
                        
                # If no removal, we still have a valid network of same size.
                # We might want to revert if no improvement, but for now we keep exploring.
                # To avoid looping, we can just continue.
                
            # Also try moving a comparator (shift by 1 or 2)
            # This is complex, so we stick to swaps for now.
            
            # Occasionally do a full forward removal pass to clean up
            if iters % 10 == 0:
                i = 0
                while i < len(network) and time_left() > 5:
                    candidate = _try_remove(network, i)
                    if candidate is not None:
                        network = candidate
                        best = list(candidate)
                        best_len = len(best)
                    else:
                        i += 1

    # Final cleanup passes
    for _ in range(3):
        if time_left() < 5:
            break
        # Forward
        i = 0
        while i < len(network) and time_left() > 2:
            candidate = _try_remove(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
        # Reverse
        i = len(network) - 1
        while i >= 0 and time_left() > 2:
            candidate = _try_remove(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
                i = len(network) - 1
            else:
                i -= 1

    return best

if __name__ == "__main__":
    # Quick self-test
    _precompute_initial_masks()
    net = _seed_network()
    print(f"Seed network size: {len(net)}")
    print(f"Seed valid: {_verify_network_fast(net)}")
    
    # Run construct
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Result size: {len(result)}")
    print(f"Result valid: {_verify_network_fast(result)}")
    print(f"Time taken: {elapsed:.2f}s")
