import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N  # 131072
DEADLINE_FRACTION = 0.92

# Pre-compute initial and expected wires for verification
# We use Python big integers where bit k represents the state of input mask k.
# wire[i] has bit k set if input mask k has bit i set.
# Expected wire[i] has bit k set if popcount(k) >= N - i.

_INITIAL_WIRES = [0] * N
_EXPECTED_WIRES = [0] * N

def _init_caches():
    """Compute the constant bitmasks for verification."""
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES[0] != 0 or _EXPECTED_WIRES[0] != 0:
        return
        
    # For bit i, the pattern is 0 for 2^i, 1 for 2^i, repeat.
    # We can build these integers efficiently by noting the periodicity.
    # However, a simple loop is fast enough for N=17 (2.2M ops total).
    # Let's use a slightly optimized approach:
    # For bit i, the value is sum(1 << k for k in range(TOTAL_MASKS) if (k >> i) & 1)
    
    for i in range(N):
        w = 0
        # Optimization: The pattern repeats every 2^(i+1).
        # In each period, the lower 2^i bits are 0, upper 2^i are 1.
        # We can construct the integer for one period and then replicate it?
        # Python ints don't have a direct "repeat bits" op, but we can use shifts and ORs.
        # Actually, the naive loop is ~2.2M iterations for all 17 wires.
        # In Python, 2.2M simple ops is ~0.1s. This is acceptable for startup.
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
        # Expected wires
        # Bit k is 1 if popcount(k) >= N - i
        w_exp = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w_exp |= (1 << k)
        _EXPECTED_WIRES[i] = w_exp

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_network(net):
    """
    Returns True if the network sorts all 131,072 Boolean inputs.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
            
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
            
    return True

def _verify_batch(net, indices_to_remove):
    """
    Efficiently checks if removing any single comparator from 'indices_to_remove'
    results in a valid network.
    Returns a list of indices that are removable.
    
    Strategy:
    1. Compute the state of the network *after* each comparator in a "checkpoint" manner?
       No, that's complex.
       
    Alternative Strategy for Steepest Descent:
    Since N is small (17) and max comparators is ~136, we can just loop.
    But we can speed it up by checking candidates in parallel? No, Python is sequential.
    
    However, we can optimize the verification itself.
    The current verification is ~0.1ms.
    136 candidates * 0.1ms = 13.6ms per pass.
    We can do ~700 passes in 10s.
    Each successful removal reduces size by 1.
    If we remove ~50 comparators, we do 50 passes.
    Then we are stuck?
    
    To improve, we can try to remove *multiple* comparators if they are independent?
    Or just stick to single removal but do it very fast.
    
    Let's refine the verification to be as fast as possible.
    The bottleneck is the big integer operations.
    
    Let's try a different approach for the search:
    Instead of checking every index one by one, we can group them?
    No, let's just make the loop tight.
    """
    valid_removals = []
    current_len = len(net)
    
    # We can pre-allocate wires? No, they change.
    
    for idx in indices_to_remove:
        # Create a view of the network without idx
        # Instead of slicing, we can just skip idx in the loop
        wires = _INITIAL_WIRES[:]
        
        for k, (i, j) in enumerate(net):
            if k == idx:
                continue
            wi = wires[i]
            wj = wires[j]
            S = wi & ~wj
            wires[i] = wi ^ S
            wires[j] = wj ^ S
            
        # Check validity
        is_valid = True
        for i in range(N):
            if wires[i] != _EXPECTED_WIRES[i]:
                is_valid = False
                break
                
        if is_valid:
            valid_removals.append(idx)
            
    return valid_removals

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Initialize caches
    _init_caches()
    
    # Start with seed
    current_net = _build_seed()
    
    # Verify seed
    if not _verify_network(current_net):
        return current_net
        
    best_net = current_net[:]
    
    # Steepest Descent Local Search
    # At each step, find the best comparator to remove.
    # "Best" means: any valid removal. 
    # If multiple are valid, does it matter which one we remove?
    # Yes, removing one might block removing others later.
    # However, for sorting networks, the "greedy" removal order often works well.
    # To be more robust, we could try to remove the one that leaves the network "most flexible"?
    # That's hard to measure.
    
    # Let's just pick the first valid removal found in a randomized order?
    # Or the first in the list?
    # Randomization helps escape local minima where a specific order of removal fails.
    # But since we check ALL removals in a pass, we can pick the one that
    # maybe is "central" or just random among valid ones.
    
    # Actually, a known strategy is to try to remove comparators from the "outside in" 
    # or "inside out". But random is safer.
    
    # Optimization:
    # If we find one valid removal, we remove it and restart the scan.
    # This is O(L^2 * V) where V is verification cost.
    # L=136. 136^2 * 0.1ms = 1856 ms. Very fast.
    
    # Let's implement:
    # 1. Loop until no valid removal found or time runs out.
    # 2. In each iteration, shuffle indices? No, that changes the "steepest" nature.
    #    Steepest descent usually implies finding the *best* move.
    #    Here, all moves have the same "cost" (size -1).
    #    So any valid move is equally good in terms of size.
    #    The difference is in the *future* potential.
    #    
    #    To maximize future potential, we might want to remove comparators that are
    #    "redundant" in a structural sense.
    #    
    #    Let's try a heuristic: 
    #    1. Find all valid removals.
    #    2. If none, stop (or try a swap/reorder move? Reordering doesn't change size).
    #    3. If some, pick one.
    #       Heuristic: Pick the one that, if removed, results in a network that is
    #       "closest" to being minimal? Hard to define.
    #       Let's just pick the first one found in a random order to ensure we explore
    #       different paths if we get stuck? 
    #       But we only keep the best (smallest) size.
    #       Since we always remove, size decreases monotonically.
    #       The risk is getting stuck at a local minimum where no single removal works.
    #       
    #       To escape local minima, we can occasionally accept a "swap" of two comparators
    #       that doesn't change size, hoping to enable a removal later.
    #       But swapping is O(1) to apply, but requires verification?
    #       No, swapping preserves validity. So we can swap freely!
    #       Wait, does swapping preserve validity?
    #       Yes, a sorting network is defined by the *set* of comparators and their *order*.
    #       Changing the order of independent comparators (those on disjoint wires) is safe.
    #       Changing the order of dependent comparators might break it?
    #       No, the network is a sequence. If we swap two comparators (i,j) and (k,l),
    #       the resulting sequence is different. It might not sort!
    #       So swapping is NOT free. It requires verification.
    #       
    #       However, if we just *reorder* the list of comparators, we are changing the network.
    #       So we must verify.
    #       
    #       Let's stick to removals first. If stuck, we can try random reordering + removal.
    
    stuck_passes = 0
    max_stuck_passes = 5 # If we do 5 full passes with no removal, try reordering
    
    while time.time() < deadline:
        # Try to find a valid removal
        # To make it "steepest", we could evaluate all and pick one.
        # But since all have same size delta, we just pick one valid.
        # Randomizing the order of checks might help find "different" valid removals
        # if the set of valid removals changes? No, the set is fixed for the current net.
        
        # Let's shuffle the indices to try different removals first?
        # Actually, if we remove A, then B might become removable.
        # If we remove B, A might become removable.
        # The path matters.
        
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            # Check if removing idx is valid
            # We can do this efficiently by simulating the network without idx.
            
            # Optimization: 
            # We can pre-compute the state *before* idx and *after* idx?
            # No, the state depends on all previous comparators.
            
            # Let's just run the verification for this specific removal.
            # It's ~0.1ms.
            
            wires = _INITIAL_WIRES[:]
            
            # Apply all comparators except idx
            for k, (i, j) in enumerate(current_net):
                if k == idx:
                    continue
                wi = wires[i]
                wj = wires[j]
                S = wi & ~wj
                wires[i] = wi ^ S
                wires[j] = wj ^ S
                
            # Check
            valid = True
            for i in range(N):
                if wires[i] != _EXPECTED_WIRES[i]:
                    valid = False
                    break
            
            if valid:
                # Remove it
                current_net.pop(idx)
                best_net = current_net[:]
                removed_any = True
                break # Restart the pass with the new, shorter network
        
        if not removed_any:
            stuck_passes += 1
            
            # If stuck, try a random reordering to escape local minimum?
            # Reordering requires verification.
            # If we reorder, we might find a removal.
            
            if stuck_passes < max_stuck_passes and len(current_net) > 1:
                # Try a random swap of two comparators
                i1 = random.randint(0, len(current_net) - 1)
                i2 = random.randint(0, len(current_net) - 1)
                if i1 != i2:
                    current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
                    # Verify the swapped network
                    if _verify_network(current_net):
                        # Valid, keep the swap. It might enable removals.
                        # We don't update best_net because size is same.
                        # Reset stuck_passes? No, we still haven't reduced size.
                        # But we changed the structure.
                        pass
                    else:
                        # Swap broke it, revert
                        current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
            else:
                # If stuck for too long, we might be at a local minimum.
                # We can try to add a comparator? No, we want to minimize.
                # We can try a "2-removal" move? 
                # i.e., remove two comparators at once.
                # This is O(L^2) checks. 136^2 = 18k checks. 18k * 0.1ms = 1.8s.
                # Feasible if we are stuck.
                
                if stuck_passes == max_stuck_passes:
                    # Try pairwise removals
                    n = len(current_net)
                    found_pair = False
                    # To speed up, we can just try a few random pairs
                    for _ in range(100):
                        i1 = random.randint(0, n - 1)
                        i2 = random.randint(0, n - 1)
                        if i1 == i2:
                            continue
                        
                        # Simulate removal of i1 and i2
                        wires = _INITIAL_WIRES[:]
                        for k, (i, j) in enumerate(current_net):
                            if k == i1 or k == i2:
                                continue
                            wi = wires[i]
                            wj = wires[j]
                            S = wi & ~wj
                            wires[i] = wi ^ S
                            wires[j] = wj ^ S
                            
                        valid = True
                        for i in range(N):
                            if wires[i] != _EXPECTED_WIRES[i]:
                                valid = False
                                break
                                
                        if valid:
                            # Remove both
                            # Ensure i1 < i2 for popping
                            if i1 > i2:
                                i1, i2 = i2, i1
                            current_net.pop(i2)
                            current_net.pop(i1)
                            best_net = current_net[:]
                            found_pair = True
                            break
                            
                    if found_pair:
                        stuck_passes = 0 # Reset because we made progress
                    else:
                        # Still stuck
                        pass
                        
                if stuck_passes > max_stuck_passes + 5:
                    # Give up on further search, return best
                    break

    return best_net

if __name__ == "__main__":
    _init_caches()
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
