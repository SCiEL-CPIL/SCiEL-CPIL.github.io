"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _verify_network_fast(network):
    """
    Fast verification using bitwise operations on integers representing sets of inputs.
    Each wire holds an integer mask where bit k is 1 if input k has a 1 on that wire.
    """
    if not network:
        return N <= 1

    # Initialize masks for each wire
    # Mask for wire w is (1 << w) - 1 if we consider inputs as integers 0..2^N-1
    # Actually, for input x (0 to 2^N-1), the bit at position w is (x >> w) & 1.
    # So the mask for wire w is a pattern of bits.
    # Instead of precomputing huge patterns, we can just use the property that
    # initially, wire w has 1s for all inputs where bit w is set.
    
    # We can represent the state of all 2^N inputs for a specific wire as a single integer
    # if N is small enough? No, 2^17 is 131072 bits, which is ~16KB per integer.
    # Python handles big integers well.
    
    # Let's create the initial masks.
    # For wire w, the mask M_w has bit k set if (k >> w) & 1 is 1.
    # This is a repeating pattern.
    # However, constructing these big integers for every verification call might be slow.
    # Let's precompute them once.
    
    global _initial_masks
    if not hasattr(_verify_network_fast, '_initial_masks'):
        _verify_network_fast._initial_masks = []
        for w in range(N):
            mask = 0
            # Pattern: 0...0 1...1 0...0 1...1 ...
            # The period is 2^(w+1). The top 2^w bits of the period are 1.
            # We can build this efficiently.
            # But 2^17 is small enough that even a loop over 131072 bits * 17 wires is fast.
            # 17 * 131072 ~ 2M operations. This is acceptable if done once.
            # But we want to avoid global state if possible or ensure it's safe.
            # Let's just compute it.
            for k in range(1 << N):
                if (k >> w) & 1:
                    mask |= (1 << k)
            _verify_network_fast._initial_masks.append(mask)
        _initial_masks = _verify_network_fast._initial_masks

    # Copy the masks
    masks = _initial_masks[:]
    
    # Apply comparators
    for i, j in network:
        # min goes to i, max goes to j
        # new_i = masks[i] & masks[j]
        # new_j = masks[i] | masks[j]
        # Wait, this logic is for bitwise AND/OR on the *values*.
        # If wire i has 1 and wire j has 0, we swap.
        # If wire i has 1 and wire j has 1, no swap.
        # If wire i has 0 and wire j has 1, swap.
        # If wire i has 0 and wire j has 0, no swap.
        
        # Let's trace:
        # Input states for a specific input k:
        # b_i = bit k of masks[i]
        # b_j = bit k of masks[j]
        
        # If b_i > b_j (1 > 0), swap.
        # If b_i <= b_j, no swap.
        
        # new_b_i = min(b_i, b_j)
        # new_b_j = max(b_i, b_j)
        
        # min(1,0)=0, min(0,1)=0, min(1,1)=1, min(0,0)=0 => AND
        # max(1,0)=1, max(0,1)=1, max(1,1)=1, max(0,0)=0 => OR
        
        # So:
        # new_mask_i = masks[i] & masks[j]
        # new_mask_j = masks[i] | masks[j]
        
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj

    # Check if sorted
    # For the network to be sorted, for all k, we must have masks[k] <= masks[k+1] bitwise?
    # No. For each specific input x, the output bits must be sorted.
    # That means for each wire w, if wire w has 0 and wire w+1 has 1, it's bad.
    # So for all w, we need (masks[w] & ~masks[w+1]) == 0.
    # This means masks[w] is a subset of masks[w+1] (bitwise).
    # i.e., masks[w] & masks[w+1] == masks[w].
    
    for w in range(N - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    # Set a safe deadline (50s) to ensure we return well before the 60s/90s limit
    deadline = time.time() + 50
    
    # Start with the odd-even transposition network
    # This uses 17*16/2 = 136 comparators
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    # Try to improve using greedy removal
    # Iterate multiple times if time permits, removing any removable comparator
    # The fast verifier allows many more iterations.
    
    random.seed(42)
    
    # Greedy phase 1: Reverse order removal
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = current_network[:idx] + current_network[idx+1:]
            if _verify_network_fast(candidate):
                current_network = candidate
                best_size = len(current_network)
                best_network = list(current_network)
                changed = True
                break # Restart loop as list changed

    # Greedy phase 2: Forward order removal
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(current_network)):
            if time.time() > deadline - 5:
                break
            candidate = current_network[:idx] + current_network[idx+1:]
            if _verify_network_fast(candidate):
                current_network = candidate
                best_size = len(current_network)
                best_network = list(current_network)
                changed = True
                break # Restart loop as list changed

    # Randomized search if time remains
    if time.time() < deadline - 2:
        while time.time() < deadline - 2:
            # Try a random mutation
            if random.random() < 0.7 and len(current_network) > 0:
                # Remove a random comparator
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
            else:
                # Add a random comparator (less likely to help reduce size, but explores space)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                idx = random.randint(0, len(current_network))
                candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            
            if _verify_network_fast(candidate):
                current_network = candidate
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = list(candidate)

    return best_network
# EVOLVE-BLOCK-END
