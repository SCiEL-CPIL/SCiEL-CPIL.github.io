"""Size-efficient sorting networks on 17 wires."""
import time
import random
import sys

N = 17
TOTAL_BITS = 1 << N  # 131072
WORD_COUNT = TOTAL_BITS // 64  # 2048 words
WORD_SIZE = 64

# EVOLVE-BLOCK-START

# Precompute masks as arrays of 64-bit words for fast verification
_INITIAL_MASKS = None

def _get_initial_masks():
    """
    Precompute initial masks as arrays of 64-bit integers.
    mask[w][k] contains the 64 bits corresponding to inputs k*64 ... k*64+63.
    Bit position p within word corresponds to input k*64+p.
    """
    global _INITIAL_MASKS
    if _INITIAL_MASKS is None:
        _INITIAL_MASKS = []
        for w in range(N):
            mask_words = [0] * WORD_COUNT
            # For wire w, the pattern is: 2^w zeros, 2^w ones, repeated.
            # We can build this efficiently.
            block = 1 << w
            # For each word, determine the pattern of bits.
            for kw in range(WORD_COUNT):
                start = kw * 64
                # For each bit position in this word (0-63)
                # The input index is start + pos
                # Wire w is 1 if (input >> w) & 1 == 1
                word = 0
                for pos in range(64):
                    inp = start + pos
                    if (inp >> w) & 1:
                        word |= (1 << pos)
                mask_words[kw] = word
            _INITIAL_MASKS.append(mask_words)
    return _INITIAL_MASKS

def _verify_network_fast(network):
    """
    Verify if the network sorts all inputs using bitwise operations on word arrays.
    Returns True if valid.
    """
    if not network:
        return N <= 1
    
    # Deep copy masks
    masks = [m[:] for m in _get_initial_masks()]
    
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        # min -> i, max -> j
        # new_mi[k] = mi[k] & mj[k]
        # new_mj[k] = mi[k] | mj[k]
        new_mi = [0] * WORD_COUNT
        new_mj = [0] * WORD_COUNT
        for k in range(WORD_COUNT):
            a = mi[k]
            b = mj[k]
            new_mi[k] = a & b
            new_mj[k] = a | b
        masks[i] = new_mi
        masks[j] = new_mj
    
    # Check sortedness: for all adjacent wires w, w+1:
    # if wire w has a 1 where wire w+1 has a 0, it's unsorted.
    # mask[w] & ~mask[w+1] must be 0 for all w
    for w in range(N - 1):
        mw = masks[w]
        mwp1 = masks[w + 1]
        for k in range(WORD_COUNT):
            if mw[k] & ~mwp1[k]:
                return False
    return True

def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    """
    Attempt to remove comparators in a deterministic greedy fashion.
    Returns the modified network.
    """
    current = list(network)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                changed = True
                break
    # Forward pass
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(current)):
            if time.time() > deadline - 2:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                changed = True
                break
    return current

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    deadline = time.time() + 80  # Safety margin
    
    # 1. Seed
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    # 2. Initial Greedy Pruning
    current_network = _greedy_remove(current_network, deadline)
    if len(current_network) < best_size:
        best_size = len(current_network)
        best_network = list(current_network)
        
    # 3. Randomized Local Search
    random.seed(42)
    current = list(current_network)
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline - 5:
        if len(current) == 0:
            break
            
        move_type = random.random()
        candidate = None
        accepted = False
        size_changed = False
        
        if move_type < 0.7 and len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                accepted = True
                size_changed = True
                
        elif move_type < 0.9 and len(current) > 1:
            i1, i2 = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            if _verify_network_fast(candidate):
                current = candidate
                accepted = True
                
        else:
            if len(current) > 1:
                i1 = random.randint(0, len(current) - 1)
                i2 = random.randint(0, len(current) - 1)
                if i1 != i2:
                    candidate = list(current)
                    item = candidate.pop(i1)
                    candidate.insert(i2, item)
                    if _verify_network_fast(candidate):
                        current = candidate
                        accepted = True

        if accepted:
            if size_changed:
                if len(current) < best_size:
                    best_size = len(current)
                    best_network = list(current)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    no_improvement_count = 0
        else:
            pass

    # 4. Final Polish
    if time.time() < deadline - 1.0:
        final_candidate = _greedy_remove(best_network, deadline)
        if len(final_candidate) < best_size:
            best_network = final_candidate
            best_size = len(final_candidate)
            
    if len(current) < len(best_network) and time.time() < deadline - 1.0:
        final_candidate = _greedy_remove(current, deadline)
        if len(final_candidate) < best_size:
            best_network = final_candidate
            best_size = len(final_candidate)

    return best_network

# EVOLVE-BLOCK-END
