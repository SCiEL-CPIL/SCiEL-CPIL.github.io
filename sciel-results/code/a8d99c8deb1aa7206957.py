import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55.0  # Safety margin from 60s limit

# Pre-compute the initial bitmasks for all wires.
# Wire w has value 1 for inputs where bit w is set.
# There are 2**N inputs. We represent the state of wire w as a Python integer 
# where bit k is 1 if the k-th input has a 1 on wire w.
# Input k corresponds to the binary representation of k.
# So for wire w, the bitmask is:
# sum(1 << k for k in range(2**N) if (k >> w) & 1)
# This pattern is periodic. For w=0, it's 0b010101... 
# For w=1, it's 0b00110011...
# We can generate these efficiently.

def _generate_initial_masks():
    masks = [0] * N
    total_inputs = 1 << N
    # For each wire w, the bit is 1 for blocks of 2**w ones and 2**w zeros.
    # We can build the integer bit by bit, but 2**18 bits is 2**18 operations per wire.
    # 18 * 262144 ~ 4.7 million operations. This is fast enough at startup.
    # Alternatively, use a recursive construction or numpy.
    # Let's use a simple loop for clarity and correctness, it's only done once.
    for w in range(N):
        m = 0
        # Pattern: 2**w zeros, 2**w ones, repeating.
        # Since Python ints are arbitrary precision, we can build this.
        # But building a 2^18 bit int bit-by-bit is slow in Python.
        # Let's use numpy to create the array of bits for each wire, then pack.
        # Actually, we can just use the property:
        # mask_w = sum(1 << k for k in range(1<<N) if (k>>w)&1)
        # This is still slow if done naively.
        
        # Faster way using numpy:
        # Create array of all inputs
        # inputs = np.arange(1<<N)
        # bits = (inputs[:, None] >> np.arange(N)[None, :]) & 1
        # Then pack each row of bits into an integer? No, we need column-wise.
        # bits[:, w] gives the bit for wire w for all inputs.
        # We want an integer where bit k is bits[k, w].
        # This is equivalent to interpreting the binary string bits[:, w] as an integer.
        
        pass # We will do this in a helper function below for speed.

# Let's pre-compute the masks using a faster method.
# The mask for wire w is a number with 2**w zeros followed by 2**w ones, repeated.
# For N=18, the total length is 2**18 bits.
# We can construct the mask for wire 0: 0x5555... (alternating 0 and 1)
# Wire 1: 0x3333... (00 11)
# Wire 2: 0x0F0F... (0000 1111)
# ...
# Wire w: pattern of 2**w zeros and 2**w ones.

def _build_mask(w, n_bits):
    # n_bits is 1 << N
    # We want an integer of n_bits bits.
    # The value at position k (0-indexed from LSB) is 1 if (k >> w) & 1 == 1.
    # This is the same as the k-th bit of the mask being 1.
    # We can build this by setting ranges.
    mask = 0
    block = 1 << w
    # The pattern repeats every 2*block bits.
    # In each period, the first 'block' bits are 0, the next 'block' bits are 1.
    # So for k in range(n_bits):
    #   if (k // block) % 2 == 1: mask |= (1 << k)
    
    # We can do this more efficiently by constructing the pattern for one period and repeating it.
    # Period length P = 2 * block.
    # Pattern value: (2**block - 1) << block. This is 'block' ones followed by 'block' zeros.
    # Wait, LSB is k=0.
    # k=0..block-1: bit is 0.
    # k=block..2*block-1: bit is 1.
    # So the pattern for the first 2*block bits is: (2**block - 1) << block.
    
    pattern = (1 << block) - 1
    pattern <<= block
    # Now we need to repeat this pattern to fill n_bits.
    # Number of periods = n_bits // (2 * block)
    num_periods = n_bits // (2 * block)
    
    # Construct the full mask by ORing shifted patterns.
    # This is still O(num_periods) which is O(2**(N-w)).
    # For w=0, num_periods = 2**(N-1) = 131072.
    # 131072 OR operations is fast.
    
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, 1 << N)

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    # Copy the initial masks
    # state[w] is an integer representing the bits on wire w for all 2**N inputs
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        
        # min(a, b) for each bit position k:
        # If a_k=0, b_k=0 -> min=0
        # If a_k=0, b_k=1 -> min=0
        # If a_k=1, b_k=0 -> min=0
        # If a_k=1, b_k=1 -> min=1
        # So min is a & b.
        
        # max(a, b) for each bit position k:
        # If a_k=0, b_k=0 -> max=0
        # If a_k=0, b_k=1 -> max=1
        # If a_k=1, b_k=0 -> max=1
        # If a_k=1, b_k=1 -> max=1
        # So max is a | b.
        
        state[i] = a & b
        state[j] = a | b
        
    # Check if sorted: wire 0 <= wire 1 <= ... <= wire N-1
    # This means for all k, state[0]_k <= state[1]_k <= ...
    # Equivalent to: (state[0] & state[1]) == state[0] is NOT the condition.
    # Condition: state[i] must be a subset of state[i+1] in terms of bits set?
    # No. Sorted ascending means if wire i has a 1, wire i+1 must have a 1?
    # Let's check:
    # Input: 0, 1. Sorted: 0, 1.
    # Wire 0: 0. Wire 1: 1.
    # state[0] = 0. state[1] = 1.
    # Is state[0] & ~state[1] == 0? 
    # 0 & ~1 = 0. Yes.
    #
    # Input: 1, 0. Sorted: 0, 1.
    # Wire 0: 0. Wire 1: 1.
    # Same.
    #
    # General condition for sorted: 
    # For every input k, state[i]_k <= state[i+1]_k.
    # This is false if there is a k such that state[i]_k == 1 and state[i+1]_k == 0.
    # This corresponds to the bit k being set in state[i] AND NOT set in state[i+1].
    # So we check: (state[i] & ~state[i+1]) == 0 for all i.
    
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Start with the seed network
    current = _odd_even_transposition()
    best = current[:]
    best_len = len(best)
    
    # Initial verification of seed (should be true)
    if not _verify(best):
        # Fallback if something went wrong with seed generation
        return best
        
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = rng.choice(['remove', 'add', 'swap', 'move'])
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'add':
            # Add a random comparator
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            # Insert at a random position
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, (i, j))
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                # Move element from i to j
                comp = neighbor.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                # Insert at j+1 to shift right, or just j? 
                # Let's insert at position j (shifting existing j right)
                # Actually, standard move: pop i, insert at j.
                # If i < j, the index j shifts left by 1 after pop.
                # So we insert at j.
                # If i > j, index j is unaffected.
                neighbor.insert(j, comp)

        # Verify neighbor
        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            
            # Acceptance criterion
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(best)
        else:
            # If invalid, we generally reject, but in SA we might accept invalid states?
            # No, for sorting networks, invalid states are "infinite cost".
            # We should only move to valid states, or penalize invalid states heavily.
            # Given the strict constraint, it's better to only consider valid neighbors
            # or use a penalty. However, checking validity is the bottleneck.
            # If we reject all invalid moves, we are doing a random walk on valid states.
            # This is effectively a Metropolis algorithm on the valid subgraph.
            # This is fine.
            pass

        T *= cooling_rate
        
        # Optional: If we get stuck, try a more aggressive restart or perturbation
        # But SA with proper cooling should handle this.
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
