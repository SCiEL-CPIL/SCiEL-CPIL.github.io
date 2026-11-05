import numpy as np
import time
from itertools import combinations

N = 18
# 18 wires -> 18 uint64 masks. Each mask holds the bits for all 2^18 inputs.
# Wire i holds a mask M_i where bit k is 1 if input k's bit at position i is 1?
# No, standard bit-parallel sorting network representation:
# We have 2^N inputs. For input k (0 <= k < 2^N), the value is the bits of k.
# Wait, the task says "sorts all 18-input sequences".
# The inputs are 18-bit sequences? Or 18 wires carrying bits?
# "compare-exchange network that sorts all 18-input sequences"
# "correctness is certified over all 262,144 Boolean inputs"
# This implies the inputs are 18-bit vectors? No, 2^18 = 262,144.
# If the inputs are 18-bit vectors, there are 2^18 possible input vectors.
# A sorting network on 18 wires sorts 18 bits?
# If we sort 18 bits, the output must be sorted bits (0s then 1s).
# The number of distinct input patterns is 2^18.
# So we are sorting 18 bits.
# State representation:
# We can represent the state of the 18 wires for all 2^18 inputs simultaneously.
# Let `states` be an array of shape (18, 2^18) of uint8? That's 18 * 262144 = 4.7 MB.
# Or, we can use bit-packing.
# Let's use an array of 18 uint64s? No, 2^18 bits = 32768 bytes = 4096 uint64s.
# So we need 18 * 4096 uint64s = 18 * 512 KB = 9 MB.
# This fits in L2/L3 cache easily.

# Let's refine the bit-parallel representation.
# Input k (0 <= k < 2^18).
# The value on wire i for input k is bit i of k?
# If we are sorting the bits of the input integer k, then yes.
# Initial state:
# states[i] is a bitset of length 2^18.
# states[i][k] = (k >> i) & 1.
# We can store this as a list of 18 numpy arrays of uint64, or one big array.
# Actually, numpy handles this well with uint8 or uint64.
# Let's use uint64 arrays for speed.
# Shape: (18, 4096) of uint64.
# 4096 * 8 bytes = 32 KB per wire. 18 wires = 576 KB. Very small.

SIZE = 1 << N
BLOCKS = SIZE // 64  # 4096

def _init_states():
    """
    Initialize the bit-parallel states.
    states[i, b] is a uint64 where bit j is 1 if input (b*64 + j) has bit i set.
    """
    states = np.zeros((N, BLOCKS), dtype=np.uint64)
    for i in range(N):
        # For each block b, we need to set bits j where (b*64 + j) has bit i set.
        # This is a regular pattern.
        # Bit i is 1 for j in [2^i, 2*2^i - 1] within a cycle of 2^(i+1).
        # Since we are dealing with 64-bit chunks, and i < 6, the pattern repeats every 2^(i+1) bits.
        # For i >= 6, the pattern spans multiple 64-bit blocks?
        # N=18. i goes up to 17.
        # If i < 6, the pattern fits within 64 bits.
        # If i >= 6, bit i depends on the block index b as well.
        # Let's just construct it simply using numpy where possible, or a loop.
        # A loop over 4096 blocks * 18 wires is 73k iterations. Fast enough.
        for b in range(BLOCKS):
            base = b * 64
            # We need bits j (0..63) such that (base + j) & (1 << i) != 0
            # This is equivalent to:
            # Let mask = 1 << i
            # We want j such that (base + j) % (2*mask) >= mask
            
            if i < 6:
                # Pattern repeats within the 64-bit word
                # Create a 64-bit mask for the pattern
                m = 0
                for j in range(64):
                    if (j & (1 << i)) != 0:
                        m |= (1 << j)
                states[i, b] = m
            else:
                # Pattern spans blocks
                # (base + j) has bit i set if (base + j) // (1 << i) is odd
                # Let period = 1 << (i+1)
                # (base + j) // period is odd
                # base // period + j // period ...
                # Simpler: check each j? 64 checks * 4096 blocks * 12 wires (i>=6) = 3M ops.
                # Or use vectorization.
                # Let's use a vectorized approach for i >= 6.
                # Actually, let's just use a precomputed table or simple logic.
                # For i >= 6, 1 << i is a multiple of 64.
                # So bit i of (base + j) is the same for all j in the block?
                # No. (base + j) ranges from b*64 to b*64+63.
                # If i >= 6, 2^i is a multiple of 64.
                # So (b*64 + j) // 2^i = b * 64 // 2^i + j // 2^i
                # Since j < 64 <= 2^i, j // 2^i is 0.
                # So the value is determined by b.
                # Bit i of (b*64) is (b >> (i-6)) & 1.
                # If this bit is 1, then all 64 bits in the block are 1?
                # Let's check.
                # i=6. 2^6=64.
                # (b*64 + j) & 64.
                # If b is even, b*64 is multiple of 128. (b*64 + j) < b*64+64.
                # So bits 6..0 are j. Bit 6 is 0 for all j.
                # If b is odd, b*64 = (2k+1)*64 = 128k + 64.
                # (128k + 64 + j) & 64 = 64. Bit 6 is 1 for all j.
                # Yes! For i >= 6, the bit is constant across the 64-bit block.
                bit_val = (b >> (i - 6)) & 1
                if bit_val:
                    states[i, b] = 0xFFFFFFFFFFFFFFFF
                else:
                    states[i, b] = 0
    return states

def _apply_comparator(states, i, j):
    """
    Apply comparator (i, j) to the bit-parallel states in-place.
    min -> i, max -> j.
    For bits:
    min_bit = a & b
    max_bit = a | b
    """
    ai = states[i].copy()
    aj = states[j].copy()
    states[i] = ai & aj
    states[j] = ai | aj

def _is_sorted(states):
    """
    Check if the network sorts all inputs.
    For each input k, the bits on wires 0..17 must be sorted.
    Sorted means: 0s then 1s.
    So wire 0 must be 0 if there are any 0s? No.
    Sorted sequence of bits: 0...0 1...1.
    So wire i must be >= wire i+1?
    Wait. If sorted ascending: 0,0,1,1.
    Wire 0 (LSB?) or Wire 0 (first wire)?
    Standard convention: Wire 0 is the first wire.
    If the output is sorted, then for any input, the number of 1s is fixed.
    The 1s must be on the highest indices? Or lowest?
    "compare-exchange writes min to wire i and max to j".
    If i < j, min goes to i, max to j.
    So smaller values go to lower indices.
    Sorted means wire 0 <= wire 1 <= ... <= wire 17.
    So states[i] <= states[i+1] for all i?
    In terms of bitsets:
    For every input k, bit i <= bit i+1.
    This means: states[i] & ~states[i+1] must be 0.
    i.e., there is no input where wire i is 1 and wire i+1 is 0.
    """
    for i in range(N - 1):
        # If there is any 1 in states[i] that is not in states[i+1], it's not sorted.
        # states[i] & ~states[i+1]
        # If this is non-zero, return False.
        if np.any(states[i] & ~states[i+1]):
            return False
    return True

def _verify_network(network, init_states):
    """
    Verify if a network sorts all inputs.
    """
    # Copy init states
    # states is (18, 4096) uint64. Copy is fast.
    states = init_states.copy()
    for i, j in network:
        _apply_comparator(states, i, j)
    return _is_sorted(states)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    init_states = _init_states()
    
    # BFS over network depth
    # Level 0: Empty network
    # Level d: Networks of depth d
    
    # We store valid networks at each depth.
    # To save memory and time, we only keep the networks at the current depth 
    # and generate the next.
    # However, the number of networks grows exponentially.
    # Depth 1: C(18,2) = 153 networks.
    # Depth 2: 153 * 153 = 23409 networks.
    # Depth 3: 23409 * 153 = 3.5M networks.
    # Depth 4: 3.5M * 153 = 540M networks. Too many.
    
    # We need to prune.
    # Pruning Strategy:
    # 1. Symmetry: We can fix the first comparator to be (0, 1) by symmetry?
    #    Not exactly, because the network must sort *all* inputs.
    #    But we can use canonical forms.
    # 2. Dominance: If network A is a prefix of network B, and A is valid, 
    #    then B is valid? No, adding comparators can break sorting?
    #    No, if A sorts all inputs, then A is a valid sorting network.
    #    Adding more comparators to a sorting network keeps it a sorting network?
    #    Yes! If a network sorts all inputs, applying more comparators 
    #    (which are monotonic and preserve order) will keep the output sorted.
    #    Wait. If the output is sorted, and I apply a comparator (i, j),
    #    does it remain sorted?
    #    Yes. If x_i <= x_j, min(x_i, x_j)=x_i, max=x_j. No change.
    #    If x_i > x_j, but the sequence is sorted, this cannot happen for i<j.
    #    So if the network is sorted, any suffix is redundant?
    #    No, the comparator might be between non-adjacent wires?
    #    If the sequence is sorted, x_i <= x_j for all i<j.
    #    So any comparator (i, j) with i<j will do nothing.
    #    Therefore, if a network is valid, any extension of it is valid.
    #    
    #    This implies that if we find a valid network of depth d, 
    #    we don't need to explore its extensions to find *minimal* depth?
    #    We are looking for the *smallest* number of comparators.
    #    So we want the first depth d where *any* network is valid.
    #    
    #    BFS finds the shortest path (in terms of number of comparators) 
    #    to a valid state.
    #    
    #    The number of states at depth d is C(18,2)^d.
    #    d=1: 153
    #    d=2: 23k
    #    d=3: 3.5M
    #    d=4: 540M (Too slow to verify all)
    #    
    #    We need to prune the search space.
    #    Pruning Rule:
    #    Two networks A and B of the same depth are equivalent if they produce 
    #    the same output for all inputs?
    #    We can hash the output states?
    #    The output state is 18 * 4096 uint64s = 576 KB.
    #    Hashing 576 KB is fast.
    #    We can keep a set of seen output hashes at each depth.
    #    If we reach the same output state via different networks, 
    #    we only need to keep one (the one with fewer comparators, or just one).
    #    
    #    This is a Deduplication Strategy.
    
    # Let's implement BFS with Deduplication.
    
    # State: (network_tuple, output_hash)
    # We don't need to store the network if we only care about existence?
    # We need to return the network.
    # So we store (network, hash).
    
    # To save memory, we can store just the network list.
    
    current_level = [()]  # Empty network
    seen_hashes = set()
    
    # Pre-compute hash of empty network output?
    # The output of empty network is the input itself.
    # We should mark this as seen?
    # Actually, we want to find the *first* valid network.
    
    # Let's estimate the number of unique states.
    # The number of distinct sorting behaviors is much smaller than 153^d.
    
    best_network = None
    
    # Depth 0
    if _verify_network((), init_states):
        return []
        
    for depth in range(1, 10):  # Max depth 9?
        if time.time() > deadline - 5:
            break
            
        next_level = []
        next_seen = set()
        
        # Generate all extensions of current_level
        for net in current_level:
            # Try adding all possible comparators
            for i in range(N):
                for j in range(i + 1, N):
                    new_net = net + ((i, j),)
                    
                    # Optional Pruning: 
                    # If the last comparator is redundant in the sense that it doesn't change the state?
                    # We can check if applying (i,j) to the state *before* the last step 
                    # changes the state. If not, it's redundant.
                    # But we don't have the state before the last step easily without re-simulating.
                    # However, we can just verify the new network.
                    
                    # Verification
                    if _verify_network(new_net, init_states):
                        # Found a valid network!
                        # Since BFS explores by depth, this is the minimum depth.
                        # But is it the minimum number of comparators?
                        # Yes, depth = number of comparators.
                        return list(new_net)
                    
                    # If not valid, add to next level if we haven't seen this state before.
                    # We need to hash the output state of new_net.
                    # To avoid re-simulating for hash, we can store the state in the level?
                    # Memory: 3.5M states * 576 KB = 2 TB. Too much.
                    
                    # We must re-simulate to get the hash?
                    # _verify_network already simulates.
                    # Let's modify _verify_network to return the final state if needed, 
                    # or just accept the cost of double simulation (one for check, one for hash).
                    # Actually, _verify_network simulates and checks.
                    # We can extract the state from it.
                    
                    # Let's create a helper that returns (is_valid, final_states)
                    def verify_and_get_state(network, init_states):
                        states = init_states.copy()
                        for i, j in network:
                            _apply_comparator(states, i, j)
                        # Check sorted
                        for i in range(N - 1):
                            if np.any(states[i] & ~states[i+1]):
                                return False, states
                        return True, states

                    is_valid, final_states = verify_and_get_state(new_net, init_states)
                    
                    if is_valid:
                        return list(new_net)
                        
                    # Hash the final states
                    # Use a simple hash of the bytes
                    h = hash(final_states.tobytes())
                    
                    if h not in seen_hashes and h not in next_seen:
                        next_seen.add(h)
                        next_level.append(new_net)
                        
        # Move to next level
        current_level = next_level
        seen_hashes.update(next_seen)
        
        if not current_level:
            break
            
    # If we didn't find one (should not happen for N=18, depth < 10), 
    # return a known good network?
    # The seed network (153 comps) is a fallback.
    return _seed_network()

def _seed_network():
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

# Helper functions defined globally as in the seed program structure
def _is_sorted(states):
    return not np.any(states[:-1] < states[1:])

def _apply_comparator(states, i, j):
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _restore_comparator(states, i, j, si, sj):
    states[i] = si
    states[j] = sj

def _verify_network(network, states):
    for i, j in network:
        _apply_comparator(states, i, j)
    return _is_sorted(states)
