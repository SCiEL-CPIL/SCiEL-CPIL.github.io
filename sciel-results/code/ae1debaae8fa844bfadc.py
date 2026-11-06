"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 50.0  # seconds, leave 10s safety margin for 60s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses a depth-based genetic/local search approach:
    1. Start with a known good network (odd-even transposition).
    2. Apply depth-aware local search to minimize comparator count.
    3. Verify using zero-one principle with bit-vector simulation.
    """
    start_time = time.time()
    
    # Baseline: odd-even transposition
    def odd_even_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with the odd-even network
    current = odd_even_network(N)
    
    # Verify it works (should always work)
    if not verify_network_fast(current):
        # This should never happen, but just in case
        return current
    
    current_count = len(current)
    best_count = current_count
    best_network = current[:]
    
    # Local search: Greedy removal followed by Randomized Hill Climbing
    # 1. Greedy Pass: Remove any redundant comparator found in order.
    def greedy_remove(net):
        improved = True
        while improved:
            improved = False
            for idx in range(len(net)):
                candidate = net[:idx] + net[idx+1:]
                if verify_network_fast(candidate):
                    net = candidate
                    improved = True
                    break
        return net

    # Initial greedy cleanup
    current = greedy_remove(current)
    current_count = len(current)
    if current_count < best_count:
        best_count = current_count
        best_network = current[:]

    # 2. Randomized Hill Climbing with Perturbations
    # We try to reduce size by:
    # a) Removing a comparator
    # b) Swapping two comparators (to enable future removals)
    # c) Moving a comparator to a different position
    
    while time.time() - start_time < DEADLINE:
        # Try a series of random moves
        move_success = False
        
        # Strategy A: Try removing a random comparator
        if len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if verify_network_fast(candidate):
                current = candidate
                current_count -= 1
                move_success = True
                if current_count < best_count:
                    best_count = current_count
                    best_network = current[:]
        
        # Strategy B: If removal failed or we just did one, try swapping two comparators
        # Swapping can change the "flow" and make a previously redundant comparator redundant.
        if not move_success and len(current) >= 2:
            i = random.randint(0, len(current) - 1)
            j = random.randint(0, len(current) - 1)
            if i != j:
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if verify_network_fast(candidate):
                    current = candidate
                    move_success = True
        
        # Strategy C: Move a comparator to a random position
        if not move_success and len(current) > 1:
            i = random.randint(0, len(current) - 1)
            j = random.randint(0, len(current) - 1)
            if i != j:
                candidate = current[:]
                comp = candidate.pop(i)
                candidate.insert(j, comp)
                if verify_network_fast(candidate):
                    current = candidate
                    move_success = True
        
        # Strategy D: Replace a comparator with a random one (mutation)
        if not move_success and len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            candidate = current[:]
            candidate[idx] = (new_i, new_j)
            if verify_network_fast(candidate):
                current = candidate
                move_success = True

        # If no move succeeded, we are stuck in a local minimum.
        # Apply a perturbation: randomly shuffle a subset of comparators or swap a few,
        # then run a short greedy removal to see if we escape.
        if not move_success:
            # Perturb: Swap 3-5 random pairs
            for _ in range(random.randint(3, 5)):
                i = random.randint(0, len(current) - 1)
                j = random.randint(0, len(current) - 1)
                if i != j:
                    current[i], current[j] = current[j], current[i]
            
            # Try to greedily remove from the perturbed state
            perturbed_current = greedy_remove(current)
            if len(perturbed_current) < current_count:
                current = perturbed_current
                current_count = len(current)
                if current_count < best_count:
                    best_count = current_count
                    best_network = current[:]
            else:
                # Revert perturbation if it didn't help immediately, 
                # or keep it if it's not worse? 
                # Usually better to revert if size didn't drop, to stay in good basin.
                # But maybe the perturbed state has potential?
                # Let's revert to avoid wandering in invalid/deep local mins.
                pass 

    # Return the best network found
    return best_network


def verify_network_fast(network):
    """
    Verify network using bit-vector simulation with Python integers.
    Each of the N wires is represented by a Python integer of N bits.
    Bit k of wire i represents the value of wire i for input k.
    """
    if not network:
        return N <= 1
    
    # Initialize wires: wire i corresponds to input bit i
    # For input x (0 to 2^N - 1), wire i has value (x >> i) & 1.
    # So wire i is an integer where bit k is 1 if input k has bit i set.
    # This is simply (1 << (1 << i)) - 1 ? No.
    # Let's construct the bit-vectors.
    # Input k is an integer. Bit i of input k is (k >> i) & 1.
    # We want a vector V_i such that bit k of V_i is (k >> i) & 1.
    # This pattern repeats every 2^(i+1) bits: 2^i zeros, then 2^i ones.
    
    wires = []
    for i in range(N):
        # Pattern: 0...0 1...1
        # Length 2^N
        # We can construct this by repeating the pattern 2^i zeros, 2^i ones.
        # A simpler way: 
        # For bit i, the value is 1 if (k & (1 << i)) != 0.
        # We can build the integer by setting bits.
        # However, constructing 17 integers of 131072 bits is fast enough.
        
        # Optimization: 
        # Wire i corresponds to bit i of the input.
        # The bit-vector for wire i has 1s at positions k where bit i of k is 1.
        # We can generate this using a loop or bitwise tricks.
        # Given N=17, 2^17 = 131072 bits.
        # Python handles big integers efficiently.
        
        v = 0
        # Iterate over the pattern period 2^(i+1)
        period = 1 << (i + 1)
        half = 1 << i
        # The pattern is 0...0 (half times) then 1...1 (half times)
        # We repeat this pattern 2^N / period times.
        # Total bits = 2^N.
        # Number of repetitions = 2^N / 2^(i+1) = 2^(N-i-1).
        
        # Construct the base pattern integer:
        # bits 0 to half-1 are 0, bits half to period-1 are 1.
        # base_pattern = ((1 << half) - 1) << half
        base_pattern = ((1 << half) - 1) << half
        
        # We need to tile this pattern.
        # Since the pattern length is period, and total length is 2^N,
        # we can use a recursive doubling or just a loop if N is small.
        # For N=17, max reps is 2^16 = 65536. A loop might be slow in Python.
        # Better approach: 
        # We can build the integer by shifting and ORing.
        # Or, simpler:
        # The bit at position k is 1 iff (k >> i) & 1 == 1.
        # Let's just build it using a list of bytes or similar? No, Python int is best.
        
        # Efficient construction:
        # v = sum( (1 << k) for k in range(2**N) if (k >> i) & 1 )
        # This is O(2^N) which is 131k operations. Doing this for 17 wires is ~2.2M ops.
        # This is done once per verification. 2.2M ops in Python is ~0.1s? Maybe too slow if we do it inside the loop.
        # We should pre-compute the initial wires!
        
        pass # Placeholder, we will move this out.
    
    # Actually, let's pre-compute the initial wire states outside the function or cache them.
    # But the function signature is fixed. We can use a global cache.
    
    if not hasattr(verify_network_fast, 'initial_wires'):
        verify_network_fast.initial_wires = []
        for i in range(N):
            half = 1 << i
            period = 1 << (i + 1)
            # Pattern: 0...0 1...1 of length period
            # We want to tile this to length 2^N.
            # Let's use a trick:
            # The bit i of input k is 1.
            # We can generate the mask by iterating k? 
            # For N=17, 131072 iterations.
            # Let's try to generate it faster.
            
            # A known trick for this specific bit-reversal/permutation pattern:
            # Actually, for wire i, the bit vector is a sequence of blocks.
            # Let's just use the loop for simplicity and speed check.
            # 17 * 131072 = 2.2 million bit sets.
            # Python bit setting is slow.
            
            # Alternative: Use bytes?
            # No, Python int is native.
            
            # Let's try to build it using string of '0' and '1'?
            # '0'*half + '1'*half repeated.
            # Then int(s, 2).
            # String construction is fast in C.
            
            block = '0' * half + '1' * half
            reps = 1 << (N - i - 1)
            s = block * reps
            v = int(s, 2)
            verify_network_fast.initial_wires.append(v)
        
    wires = verify_network_fast.initial_wires[:]
    
    for i, j in network:
        wi = wires[i]
        wj = wires[j]
        # Compare: if wi has 1 where wj has 0, swap.
        # mask = wi & ~wj
        # New wi = (wi & wj) | (~wi & wj) ? No.
        # If wi > wj (1 vs 0), then wi becomes 0, wj becomes 1.
        # So:
        # new_wi = wi & ~mask ? No.
        # If mask is set (wi=1, wj=0):
        #   new_wi should be 0.
        #   new_wj should be 1.
        # Else (wi=0, wj=1 or both 0 or both 1):
        #   new_wi = wi
        #   new_wj = wj
        
        # mask = wi & ~wj
        # new_wi = wi & ~mask  => wi & (wi | wj) ? No.
        # If mask is 1, wi is 1. We want 0.
        # new_wi = wi & ~mask
        # new_wj = wj | mask
        
        mask = wi & ~wj
        wires[i] = wi & ~mask
        wires[j] = wj | mask
    
    # Check if sorted
    # Sorted means for all k, wire[k] <= wire[k+1] bitwise?
    # No, sorted means for each input k (bit position), the sequence of values is non-decreasing.
    # i.e., for each bit position k, we check if wires[0]_k <= wires[1]_k <= ...
    # This is equivalent to checking if for any adjacent wires i, i+1, there is a bit k
    # where wires[i]_k = 1 and wires[i+1]_k = 0.
    # So we check if (wires[i] & ~wires[i+1]) is 0 for all i.
    
    for i in range(N - 1):
        if wires[i] & ~wires[i + 1]:
            return False
            
    return True


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    We only need to check all 2^N Boolean inputs.
    For N=17, that's 131072 inputs.
    Use bit-vector simulation for efficiency.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
