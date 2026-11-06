"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
DEADLINE = 90.0  # wall-clock seconds

# EVOLVE-BLOCK-START

def _verify(network, N=N):
    """Verify sorting network on all 2^N boolean inputs using bit-parallel simulation."""
    if not network:
        # Empty network only sorts if N <= 1
        return N <= 1
    # Pre-allocate state: 2^N x N bits packed into integers
    # Use numpy uint64 if N <= 64
    total = 1 << N
    # Represent each wire as a bit in an integer; state is array of N uint64s (or Python ints)
    # For speed, use list of Python ints (arbitrary precision)
    # state[w] is an integer where bit k is the value on wire w for input k
    state = [0] * N
    # Initialize: for each input k (0..2^N-1), wire w gets bit w of k
    for w in range(N):
        # Bit w of k for all k: this is a pattern
        # For wire w, the value is 1 for inputs where bit w is set
        # Pattern: blocks of 2^w zeros then 2^w ones, repeating
        mask = 0
        for k in range(total):
            if (k >> w) & 1:
                mask |= (1 << k)
        state[w] = mask
    
    for (i, j) in network:
        # min goes to i, max goes to j
        # min_bit = state[i] & state[j]
        # max_bit = state[i] | state[j]
        mi = state[i] & state[j]
        mj = state[i] | state[j]
        state[i] = mi
        state[j] = mj
    
    # Check sorted: for each wire w, state[w] should be 0 for first (2^N - 2^w) inputs and 1 for last 2^w inputs
    # Actually, after sorting, wire w (0-indexed) should hold the w-th smallest value.
    # For boolean inputs, sorted means: wire 0 has 0s, wire N-1 has 1s.
    # More precisely: wire w should be 0 for inputs with fewer than (N-w) ones, and 1 otherwise.
    # Simpler check: state[0] should be 0 (all inputs have min=0 on wire 0? No.)
    # Let's think: after sorting, wire 0 holds the global minimum. For boolean, min is 0 unless all inputs are 1.
    # Actually, the standard check: for each input k, the sorted output should have 0s followed by 1s.
    # Wire w should be 1 iff the number of 1s in input k is > (N-1-w).
    # Equivalent: state[w] should equal the mask for "bit (N-1-w) of the popcount of k"... no.
    
    # Better check: for each wire w, state[w] should be:
    # 1 for inputs k where popcount(k) >= N - w
    # 0 otherwise
    # Let's verify this way.
    
    for w in range(N):
        expected = 0
        for k in range(total):
            if k.bit_count() >= N - w:
                expected |= (1 << k)
        if state[w] != expected:
            return False
    return True


def _depth(network, N=N):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    # depth[i] = earliest time wire i is free
    # Use longest path in DAG
    # depth of comparator (i,j) at step t is max(depth_i, depth_j) + 1
    # We want the max over all comparators
    # Simple simulation:
    wire_depth = [0] * N
    max_depth = 0
    for (i, j) in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        if d > max_depth:
            max_depth = d
    return max_depth


def construct():
    """Construct a sorting network for N=18 wires."""
    start_time = time.time()
    random.seed(42)
    
    # Start from odd-even transposition network (153 comparators)
    # Try to improve via local search
    
    # Initial network: odd-even transposition
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial
    if not _verify(network):
        # Should always be valid
        return network
    
    best_network = list(network)
    best_count = len(best_network)
    
    # Local search: try to remove or replace comparators
    # Strategy: 
    # 1. Try removing each comparator; if still valid, keep removal
    # 2. Try replacing each comparator with a different one
    # 3. Try swapping adjacent comparators
    
    # First pass: greedy removal
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 5:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if _verify(candidate):
                network = candidate
                changed = True
                if len(network) < best_count:
                    best_count = len(network)
                    best_network = list(network)
    
    # Second pass: try replacing comparators with others
    for idx in range(len(network)):
        if (time.time() - start_time) > DEADLINE - 10:
            break
        i, j = network[idx]
        # Try different pairs
        for new_i in range(N):
            if new_i >= j:
                continue
            for new_j in range(new_i + 1, N):
                if (new_i, new_j) == (i, j):
                    continue
                candidate = list(network)
                candidate[idx] = (new_i, new_j)
                if _verify(candidate):
                    network = candidate
                    if len(network) < best_count:
                        best_count = len(network)
                        best_network = list(network)
                    break  # Found a valid replacement, stop
            if (time.time() - start_time) > DEADLINE - 10:
                break
    
    # Try to add comparators that allow further removals? (More complex, skip for now)
    
    # Return best found
    return best_network

# EVOLVE-BLOCK-END
