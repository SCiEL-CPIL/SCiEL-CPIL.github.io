"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def _make_seed():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _verify(network):
    """Verify network sorts all 2^N Boolean inputs using zero-one principle.
    Returns True if valid, False otherwise."""
    # Build all 2^N bit vectors as a (NUM_INPUTS, N) uint8 array
    # Each row is a distinct Boolean input
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Expand to N bits: bit k of index goes to wire k
    # wires[k] = (indices >> k) & 1
    bits = [(indices >> k) & 1 for k in range(N)]
    # Stack into (NUM_INPUTS, N) array
    state = np.stack(bits, axis=1).astype(np.uint8)  # shape (NUM_INPUTS, N)

    # Apply each comparator
    for (i, j) in network:
        si = state[:, i]
        sj = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(si, sj)
        state[:, j] = np.maximum(si, sj)

    # After sorting, each row should be sorted: non-decreasing left to right
    # i.e., state[:, k] <= state[:, k+1] for all k
    # Equivalent: for each row, the number of 1s should be in the rightmost positions
    # Check: for each row, state should equal its sorted version
    # A row is sorted iff for all k < l: state[k] <= state[l]
    # Efficient: check that no 1 appears before a 0 in each row
    # i.e., cumulative sum of (state==0) from left should be 0 at the position of first 1... 
    # Simpler: check state[:, k] <= state[:, k+1] for all k
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k + 1]):
            return False
    return True


def _depth(network):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    # Track the latest stage each wire is involved in
    # A comparator (i,j) at position p can be placed in a stage after both wires i and j are free
    # We use a greedy layering: assign each comparator to the earliest possible stage
    # where both its wires are free.
    # Simpler: use the standard definition - two comparators can be parallel if they don't share wires.
    # Compute depth via greedy coloring of comparators (interval graph coloring).
    # Each comparator occupies wires i and j. Two comparators conflict if they share a wire.
    # Depth = chromatic number of this conflict graph (greedy gives upper bound).
    # For a quick estimate, use the maximum number of comparators on any wire + 1... no.
    # Actually, the standard depth is the minimum number of stages.
    # Let's compute it properly: process comparators in order, assign each to the first stage
    # where neither of its wires was used in that stage.
    wire_last_stage = [-1] * N
    max_stage = 0
    for (i, j) in network:
        # Find the earliest stage > max(wire_last_stage[i], wire_last_stage[j])
        s = max(wire_last_stage[i], wire_last_stage[j]) + 1
        wire_last_stage[i] = s
        wire_last_stage[j] = s
        if s > max_stage:
            max_stage = s
    return max_stage


def construct():
    """Search for a small valid sorting network on N=17 wires using simulated annealing."""
    start_time = time.time()
    
    # Initialize with seed network
    best_net = _make_seed()
    best_len = len(best_net)
    
    # Verify seed is valid (should be)
    if not _verify(best_net):
        # Fallback: return seed anyway
        return best_net
    
    # Simulated Annealing
    current_net = list(best_net)
    current_len = len(current_net)
    
    # Temperature schedule
    T_start = 1.0
    T_min = 0.001
    # We'll do a fixed number of iterations, but check time each iteration
    # Each iteration: try a move, verify, accept/reject
    
    # Move types:
    # 1. Remove a random comparator (if len > lower bound, say 50)
    # 2. Add a random comparator (i, j) at a random position
    # 3. Swap two comparators
    # 4. Move a comparator to a different position
    
    random.seed(42)
    
    # Precompute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Track best
    iteration = 0
    while time.time() - start_time < DEADLINE:
        # Choose a move
        move_type = random.random()
        new_net = list(current_net)
        
        if move_type < 0.3 and len(new_net) > 60:
            # Remove a random comparator
            idx = random.randint(0, len(new_net) - 1)
            new_net.pop(idx)
        elif move_type < 0.6:
            # Add a random comparator at a random position
            pair = random.choice(all_pairs)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, pair)
        elif move_type < 0.8:
            # Swap two comparators
            if len(new_net) >= 2:
                i, j = random.sample(range(len(new_net)), 2)
                new_net[i], new_net[j] = new_net[j], new_net[i]
        else:
            # Move a comparator to a different position
            if len(new_net) >= 2:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    comp = new_net.pop(i)
                    new_net.insert(j, comp)
        
        new_len = len(new_net)
        
        # Quick check: if new_len >= current_len and we removed nothing useful, skip
        # But we need to verify correctness
        # Only verify if it could be better or equal in length
        if new_len <= best_len:
            # Verify
            if _verify(new_net):
                # Accept and update best
                current_net = new_net
                current_len = new_len
                if new_len < best_len:
                    best_net = list(new_net)
                    best_len = new_len
                    # Reset temperature boost on improvement
            else:
                # Invalid, reject
                pass
        else:
            # New network is longer; only accept with small probability (annealing)
            # Actually, for minimization, we should rarely accept longer networks
            # unless we're in a deep local minimum. Let's accept with prob exp(-delta/T)
            # where delta = new_len - current_len > 0
            # But this is expensive to verify. Skip for now to save time.
            pass
        
        iteration += 1
        # Check deadline more frequently
        if iteration % 10 == 0 and time.time() - start_time > DEADLINE - 5:
            break
    
    # Final verification of best
    if not _verify(best_net):
        # Should not happen, but safety
        return _make_seed()
    
    return best_net
# EVOLVE-BLOCK-END
