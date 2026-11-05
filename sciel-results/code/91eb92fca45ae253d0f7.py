"""
Strategy: Bit-Parallel Simulated Annealing for Sorting Networks.

Unlike the baseline which starts with a fixed 120-comparator network and only removes
comparators (greedy descent), this strategy uses a probabilistic search (Simulated
Annealing) that allows temporary increases in network size to escape local minima.

Key Innovations:
1. Bit-Parallel Verification: Uses numpy uint64 arrays to verify all 65,536 binary inputs
   simultaneously. A single comparator operation is a bitwise AND/OR on two 1024-element
   arrays, taking ~microseconds. This allows thousands of search steps per second.
2. Move Set:
   - Remove: Try to delete a random comparator.
   - Insert: Try to add a random comparator (allows escaping local minima where removal
     is stuck but a larger network can be rearranged into a smaller one).
   - Double-Remove: Try to delete two comparators at once. This is crucial for jumping
     over "plateaus" where no single removal is valid, but a pair is.
3. Annealing Schedule: Starts with a high "temperature" to explore broadly (accepting
   worse solutions) and cools down to focus on exploitation (only accepting improvements
   or equivalent solutions).

This approach is materially different from pure greedy removal because it can navigate
the search space through non-improving moves, which is necessary to find the global
optimum (60 comparators) from a random or suboptimal starting point.
"""

import numpy as np
import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
CHUNKS = NUM_INPUTS // 64  # 1024 chunks of 64 inputs each

# Pre-compute the initial states for all 65536 binary inputs.
# State shape: (N, CHUNKS) where each element is uint64.
# Bit 'k' in chunk 'c' corresponds to input vector (c * 64 + k).
# We need to set bit 'w' of input 'v' to 1 if the w-th bit of v is 1.

def _init_states():
    states = np.zeros((N, CHUNKS), dtype=np.uint64)
    # Optimization: Compute masks for each wire and chunk
    for w in range(N):
        for c in range(CHUNKS):
            mask = 0
            # Unroll inner loop for speed? No, 64 iterations is fine.
            # v = c * 64 + k
            # (v >> w) & 1 == 1
            # If w < 6: depends on k and c (if w+6 > 6? No, w is small)
            # Let's just use a fast loop.
            base = c * 64
            for k in range(64):
                if (base + k) & (1 << w):
                    mask |= (1 << k)
            states[w, c] = mask
    return states

_INIT_STATES = None

def _get_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_states()
    return _INIT_STATES

def _verify_network_bit_parallel(comparators):
    """
    Verifies if the network sorts all 65536 binary inputs.
    Returns True if sorted, False otherwise.
    """
    states = _get_states()
    # Copy initial states
    curr = states.copy()
    
    for i, j in comparators:
        # min(curr[i], curr[j]) = curr[i] & curr[j]
        # max(curr[i], curr[j]) = curr[i] | curr[j]
        min_val = curr[i] & curr[j]
        max_val = curr[i] | curr[j]
        curr[i] = min_val
        curr[j] = max_val
    
    # Check if sorted: for all adjacent wires i, i+1, we must have curr[i] <= curr[i+1]
    # In bit-parallel, this means: (curr[i] & ~curr[i+1]) == 0 for all i.
    for i in range(N - 1):
        if np.any(curr[i] & ~curr[i+1]):
            return False
    return True

def _odd_even_transposition_network():
    """Generate the standard 120-comparator Odd-Even Transposition network."""
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initialize random seed for reproducibility in local runs, 
    # but randomness is key for the search.
    # We don't strictly need a fixed seed for the evaluator, 
    # but it helps debugging. The evaluator runs once.
    
    # Start with a valid network
    current_net = _odd_even_transposition_network()
    if not _verify_network_bit_parallel(current_net):
        # Fallback if initialization failed (should not happen)
        return current_net

    best_net = current_net.copy()
    best_len = len(best_net)
    
    # Simulated Annealing Parameters
    # We want to explore the space of networks.
    # Cost = number of comparators.
    
    # Temperature schedule:
    # Start high to accept worse solutions, cool down to exploit.
    # T0 = 1.0, T_end = 0.0
    # We will run a fixed number of steps or until time runs out.
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    num_comps = len(all_comps)
    
    # For fast lookup of indices in the network list, we can just use list operations,
    # but for "remove" moves, we pick a random index.
    
    # Annealing Schedule
    # Let's run for a fixed number of iterations, checking time periodically.
    # Or run until deadline.
    
    T = 1.0  # Initial temperature
    cooling_rate = 0.9999  # Very slow cooling to explore more
    # Actually, let's use a time-based cooling or step-based.
    # Step-based is safer.
    # Let's estimate steps: ~1000-5000 steps per second. 
    # 85 seconds -> ~50,000 steps.
    # Let's just loop until deadline.
    
    while time.time() < deadline:
        # Generate a move
        # 1. Remove a random comparator
        # 2. Insert a random comparator
        # 3. Swap two comparators? (Less effective for size reduction)
        # 4. Double Remove
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Remove a random comparator
            if len(current_net) > 0:
                idx = random.randint(0, len(current_net) - 1)
                candidate = current_net[:idx] + current_net[idx+1:]
                is_valid = _verify_network_bit_parallel(candidate)
                new_len = len(candidate)
                delta_cost = new_len - len(current_net) # -1
                
                if is_valid:
                    # Always accept valid moves
                    current_net = candidate
                    if new_len < best_len:
                        best_len = new_len
                        best_net = candidate.copy()
                else:
                    # Reject invalid moves in SA for size minimization?
                    # Usually we only consider valid states in sorting network search.
                    # If we allow invalid states, we need a penalty.
                    # Let's stick to valid-only moves for simplicity and robustness.
                    # If we reject, we don't change state.
                    pass
            
        elif move_type < 0.8:
            # Insert a random comparator
            # Pick a random position and a random comparator
            # This allows escaping local minima where we are stuck with a valid
            # network that cannot be reduced by single removals.
            comp = random.choice(all_comps)
            pos = random.randint(0, len(current_net))
            candidate = current_net[:pos] + [comp] + current_net[pos:]
            
            # We must verify the candidate is valid.
            # If valid, we accept it with probability based on delta cost (which is +1, so worse).
            # P(accept) = exp(-delta_cost / T)
            is_valid = _verify_network_bit_parallel(candidate)
            if is_valid:
                delta_cost = 1
                if random.random() < np.exp(-delta_cost / T):
                    current_net = candidate
                    # Update best? No, this is worse. But it might lead to a better solution.
            
        else:
            # Double Remove
            # Try to remove two random comparators simultaneously
            if len(current_net) >= 2:
                idx1 = random.randint(0, len(current_net) - 1)
                idx2 = random.randint(0, len(current_net) - 1)
                if idx1 == idx2:
                    continue
                
                # Remove both
                # Sort indices to handle removal correctly
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                
                candidate = current_net[:idx2] + current_net[idx2+1:idx1] + current_net[idx1+1:]
                is_valid = _verify_network_bit_parallel(candidate)
                new_len = len(candidate)
                
                if is_valid:
                    current_net = candidate
                    if new_len < best_len:
                        best_len = new_len
                        best_net = candidate.copy()
            
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Prevent underflow, keep some exploration

    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Construct returned {len(net)} comparators.")
    print(f"Valid: {_verify_network_bit_parallel(net)}")
