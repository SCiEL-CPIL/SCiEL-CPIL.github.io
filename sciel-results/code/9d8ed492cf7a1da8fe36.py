"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

N = 17

# Precompute the 17 bitmasks for zero-one principle (2^17 = 131072)
# We represent a 17-bit pattern as an integer 0..131071
# A wire configuration is the sorted version of the input.
# For zero-one, the network is correct if for every subset S of wires,
# applying the network to the indicator vector of S yields the same multiset
# (i.e., the number of 1s is preserved and the output is the sorted vector).
# Actually, the standard zero-one principle says: it suffices to check all
# 2^N boolean inputs. For each input (a bit vector), run the network and
# check if the output is sorted (non-decreasing).

# Optimization: represent state as a 17-bit integer.
# Comparator (i, j): if bit i > bit j, swap them.
# In integer representation: bit i is (state >> i) & 1, bit j is (state >> j) & 1.
# If bit_i == 1 and bit_j == 0, we need to set bit_i to 0 and bit_j to 1.

def apply_network(state, network):
    """Apply the network to a 17-bit state (integer). Returns new state."""
    for i, j in network:
        bi = (state >> i) & 1
        bj = (state >> j) & 1
        if bi == 1 and bj == 0:
            state ^= (1 << i) | (1 << j)
    return state

def is_sorted_state(state):
    """Check if the 17-bit state represents a sorted (non-decreasing) wire assignment.
    Sorted means all 0s come before all 1s.
    So the state should be of the form: 0...0 1...1
    i.e., if there are k ones, the lowest k bits are 1 and the rest are 0?
    Wait. Wire 0 is the "first" wire. Sorted non-decreasing means wire 0 <= wire 1 <= ... <= wire 16.
    So if we have k ones, they must be on the last k wires: wires N-k through N-1.
    So the state should be: bits N-k to N-1 are 1, bits 0 to N-k-1 are 0.
    Equivalently, the state is a contiguous block of 1s at the high end.
    """
    # A state is sorted if it is of the form 0^a 1^b where a+b=N.
    # This means the state is either 0, or of the form 2^N - 2^a for some a in 0..N.
    # Actually, let's just check: find the position of the first 1 from the left?
    # No, easier: a sorted 0-1 sequence has the property that if you see a 0, all previous are 0.
    # Let's precompute all valid sorted states.
    # There are N+1 = 18 such states.
    # State with k ones: the ones are on wires N-k, ..., N-1.
    # Value = sum_{i=N-k}^{N-1} 2^i = 2^N - 2^(N-k)
    # We'll check against a precomputed set.
    return state in _SORTED_STATES

_SORTED_STATES = set()
for k in range(N + 1):
    # k ones, on the top k wires
    val = 0
    for i in range(N - k, N):
        val |= (1 << i)
    _SORTED_STATES.add(val)

def verify_network(network):
    """Verify that the network sorts all 2^17 inputs using the zero-one principle."""
    for state in range(1 << N):
        out = apply_network(state, network)
        if out not in _SORTED_STATES:
            return False
    return True

def verify_network_fast(network):
    """Vectorized verification using numpy. Much faster than Python loop."""
    # Precompute all 2^17 states as a numpy array
    all_states = np.arange(1 << N, dtype=np.uint32)
    # Apply network to all states at once
    for i, j in network:
        bi = (all_states >> i) & 1
        bj = (all_states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        all_states[mask] ^= (1 << i) | (1 << j)
    # Check if all outputs are sorted
    # A state is sorted if it's in _SORTED_STATES
    # Create a lookup: for each state, is it sorted?
    # We can check: sorted states have the form where all 1s are at the high end
    # i.e., state & (state + 1) == 0 is NOT the right test...
    # Let's just check against the set using numpy
    sorted_arr = np.array(sorted(_SORTED_STATES), dtype=np.uint32)
    # Check membership
    for s in all_states:
        if s not in _SORTED_STATES:
            return False
    return True

def verify_network_numpy(network):
    """Fast verification using numpy vectorization."""
    all_states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        bi = (all_states >> i) & 1
        bj = (all_states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        all_states[mask] ^= (1 << i) | (1 << j)
    # Check sorted: a 17-bit state is sorted (non-decreasing) iff
    # it has the form 0...01...1, i.e., no 0 appears after a 1.
    # This is equivalent to: for each bit position, if bit k is 0, all higher bits must be 0.
    # Or: the state is of the form (2^N - 1) ^ (2^a - 1) for some a in 0..N
    # i.e., state == (2^N - 1) - (2^a - 1) = 2^N - 2^a
    # So valid states are: 0, 2^17-2^0, 2^17-2^1, ..., 2^17-2^16
    # Let's check: state must equal 2^N - 2^a for some a, or state == 0 (a=N)
    # Actually 2^17 - 2^17 = 0, so a ranges 0 to 17.
    # We can check: (state + 1) & ~state == 0 is the test for "all 1s then all 0s" from MSB
    # Wait, let me think differently.
    # A number is of the form 0^a 1^b (b ones at low end) iff...
    # No, sorted non-decreasing means 0s first, 1s last.
    # So wire 0 is lowest, wire 16 is highest.
    # Sorted: wire0 <= wire1 <= ... <= wire16.
    # With 0/1 values, this means some prefix of 0s and suffix of 1s.
    # If k ones, they are on wires 17-k through 16.
    # Value = 2^17 - 2^(17-k) for k=0..17.
    # For k=0: value = 0
    # For k=1: value = 2^17 - 2^16 = 2^16
    # For k=17: value = 2^17 - 1
    # So valid values are: {2^17 - 2^(17-k) : k=0..17}
    # Let's precompute a boolean lookup array
    valid = np.zeros(1 << N, dtype=bool)
    for k in range(N + 1):
        val = (1 << N) - (1 << (N - k)) if k < N else 0
        valid[val] = True
    # Wait, for k=0: 2^17 - 2^17 = 0. For k=N: 2^17 - 2^0 = 2^17-1.
    # Let me recompute:
    for k in range(N + 1):
        if k == 0:
            val = 0
        else:
            val = (1 << N) - (1 << (N - k))
        valid[val] = True
    
    if not np.all(valid[all_states]):
        return False
    return True

def count_comparators(network):
    return len(network)

def compute_depth(network):
    """Compute the parallel depth of the network.
    Depth is the minimum number of parallel steps needed.
    Two comparators (i,j) and (k,l) can be parallel if they are disjoint (i != k, i != l, j != k, j != l).
    We use a greedy layering: assign each comparator to the earliest layer where it doesn't conflict.
    This is an approximation; the true minimum depth might be lower, but for scoring we just report it.
    Actually, the task says "parallel depth is reported separately", and the score is comparator count.
    So depth doesn't affect the score, but we'll compute it for reporting.
    """
    if not network:
        return 0
    layers = []
    # Map from comparator to the earliest layer it can be placed in
    # We process comparators in order. For each comparator, find the first layer
    # where no comparator in that layer shares a wire with it.
    # A wire is "free" in a layer if no comparator in that layer uses it.
    for comp in network:
        placed = False
        for layer_idx in range(len(layers)):
            wires_in_layer = set()
            for c in layers[layer_idx]:
                wires_in_layer.add(c[0])
                wires_in_layer.add(c[1])
            if comp[0] not in wires_in_layer and comp[1] not in wires_in_layer:
                layers[layer_idx].append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return len(layers)

# Seed network: odd-even transposition
def seed_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# Known good networks from literature for small N
# For N=17, the best known is 71 comparators. We don't have it memorized,
# so we'll rely on search.

# Search approach:
# 1. Start with a valid network (seed or known).
# 2. Try to remove comparators one by one, checking validity.
# 3. Try local search: replace a comparator with another, or reorder.
# 4. Use randomized search with restarts.

def try_remove_comparator(network, idx):
    """Try removing comparator at idx. Return new network if valid, else None."""
    new_net = network[:idx] + network[idx+1:]
    if verify_network(new_net):
        return new_net
    return None

def try_replace_comparator(network, idx, new_comp):
    """Try replacing comparator at idx with new_comp. Return new network if valid, else None."""
    new_net = network[:idx] + [new_comp] + network[idx+1:]
    if verify_network(new_net):
        return new_net
    return None

def local_search_remove(network, time_limit, rng):
    """Try to remove as many comparators as possible using greedy removal."""
    current = list(network)
    removed = 0
    # Try removing from the end first (often easier)
    # Use a random order for robustness
    indices = list(range(len(current)))
    rng.shuffle(indices)
    
    for idx in indices:
        if time.time() > time_limit:
            break
        # Adjust idx to account for previous removals
        # Actually, it's easier to rebuild the list.
        # Let's just iterate through a copy and try to remove each.
        pass
    
    # Better approach: repeatedly try to remove any comparator until none can be removed.
    progress = True
    while progress and time.time() < time_limit:
        progress = False
        # Shuffle the order to break symmetry
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > time_limit:
                break
            new_net = try_remove_comparator(current, idx)
            if new_net is not None:
                current = new_net
                progress = True
                break
    return current

def random_search_add_remove(network, time_limit, rng):
    """Random search: try adding random comparators and then removing."""
    # This is expensive. Instead, let's do a different approach.
    # We'll keep this simple for now.
    return network

# Main construct function
# EVOLVE-BLOCK-START
def construct():
    # Deadline: 90s total, but we must return before 60s as per task.
    # Let's use 55s as our safety margin.
    start_time = time.time()
    deadline = start_time + 55.0
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Start with the seed network
    best_net = seed_network()
    best_count = len(best_net)
    
    # Phase 1: Greedy removal with multiple restarts
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        current = list(best_net)
        
        # Multiple passes of greedy removal with different random orders
        for pass_num in range(50):
            if time.time() > deadline:
                break
            
            progress = False
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                new_net = try_remove_comparator(current, idx)
                if new_net is not None:
                    current = new_net
                    progress = True
                    break
            
            if not progress:
                break
        
        if len(current) < best_count:
            best_net = current
            best_count = len(current)
            print(f"Phase 1: Improved to {best_count} comparators", flush=True)
    
    # Phase 2: Local search with replacement and reordering
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        current = list(best_net)
        improved_this_iter = False
        
        # Try random replacements
        for _ in range(30):
            if time.time() > deadline:
                break
            
            if len(current) < 2:
                break
            
            idx = rng.randint(0, len(current) - 1)
            
            # Generate a random alternative comparator
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            
            new_comp = (new_i, new_j)
            
            new_net = try_replace_comparator(current, idx, new_comp)
            if new_net is not None:
                # Try to remove some comparators
                progress = True
                while progress and time.time() < deadline:
                    progress = False
                    indices = list(range(len(new_net)))
                    rng.shuffle(indices)
                    for rem_idx in indices:
                        if time.time() > deadline:
                            break
                        reduced = try_remove_comparator(new_net, rem_idx)
                        if reduced is not None:
                            new_net = reduced
                            break
                
                if len(new_net) < best_count:
                    best_net = new_net
                    best_count = len(new_net)
                    improved_this_iter = True
                    print(f"Phase 2: Improved to {best_count} comparators", flush=True)
                else:
                    current = new_net
        
        # Try swapping adjacent comparators
        if not improved_this_iter and len(current) > 1:
            idx = rng.randint(0, len(current) - 2)
            new_net = list(current)
            new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
            if verify_network(new_net):
                # Try to remove
                progress = True
                while progress and time.time() < deadline:
                    progress = False
                    indices = list(range(len(new_net)))
                    rng.shuffle(indices)
                    for rem_idx in indices:
                        if time.time() > deadline:
                            break
                        reduced = try_remove_comparator(new_net, rem_idx)
                        if reduced is not None:
                            new_net = reduced
                            break
                
                if len(new_net) < best_count:
                    best_net = new_net
                    best_count = len(new_net)
                    print(f"Phase 2 (swap): Improved to {best_count} comparators", flush=True)
    
    # Phase 3: Simulated annealing style - allow temporary worsening
    current = list(best_net)
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try removal
            if len(current) > 1:
                idx = rng.randint(0, len(current) - 1)
                new_net = current[:idx] + current[idx+1:]
                if verify_network(new_net):
                    current = new_net
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
                        print(f"Phase 3: Improved to {best_count} comparators", flush=True)
        
        elif move_type < 0.7:
            # Try replacement
            if len(current) > 0:
                idx = rng.randint(0, len(current) - 1)
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(0, N - 1)
                if new_i != new_j:
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    new_net = current[:idx] + [(new_i, new_j)] + current[idx+1:]
                    if verify_network(new_net):
                        current = new_net
                        if len(current) < best_count:
                            best_net = current
                            best_count = len(current)
                            print(f"Phase 3: Improved to {best_count} comparators", flush=True)
        
        else:
            # Try adding a comparator (to enable future removals)
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i != new_j:
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                pos = rng.randint(0, len(current))
                new_net = current[:pos] + [(new_i, new_j)] + current[pos:]
                if verify_network(new_net):
                    current = new_net
        
        # After each move, try to remove
        if time.time() < deadline:
            progress = True
            while progress and time.time() < deadline:
                progress = False
                indices = list(range(len(current)))
                rng.shuffle(indices)
                for rem_idx in indices:
                    if time.time() > deadline:
                        break
                    reduced = try_remove_comparator(current, rem_idx)
                    if reduced is not None:
                        current = reduced
                        progress = True
                        break
            
            if len(current) < best_count:
                best_net = current
                best_count = len(current)
                print(f"Phase 3: Improved to {best_count} comparators", flush=True)
    
    # Final check: verify the best network
    if not verify_network(best_net):
        # Fallback to seed
        best_net = seed_network()
    
    return best_net
# EVOLVE-BLOCK-END
