"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 80.0  # safety margin before 90s kill

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network(network):
    """
    Verify network correctness using the zero-one principle.
    Returns True if the network sorts all 2^16 boolean inputs.
    """
    # We need to check all 2^16 = 65536 boolean inputs.
    # For each input, after applying the network, the output must be sorted
    # (non-decreasing). For boolean inputs, this means all 0s come before all 1s.
    
    # Efficient approach: iterate through all masks
    # A boolean input is determined by which wires have 1s.
    # After sorting, the 1s must be on the highest-indexed wires.
    
    for mask in range(1 << N):
        x = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.uint8)
        _apply_network(network, x)
        # Check if sorted: x should be non-decreasing
        # For boolean, this means once we see a 1, all subsequent must be 1
        seen_one = False
        for val in x:
            if val == 1:
                seen_one = True
            elif seen_one:
                return False
    return True

def _verify_network_fast(network):
    """
    Faster verification using batch processing with numpy.
    Processes all 65536 boolean inputs in parallel batches.
    """
    n_inputs = 1 << N
    # Generate all boolean inputs as a matrix: (n_inputs, N)
    # Input k has 1s at positions where bit k of k is set
    indices = np.arange(n_inputs)
    # For each position i, the input has 1 if (indices >> i) & 1
    # Create a matrix where row k is the boolean representation of k
    inputs = np.zeros((n_inputs, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (indices >> i) & 1
    
    # Apply network to all inputs in parallel
    x = inputs.copy()
    for i, j in network:
        # Compare-exchange: min goes to i, max goes to j
        # For each row, if x[row, i] > x[row, j], swap
        mask = x[:, i] > x[:, j]
        x[mask, i] = 0
        x[mask, j] = 1
        # Where not masked, x[i] <= x[j] already, so no change needed
        # But we need to ensure min goes to i and max to j
        # If x[i] > x[j], after swap: x[i]=0, x[j]=1 (for boolean)
        # If x[i] <= x[j], no swap: x[i] stays, x[j] stays
    
    # Check if all outputs are sorted
    # For boolean sorted: once a 1 appears, all subsequent must be 1
    # Equivalent: for each row, x[:, i] <= x[:, i+1] for all i
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _verify_network_fastest(network):
    """
    Even faster: use the zero-one principle more cleverly.
    Instead of checking all 65536, we can use the fact that a sorting network
    must sort all boolean inputs. We'll batch process.
    """
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)
    
    # Build input matrix
    x = np.zeros((n_inputs, N), dtype=np.uint8)
    for i in range(N):
        x[:, i] = (indices >> i) & 1
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    # Depth is the length of the longest chain of dependent comparators.
    # Two comparators (i,j) and (k,l) are dependent if they share a wire.
    # We can model this as a DAG where each comparator is a node, and edges
    # go from comparator A to B if A and B share a wire and A comes before B.
    # The depth is the longest path in this DAG.
    
    if not network:
        return 0
    
    n = len(network)
    # For each comparator, track which wires it uses
    # Build dependency: comparator a depends on comparator b if they share a wire
    # and b is before a in the sequence.
    
    # Use a topological-like approach: for each comparator, compute its depth
    # as 1 + max(depth of all previous comparators that share a wire with it)
    
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires.
    Uses a known good construction as baseline, then attempts local search
    to reduce comparator count.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    # Start with odd-even transposition: 120 comparators
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_network = odd_even_transposition()
    best_count = len(best_network)
    
    # Try to improve via local search: remove redundant comparators
    # and try small perturbations.
    
    def is_valid(net):
        return _verify_network_fastest(net)
    
    # First, try removing comparators one by one (greedy removal)
    # Iterate through the network and try to remove each comparator
    # if the network still sorts.
    
    current = list(best_network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = current[:idx] + current[idx+1:]
            if is_valid(candidate):
                current = candidate
                changed = True
        if time.time() > deadline - 5:
            break
    
    best_network = current
    best_count = len(best_network)
    
    # Now try to improve further with local search:
    # Try swapping adjacent comparators, or replacing pairs.
    
    def try_local_search(net, max_iters=1000):
        """Try to reduce comparator count via local moves."""
        net = list(net)
        current_count = len(net)
        best = list(net)
        best_count = current_count
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(net) < best_count:
                best = list(net)
                best_count = len(net)
            
            # Try removing a random comparator
            if len(net) > 1:
                idx = rng.randint(0, len(net) - 1)
                candidate = net[:idx] + net[idx+1:]
                if is_valid(candidate):
                    net = candidate
                    continue
            
            # Try swapping two adjacent comparators
            if len(net) >= 2:
                idx = rng.randint(0, len(net) - 2)
                candidate = net[:]
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if is_valid(candidate):
                    net = candidate
            
            # Try replacing a comparator with a different one
            if len(net) > 0:
                idx = rng.randint(0, len(net) - 1)
                i, j = net[idx]
                # Try a random different comparator
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(new_i + 1, N - 1)
                if (new_i, new_j) != (i, j):
                    candidate = net[:]
                    candidate[idx] = (new_i, new_j)
                    if is_valid(candidate):
                        net = candidate
        
        return best, best_count
    
    # Run a few rounds of local search
    for round_num in range(5):
        if time.time() > deadline - 10:
            break
        improved_net, improved_count = try_local_search(best_network, max_iters=500)
        if improved_count < best_count:
            best_network = improved_net
            best_count = improved_count
    
    # Also try a different starting point: a known smaller network
    # For 16 wires, there are known constructions. Let's try a few heuristics.
    
    # Attempt: use a merge-based construction
    def merge_network(left_net, right_net, left_size, right_size):
        """Merge two sorted networks."""
        # This is a simplified merge; not optimal but may help.
        pass  # Skip complex merge for now
    
    # Try starting from a random valid network and optimizing
    if time.time() < deadline - 20:
        # Generate a random network and see if we can find something better
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            # Start with a larger random network and prune
            random_net = []
            for _ in range(150):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                random_net.append((i, j))
            # Prune
            pruned = list(random_net)
            changed = True
            while changed:
                changed = False
                for idx in range(len(pruned) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    candidate = pruned[:idx] + pruned[idx+1:]
                    if is_valid(candidate):
                        pruned = candidate
                        changed = True
            if len(pruned) < best_count:
                best_network = pruned
                best_count = len(pruned)
    
    return best_network
# EVOLVE-BLOCK-END
