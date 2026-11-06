"""Size-efficient sorting network search on 18 wires."""
import time
import random
import numpy as np

N = 18
NUM_COMPARATORS_LIMIT = 4096
EVAL_DEADLINE = 75  # seconds, safety margin before 90s kill

def _apply_network(nets, x):
    """Apply a list of comparator pairs to array x in-place.
    Each comparator (i, j) ensures x[i] <= x[j]."""
    for i, j in nets:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted(x):
    """Check if x is sorted in non-decreasing order."""
    return all(x[i] <= x[i+1] for i in range(len(x)-1))

def _verify_network(nets, timeout=5.0):
    """Verify that the network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise or on timeout."""
    start = time.time()
    n_bools = 1 << N
    # Use numpy for speed: represent each input as a bit vector
    # We'll check all 2^18 = 262144 inputs
    for mask in range(n_bools):
        if time.time() - start > timeout:
            return False
        # Build the boolean input
        x = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        # Apply network
        for i, j in nets:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        # Check sorted
        if not _is_sorted(x):
            return False
    return True

def _verify_network_fast(nets, timeout=5.0):
    """Faster verification using bit-parallel approach where possible.
    For 18 wires, 2^18 = 262144 inputs. We process them in batches."""
    start = time.time()
    n_bools = 1 << N
    
    # Pre-extract comparator indices
    comps = [(int(i), int(j)) for i, j in nets]
    
    for mask in range(n_bools):
        if time.time() - start > timeout:
            return False
        # Unpack mask to list
        x = [(mask >> i) & 1 for i in range(N)]
        # Apply network
        for i, j in comps:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        # Check sorted
        for k in range(N - 1):
            if x[k] > x[k+1]:
                return False
    return True

def _depth_of_network(nets):
    """Compute parallel depth of a sorting network."""
    if not nets:
        return 0
    # Simple upper bound: number of comparators (sequential)
    # Better: simulate layers
    # For each wire, track last time it was used
    last_used = [0] * N
    depth = 0
    for i, j in nets:
        # A comparator can start after both wires are free
        start = max(last_used[i], last_used[j])
        end = start + 1
        depth = max(depth, end)
        last_used[i] = end
        last_used[j] = end
    return depth

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _random_network(num_comparators):
    """Generate a random sorting network candidate."""
    nets = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        nets.append((i, j))
    return nets

def _improve_network(nets, max_iterations=2000, timeout=5.0):
    """Try to improve a network by local search: swap, insert, delete comparators."""
    start = time.time()
    best_nets = list(nets)
    best_size = len(best_nets)
    
    for _ in range(max_iterations):
        if time.time() - start > timeout:
            break
        
        # Random mutation
        mutated = list(best_nets)
        op = random.random()
        
        if op < 0.3 and mutated:
            # Delete a random comparator
            idx = random.randint(0, len(mutated) - 1)
            mutated.pop(idx)
        elif op < 0.6:
            # Insert a random comparator at random position
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(mutated))
            mutated.insert(pos, (i, j))
        elif op < 0.8:
            # Swap two comparators
            if len(mutated) >= 2:
                idx1 = random.randint(0, len(mutated) - 1)
                idx2 = random.randint(0, len(mutated) - 1)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        else:
            # Change a comparator's wires
            if mutated:
                idx = random.randint(0, len(mutated) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                mutated[idx] = (i, j)
        
        if len(mutated) > NUM_COMPARATORS_LIMIT:
            continue
        
        # Verify
        if _verify_network_fast(mutated, timeout=timeout/2):
            best_nets = mutated
            best_size = len(mutated)
    
    return best_nets

def construct():
    """Construct a sorting network on 18 wires with as few comparators as possible."""
    global_start = time.time()
    deadline = global_start + EVAL_DEADLINE
    
    # Start with seed network
    seed = _seed_network()
    best_nets = list(seed)
    best_size = len(best_nets)
    
    # Try various strategies
    strategies = []
    
    # Strategy 1: Local search from seed
    def strategy_local_search():
        nonlocal best_nets, best_size
        if time.time() > deadline - 10:
            return
        improved = _improve_network(best_nets, max_iterations=5000, timeout=deadline - time.time() - 5)
        if len(improved) < best_size:
            best_nets = improved
            best_size = len(best_nets)
    
    # Strategy 2: Random restarts with smaller initial networks
    def strategy_random_restarts():
        nonlocal best_nets, best_size
        while time.time() < deadline - 15:
            # Try random networks of various sizes
            for size in [80, 85, 90, 95, 100]:
                if time.time() > deadline - 15:
                    break
                if size >= best_size:
                    continue
                nets = _random_network(size)
                if _verify_network_fast(nets, timeout=2.0):
                    if size < best_size:
                        best_nets = nets
                        best_size = size
    
    # Strategy 3: Build up from smaller networks
    def strategy_build_up():
        nonlocal best_nets, best_size
        if time.time() > deadline - 10:
            return
        # Start with a known small network and try to extend
        # Use a simpler base: bitonic sort for 16 wires + extensions
        # Bitonic sort for 16 wires uses 60 comparators
        # For 18, we can merge 16+2 or 9+9 etc.
        
        # Try a merge-based approach: sort 9+9, then merge
        # Bitonic merge of two sorted 9-element sequences
        # This is complex; let's try a different approach
        
        # Use the seed but try to prune
        if len(best_nets) > 77:
            # Try to remove comparators one at a time
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                if not best_nets:
                    break
                idx = random.randint(0, len(best_nets) - 1)
                candidate = best_nets[:idx] + best_nets[idx+1:]
                if _verify_network_fast(candidate, timeout=3.0):
                    best_nets = candidate
                    best_size = len(best_nets)
    
    # Execute strategies in order of expected value
    strategy_build_up()
    
    if time.time() < deadline - 20:
        strategy_local_search()
    
    if time.time() < deadline - 5:
        strategy_random_restarts()
    
    # Final check: make sure we have a valid network
    if not _verify_network_fast(best_nets, timeout=5.0):
        # Fallback to seed
        best_nets = _seed_network()
        best_size = len(best_nets)
    
    return best_nets

# Ensure the function is accessible
if __name__ == "__main__":
    nets = construct()
    print(f"Network size: {len(nets)}")
    print(f"Depth: {_depth_of_network(nets)}")
