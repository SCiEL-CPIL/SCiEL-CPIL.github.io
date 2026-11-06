"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _generate_all_inputs(n=N):
    """Generate all 2^n boolean inputs as a uint8 array of shape (2^n, n)."""
    x = np.arange(1 << n, dtype=np.uint64)[:, None]
    bits = (x >> np.arange(n, dtype=np.uint64)) & 1
    return bits.astype(np.uint8)

_ALL_INPUTS = _generate_all_inputs(N)
_NUM_INPUTS = 1 << N

def _verify_network_fast(comparators):
    """Vectorized verification: check if network sorts all 2^N inputs."""
    if not comparators:
        return True
    state = _ALL_INPUTS.copy()
    for i, j in comparators:
        vi = state[:, i]
        vj = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = vi & vj
        state[:, j] = vi | vj
    # Check sorted: no 1 before 0 in each row
    # For binary: state[:, k] <= state[:, k+1] for all k
    return bool(np.all(state[:, :-1] <= state[:, 1:]))

def _count_disorder(state):
    """Count number of inversions (1-before-0 pairs) across all inputs."""
    # For each row, count pairs (i,j) with i<j and state[i]=1, state[j]=0
    # Vectorized: for each position, count how many later positions are 0
    total = 0
    n = state.shape[1]
    for i in range(n - 1):
        # state[:, i] == 1 and state[:, j] == 0 for j > i
        ones_at_i = state[:, i]  # 1 where wire i has 1
        zeros_after = 1 - state[:, i+1:]  # 1 where wire j has 0
        total += np.sum(ones_at_i[:, None] * zeros_after)
    return total

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _build_initial_network():
    """Build a good initial network using odd-even transposition."""
    return _odd_even_transposition(N)

def _remove_redundant(comparators, deadline):
    """Try to remove each comparator if the network remains valid."""
    current = list(comparators)
    n = len(current)
    # Try removing from the end first (often less critical)
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate = current[:idx] + current[idx+1:]
        if _verify_network_fast(candidate):
            current = candidate
    return current

def _random_comparator():
    """Generate a random valid comparator."""
    a = random.randint(0, N - 1)
    b = random.randint(0, N - 1)
    while a == b:
        b = random.randint(0, N - 1)
    if a > b:
        a, b = b, a
    return (a, b)

def _local_search_remove(comparators, deadline, max_iters=5000):
    """Local search focused on removal."""
    current = list(comparators)
    best = list(current)
    best_len = len(current)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        
        # Try removing a random comparator
        idx = random.randint(0, len(current) - 1)
        candidate = current[:idx] + current[idx+1:]
        if _verify_network_fast(candidate):
            current = candidate
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
                # Greedy cascade: try more removals
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    if len(current) <= 70:
                        break
                    idx2 = random.randint(0, len(current) - 1)
                    candidate2 = current[:idx2] + current[idx2+1:]
                    if _verify_network_fast(candidate2):
                        current = candidate2
                        if len(current) < best_len:
                            best = list(current)
                            best_len = len(current)
    
    return best

def _local_search_replace(comparators, deadline, max_iters=5000):
    """Local search focused on replacement to enable further removals."""
    current = list(comparators)
    best = list(current)
    best_len = len(current)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        
        # Replace a random comparator with a different one
        if len(current) == 0:
            continue
        idx = random.randint(0, len(current) - 1)
        new_comp = _random_comparator()
        if new_comp == current[idx]:
            continue
        candidate = list(current)
        candidate[idx] = new_comp
        if _verify_network_fast(candidate):
            current = candidate
            # Try to remove something after replacement
            for _ in range(3):
                if time.time() > deadline:
                    break
                if len(current) <= 70:
                    break
                idx2 = random.randint(0, len(current) - 1)
                candidate2 = current[:idx2] + current[idx2+1:]
                if _verify_network_fast(candidate2):
                    current = candidate2
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
    
    return best

def _layered_construction(deadline):
    """
    Construct network layer by layer.
    Each layer is a set of comparators that can be executed in parallel.
    We build layers that progressively sort the network.
    """
    # Start with a reasonable number of layers
    # Odd-even transposition has N layers of ~N/2 comparators each
    # We'll try to build with fewer, denser layers
    
    network = []
    remaining_time_budget = 40.0  # seconds for this phase
    
    # Phase 1: Build a coarse network
    # Use a heuristic: add comparators that reduce disorder most
    state = _ALL_INPUTS.copy()
    disorder = _count_disorder(state)
    
    max_layers = 20
    for layer in range(max_layers):
        if time.time() > deadline:
            break
        if disorder == 0:
            break
        
        # Find the comparator that most reduces disorder
        best_comp = None
        best_reduction = 0
        
        # Sample a subset of possible comparators for speed
        candidates = []
        for _ in range(200):
            candidates.append(_random_comparator())
        # Also include adjacent comparators
        for i in range(N - 1):
            candidates.append((i, i + 1))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        for comp in candidates:
            if time.time() > deadline:
                break
            i, j = comp
            # Simulate this comparator on current state
            vi = state[:, i]
            vj = state[:, j]
            new_vi = vi & vj
            new_vj = vi | vj
            
            # Compute disorder reduction
            # Only need to check pairs involving i or j
            old_disorder = 0
            new_disorder = 0
            
            # Pairs involving wire i
            for k in range(N):
                if k == i:
                    continue
                if k < i:
                    # pair (k, i): old is (state[k], state[i]), new is (state[k], new_vi)
                    old_disorder += np.sum((state[:, k] == 1) & (state[:, i] == 0))
                    new_disorder += np.sum((state[:, k] == 1) & (new_vi == 0))
                else:
                    # pair (i, k): old is (state[i], state[k]), new is (new_vi, state[k])
                    old_disorder += np.sum((state[:, i] == 1) & (state[:, k] == 0))
                    new_disorder += np.sum((new_vi == 1) & (state[:, k] == 0))
            
            # Pairs involving wire j
            for k in range(N):
                if k == j:
                    continue
                if k < j:
                    # pair (k, j): old is (state[k], state[j]), new is (state[k], new_vj)
                    old_disorder += np.sum((state[:, k] == 1) & (state[:, j] == 0))
                    new_disorder += np.sum((state[:, k] == 1) & (new_vj == 0))
                else:
                    # pair (j, k): old is (state[j], state[k]), new is (new_vj, state[k])
                    old_disorder += np.sum((state[:, j] == 1) & (state[:, k] == 0))
                    new_disorder += np.sum((new_vj == 1) & (state[:, k] == 0))
            
            reduction = old_disorder - new_disorder
            if reduction > best_reduction:
                best_reduction = reduction
                best_comp = comp
        
        if best_comp is None or best_reduction <= 0:
            break
        
        # Apply the best comparator
        i, j = best_comp
        vi = state[:, i]
        vj = state[:, j]
        state[:, i] = vi & vj
        state[:, j] = vi | vj
        network.append(best_comp)
        disorder = _count_disorder(state)
    
    if _verify_network_fast(network):
        return network
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    random.seed(42)
    
    # Start with odd-even transposition
    current = _build_initial_network()
    best = list(current)
    best_len = len(current)
    
    # Verify initial
    if not _verify_network_fast(current):
        return best
    
    # Phase 1: Greedy removal from the end
    if time.time() < deadline - 30:
        current = _remove_redundant(current, deadline)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    # Phase 2: Random local search with removal
    if time.time() < deadline - 20:
        result = _local_search_remove(current, deadline, max_iters=10000)
        if len(result) < best_len:
            best = result
            best_len = len(result)
            current = result
    
    # Phase 3: Random local search with replacement
    if time.time() < deadline - 10:
        result = _local_search_replace(current, deadline, max_iters=5000)
        if len(result) < best_len:
            best = result
            best_len = len(result)
            current = result
    
    # Phase 4: Try layered construction
    if time.time() < deadline - 5:
        layered = _layered_construction(deadline)
        if layered is not None and len(layered) < best_len:
            best = layered
            best_len = len(layered)
    
    # Final: one more pass of greedy removal
    if time.time() < deadline - 2:
        final = _remove_redundant(best, deadline)
        if len(final) < best_len:
            best = final
            best_len = len(final)
    
    return best

# EVOLVE-BLOCK-END
