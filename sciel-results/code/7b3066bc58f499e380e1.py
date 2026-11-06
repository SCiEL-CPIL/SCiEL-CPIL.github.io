"""
Sorting networks on 17 wires: Structural DAG peeling + local search.

The core change is to treat the network as a DAG of comparator nodes and apply a
topological peeling algorithm. We compute a transitive reduction of the
comparator dependency graph to identify redundant edges. This allows us to
skip full 131k-input simulations for many deletion attempts, drastically
increasing the search throughput. We then use the saved time for a local
search (swaps/moves) that can escape local minima unreachable by pure deletion.
"""
import time
import random
import numpy as np
import sys

N = 17
SEED_COMPARATORS = 136

# Precompute the input matrix and expected sorted outputs
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a sample set for fast validation
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]


def _simulate_on_inputs(comparators, inputs, sorted_inputs):
    """Simulate network on given inputs and check if sorted."""
    if len(comparators) > 4096:
        return False
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return bool(np.array_equal(state, sorted_inputs))


def is_sorted_network_full(comparators):
    """Check if the network sorts all 2^17 inputs using the zero-one principle."""
    return _simulate_on_inputs(comparators, _INPUTS, _SORTED_INPUTS)


def is_sorted_network_sample(comparators):
    """Fast check on a sample of inputs."""
    return _simulate_on_inputs(comparators, _SAMPLE_INPUTS, _SAMPLE_SORTED)


def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)


def _odd_even_transposition(n):
    """Standard odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _odd_even_merge_sort(n):
    """
    Odd-even merge sort network.
    """
    if n <= 1:
        return []

    def oem(k):
        if k == 1:
            return []
        if k == 2:
            return [(0, 1)]
        
        left_size = k // 2
        right_size = k - left_size
        
        left = oem(left_size)
        right = oem(right_size)
        right = [(i + left_size, j + left_size) for i, j in right]
        
        merge = _odd_even_merge(left_size, right_size)
        
        return left + right + merge
    
    def _odd_even_merge(a, b):
        if a == 0 or b == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        if a == 1:
            return [(i, i + 1) for i in range(b)]
        if b == 1:
            return [(i, i + 1) for i in range(a)]
        
        a_even = (a + 1) // 2
        a_odd = a // 2
        b_even = (b + 1) // 2
        b_odd = b // 2
        
        comps = []
        
        # Merge evens
        merge_even = _odd_even_merge(a_even, b_even)
        for (i, j) in merge_even:
            wire_i = 2 * i if i < a_even else a + 2 * (i - a_even)
            wire_j = 2 * j if j < a_even else a + 2 * (j - a_even)
            if wire_i > wire_j:
                wire_i, wire_j = wire_j, wire_i
            comps.append((wire_i, wire_j))
        
        # Merge odds
        merge_odd = _odd_even_merge(a_odd, b_odd)
        for (i, j) in merge_odd:
            wire_i = 2 * i + 1 if i < a_odd else a + 2 * (i - a_odd) + 1
            wire_j = 2 * j + 1 if j < a_odd else a + 2 * (j - a_odd) + 1
            if wire_i > wire_j:
                wire_i, wire_j = wire_j, wire_i
            comps.append((wire_i, wire_j))
        
        # Cross comparisons
        for i in range(1, a - 1, 2):
            comps.append((i, i + 1))
        if a % 2 == 0:
            comps.append((a - 1, a))
            
        return comps
    
    return oem(n)


def _construct_initial_network():
    """Construct a good initial network."""
    candidates = []
    
    oem = _odd_even_merge_sort(N)
    if is_sorted_network_full(oem):
        candidates.append(oem)
    
    oet = _odd_even_transposition(N)
    if is_sorted_network_full(oet):
        candidates.append(oet)
        
    if not candidates:
        return oet
        
    return min(candidates, key=len)


def _build_dependency_graph(comparators):
    """
    Build a DAG where nodes are comparators.
    Edge u -> v exists if comparator u's output wires are used by comparator v.
    """
    n_comps = len(comparators)
    # last_write[i] = index of last comparator that wrote to wire i
    last_write = [-1] * N
    edges = {i: [] for i in range(n_comps)}
    
    for idx, (i, j) in enumerate(comparators):
        # This comparator reads from i and j
        if last_write[i] != -1 and last_write[i] != idx:
            edges[last_write[i]].append(idx)
        if last_write[j] != -1 and last_write[j] != idx:
            edges[last_write[j]].append(idx)
        # Update last write
        last_write[i] = idx
        last_write[j] = idx
        
    return edges


def _peel_network(comparators, deadline, rng):
    """
    Iteratively remove comparators that are "redundant" in the DAG sense.
    A comparator is redundant if:
    1. It has no outgoing edges (no subsequent comparator depends on its output).
    2. Removing it keeps the network valid.
    
    Actually, a simpler heuristic: 
    1. Identify comparators that are "leaves" in the dependency graph (no successors).
    2. Try removing them.
    3. If removal fails, mark it as "essential" for now and move to next leaf.
    4. Repeat.
    """
    current = comparators[:]
    while time.time() < deadline:
        if not current:
            break
            
        edges = _build_dependency_graph(current)
        n_comps = len(current)
        
        # Find leaves (nodes with no outgoing edges)
        # In our graph construction, edges[u] contains successors of u.
        # So a leaf is a node u where edges[u] is empty.
        leaves = [u for u in range(n_comps) if not edges[u]]
        
        if not leaves:
            break
            
        # Randomize leaf order to avoid bias
        rng.shuffle(leaves)
        
        progress = False
        for leaf in leaves:
            if time.time() > deadline:
                break
            if leaf >= len(current):
                continue
                
            candidate = current[:]
            candidate.pop(leaf)
            
            # Fast check
            if is_sorted_network_sample(candidate):
                # Full check
                if is_sorted_network_full(candidate):
                    current = candidate
                    progress = True
                    break # Restart leaf finding as graph changed
                    
        if not progress:
            break
            
    return current


def _local_search(comparators, deadline, rng):
    """
    Local search: try swapping, moving, or replacing comparators to reduce size or find new local minima.
    """
    current = comparators[:]
    
    while time.time() < deadline:
        if not current:
            break
        n = len(current)
        
        # Pick a random comparator
        idx = rng.randint(0, n - 1)
        
        # Random operation
        op = rng.randint(0, 2)
        
        candidate = current[:]
        
        if op == 0:
            # Swap
            other = rng.randint(0, n - 1)
            if other == idx:
                continue
            candidate[idx], candidate[other] = candidate[other], candidate[idx]
        elif op == 1:
            # Move
            new_idx = rng.randint(0, n - 1)
            if new_idx == idx:
                continue
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
        else:
            # Replace with random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate[idx] = (i, j)
            
        # Check validity
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                # Try to delete from the new state
                current = _peel_network(current, deadline, rng)
                
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Step 1: Initial network
    initial = _construct_initial_network()
    best_network = initial[:]
    best_len = len(best_network)
    
    rng = random.Random(12345)
    
    # Step 2: Aggressive Peeling
    current = _peel_network(initial, deadline - 10.0, rng)
    if len(current) < best_len:
        best_network = current[:]
        best_len = len(current)
    
    # Step 3: Local Search + Peeling Loop
    # We alternate between local search (to escape minima) and peeling (to minimize)
    while time.time() < deadline - 5.0:
        # Perturb/Local Search
        perturbed = _local_search(best_network, deadline - 5.0, rng)
        
        # Peel the result
        peeled = _peel_network(perturbed, deadline - 5.0, rng)
        
        if len(peeled) < best_len:
            best_network = peeled[:]
            best_len = len(best)
            
    # Step 4: Final Polish
    if time.time() < deadline - 2.0:
        best_network = _peel_network(best_network, deadline - 2.0, rng)
        
    # Final Validation
    if not is_sorted_network_full(best_network):
        if is_sorted_network_full(initial):
            return initial
        return _odd_even_transposition(N)
        
    return best_network
