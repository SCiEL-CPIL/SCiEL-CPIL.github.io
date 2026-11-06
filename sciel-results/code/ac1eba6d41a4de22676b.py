"""Size-efficient sorting networks on 18 wires.

Strategy:
1. Build a strong seed using Batcher odd-even mergesort (known to be near-optimal
   for power-of-2 sizes, and good for 18).
2. Aggressively remove redundant comparators via the 0-1 principle check.
3. Run simulated annealing with add/remove/move/swap moves to escape local minima.
4. Use a deadline guard to return the best valid network found within ~85s.
"""

import time
import random
import numpy as np

N = 18
DEADLINE = time.time() + 85  # 5s safety margin before 90s kill

# ---------- Utility: check if a network sorts all 0-1 inputs ----------

def _is_valid_network(net):
    """Check sorting network validity over all 2^N 0-1 inputs using 0-1 principle."""
    # We can speed this up: only need to check inputs where exactly k ones are set,
    # but for N=18 that's still sum(C(18,k)) = 2^18 = 262144.
    # Better: represent the network as a list of comparator pairs and simulate.
    # For speed, use bit-packing: each input is an 18-bit int.
    # A comparator (i, j) with i < j: if bit i is 1 and bit j is 0, swap them.
    
    n_comp = len(net)
    if n_comp == 0:
        return True
    
    # Pre-extract comparator pairs for faster access
    comps = [(i, j) for i, j in net]
    
    for mask in range(1 << N):
        x = mask
        for i, j in comps:
            # Check if bit i is 1 and bit j is 0
            if (x >> i) & 1 and not ((x >> j) & 1):
                # Swap: clear bit i, set bit j
                x = (x & ~(1 << i)) | (1 << j)
        # After sorting, all 1s should be in the highest positions
        # i.e., x should equal ((1 << N) - 1) ^ ((1 << (N - popcount)) - 1)
        # More simply: the sorted form has the top k bits set where k = popcount(x)
        k = bin(x).count('1')
        expected = ((1 << N) - 1) ^ ((1 << (N - k)) - 1) if k < N else (1 << N) - 1
        # Actually: top k bits set = ((1 << N) - 1) - ((1 << (N-k)) - 1)
        # = (1<<N) - 1 - (1<<(N-k)) + 1 = (1<<N) - (1<<(N-k))
        if k == 0:
            expected = 0
        else:
            expected = (1 << N) - (1 << (N - k))
        if x != expected:
            return False
    return True


def _is_valid_network_fast(net):
    """Faster validity check using numpy vectorization over all 2^N inputs."""
    if not net:
        return True
    
    # Use the 0-1 principle: check all 2^N inputs
    # For N=18, 2^18 = 262144 inputs, each with up to ~80 comparators
    # That's ~21M operations, should be fast enough with numpy
    
    n_comp = len(net)
    if n_comp == 0:
        return True
    
    # Generate all inputs as a column of 18-bit integers
    inputs = np.arange(1 << N, dtype=np.uint32)
    
    # Apply each comparator
    for i, j in net:
        # Extract bits i and j
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        # Where bit_i == 1 and bit_j == 0, we need to swap
        swap_mask = (bit_i == 1) & (bit_j == 0)
        # Clear bit i and set bit j where swap_mask is true
        inputs = np.where(swap_mask, (inputs & ~(1 << i)) | (1 << j), inputs)
    
    # Check sortedness: for each input, the number of 1s should be in the top k positions
    # Count 1s
    counts = np.array([bin(x).count('1') for x in inputs], dtype=np.uint8)
    # Expected: top k bits set
    # expected = (1<<N) - (1<<(N-k))
    expected = np.where(counts == 0, 0, (1 << N) - (1 << (N - counts)))
    
    return np.all(inputs == expected)


# ---------- Seed generation: Batcher odd-even mergesort ----------

def _batcher_odd_even_mergesort(n):
    """Generate Batcher's odd-even mergesort network for n inputs."""
    if n == 1:
        return []
    
    # Recursively sort odd and even indexed inputs
    odd_indices = [2 * i for i in range((n + 1) // 2)]
    even_indices = [2 * i + 1 for i in range(n // 2)]
    
    odd_net = _batcher_odd_even_mergesort(len(odd_indices))
    even_net = _batcher_odd_even_mergesort(len(even_indices))
    
    # Remap indices
    net = []
    for i, j in odd_net:
        net.append((odd_indices[i], odd_indices[j]))
    for i, j in even_net:
        net.append((even_indices[i], even_indices[j]))
    
    # Merge the two sorted halves
    net.extend(_batcher_odd_even_merge(len(odd_indices), len(even_indices)))
    
    return net


def _batcher_odd_even_merge(n1, n2):
    """Batcher's odd-even merge of two sorted sequences of lengths n1 and n2."""
    n = n1 + n2
    if n == 1:
        return []
    
    net = []
    # First level: compare (i, i+1) for odd i
    for i in range(1, n - 1, 2):
        net.append((i, i + 1))
    
    # Recursively merge even and odd sub-sequences
    # Even sub-sequence: indices 0, 2, 4, ...
    # Odd sub-sequence: indices 1, 3, 5, ...
    n_even = (n + 1) // 2
    n_odd = n // 2
    
    even_net = _batcher_odd_even_merge(n_even, n_odd) if n_even != n_odd else _batcher_odd_even_merge(n_even, n_odd)
    # Actually, the recursive merge is on the interleaved structure
    # Let me use the standard formulation:
    
    # The standard Batcher odd-even merge:
    # 1. Compare (1,2), (3,4), ... (0-indexed odd, even pairs)
    # 2. Recursively merge the even-indexed and odd-indexed sub-sequences
    # 3. Compare (2,3), (4,5), ... (0-indexed even, odd pairs)
    
    net = []
    # Step 1: compare (1,2), (3,4), ...
    for i in range(1, n - 1, 2):
        net.append((i, i + 1))
    
    # Step 2: recursively merge
    # Even sub-sequence has indices 0, 2, 4, ..., length n_even
    # Odd sub-sequence has indices 1, 3, 5, ..., length n_odd
    if n_even > 1:
        even_sub = _batcher_odd_even_merge(n_even, n_odd) if n_even != n_odd else []
        # Remap
        even_indices = [2 * i for i in range(n_even)]
        for i, j in even_sub:
            net.append((even_indices[i], even_indices[j]))
    
    if n_odd > 1:
        odd_sub = _batcher_odd_even_merge(n_even, n_odd) if n_even != n_odd else []
        odd_indices = [2 * i + 1 for i in range(n_odd)]
        for i, j in odd_sub:
            net.append((odd_indices[i], odd_indices[j]))
    
    # Step 3: compare (2,3), (4,5), ...
    for i in range(2, n - 1, 2):
        net.append((i, i + 1))
    
    return net


def _make_seed():
    """Create a seed sorting network."""
    # Try Batcher odd-even mergesort
    try:
        net = _batcher_odd_even_mergesort(N)
        if _is_valid_network_fast(net):
            return list(net)
    except Exception:
        pass
    
    # Fallback: odd-even transposition
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


# ---------- Greedy removal ----------

def _greedy_remove(net):
    """Remove redundant comparators one by one."""
    net = list(net)
    changed = True
    while changed:
        changed = False
        for idx in range(len(net)):
            candidate = net[:idx] + net[idx+1:]
            if _is_valid_network_fast(candidate):
                net = candidate
                changed = True
                break  # Restart from beginning after a removal
    return net


# ---------- Simulated Annealing ----------

def _sa_optimize(net, deadline):
    """Optimize network using simulated annealing."""
    best_net = list(net)
    best_len = len(net)
    current_net = list(net)
    current_len = len(net)
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if current_len <= best_len:
            best_net = list(current_net)
            best_len = current_len
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4 and current_net:
            # Remove a random comparator
            idx = rng.randrange(len(current_net))
            candidate = current_net[:idx] + current_net[idx+1:]
            delta = -1  # Length decreases by 1
        elif move_type < 0.6:
            # Add a random comparator
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            candidate = current_net + [(i, j)]
            delta = 1
        elif move_type < 0.8 and len(current_net) >= 2:
            # Move a comparator to a random position
            idx = rng.randrange(len(current_net))
            new_idx = rng.randrange(len(current_net))
            if idx == new_idx:
                continue
            comp = current_net[idx]
            candidate = current_net[:idx] + current_net[idx+1:]
            candidate.insert(new_idx, comp)
            delta = 0
        else:
            # Swap two comparators
            if len(current_net) < 2:
                continue
            idx1 = rng.randrange(len(current_net))
            idx2 = rng.randrange(len(current_net))
            if idx1 == idx2:
                continue
            candidate = list(current_net)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            delta = 0
        
        if len(candidate) > 4096:
            continue
        
        # Check validity
        valid = _is_valid_network_fast(candidate)
        
        if valid:
            if delta < 0 or (delta == 0 and rng.random() < 0.3):
                current_net = candidate
                current_len = len(candidate)
            elif delta > 0:
                # Accept with probability exp(-delta / T)
                if rng.random() < np.exp(-delta / T):
                    current_net = candidate
                    current_len = len(candidate)
        # If invalid, reject
        
        T = max(T * cooling, T_min)
    
    # Final greedy removal on best
    if time.time() < deadline - 5:
        best_net = _greedy_remove(best_net)
    
    return best_net


# ---------- Main construct function ----------

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Step 1: Generate seed
    net = _make_seed()
    best_net = list(net)
    best_len = len(net)
    
    # Step 2: Greedy removal
    if time.time() < deadline - 10:
        net = _greedy_remove(net)
        if len(net) < best_len:
            best_net = list(net)
            best_len = len(net)
    
    # Step 3: Simulated annealing
    if time.time() < deadline - 10:
        net = _sa_optimize(net, deadline - 5)
        if len(net) < best_len:
            best_net = list(net)
            best_len = len(net)
    
    # Step 4: Final greedy removal on the best
    if time.time() < deadline - 3:
        final_net = _greedy_remove(best_net)
        if len(final_net) < best_len:
            best_net = final_net
            best_len = len(final_net)
    
    return list(best_net)
