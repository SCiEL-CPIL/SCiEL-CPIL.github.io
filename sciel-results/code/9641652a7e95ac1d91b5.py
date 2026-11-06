"""Size-efficient sorting networks on 18 wires.

Strategy:
1. Build a reasonably small initial network via divide-and-conquer (sort 9+9, merge).
2. Use bitwise (NumPy) verification over all 2^18 Boolean inputs to check correctness fast.
3. Run a local search / simulated annealing to remove or swap comparators to reduce count.
4. Deadline guard: return best valid network found before 85s.
"""

import time
import numpy as np
import random

N = 18
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _apply_network_vectorized(comps, states):
    """
    Apply the comparator network to a set of boolean states.
    states: array of shape (num_states, N) with 0/1 values.
    comps: list of (i, j) comparators.
    Returns the final states array.
    """
    for i, j in comps:
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b for booleans
        a = states[:, i]
        b = states[:, j]
        states[:, i] = a & b
        states[:, j] = a | b
    return states

def _verify_network(comps):
    """
    Verify that the network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Generate all 2^N boolean inputs
    # We can represent each input as an integer 0..2^N-1
    # But 2^18 = 262144, which is manageable.
    
    # Create all states
    all_states = np.zeros((1 << N, N), dtype=np.int8)
    for bit in range(N):
        # Set bit 'bit' to 1 for states where that bit is set
        mask = (1 << bit)
        # For state s, bit is set if (s & mask) != 0
        # We can vectorize this
        # states[:, bit] = (np.arange(1<<N) & mask) > 0
        all_states[:, bit] = ((np.arange(1 << N) & mask) > 0).astype(np.int8)
    
    # Apply network
    _apply_network_vectorized(comps, all_states)
    
    # Check if sorted: for each state, all 1s should be in higher indices
    # i.e., for each row, once we see a 0, all subsequent must be 0
    # Equivalently: the sorted version of a boolean vector has all 0s then all 1s
    # So state[:, k] >= state[:, k+1] for all k
    
    # Check: for each adjacent pair, state[:, i] >= state[:, i+1]
    # If any state has state[:, i] < state[:, i+1], it's not sorted
    for i in range(N - 1):
        if np.any(all_states[:, i] < all_states[:, i + 1]):
            return False
    return True

def _depth(comps):
    """Compute parallel depth of the network."""
    # This is a heuristic: count the number of "layers" where comparators can be parallelized
    # For simplicity, we'll just return the number of comparators as a proxy,
    # but the task says depth is reported separately, so we don't need to optimize it.
    return len(comps)

def _build_initial_network():
    """
    Build an initial network using divide-and-conquer:
    Sort first 9 wires, sort last 9 wires, then merge.
    """
    comps = []
    
    # Sort first 9 wires using odd-even transposition
    for phase in range(9):
        for i in range(phase % 2, 8, 2):
            comps.append((i, i + 1))
    
    # Sort last 9 wires using odd-even transposition
    for phase in range(9):
        for i in range(phase % 2, 8, 2):
            comps.append((9 + i, 9 + i + 1))
    
    # Merge the two sorted 9-wire sequences
    # Use a simple merge network: compare (i, 9+i) for i in 0..8, then fix up
    # Actually, a proper merge is needed. Let's use a bitonic merge or odd-even merge.
    # For simplicity, let's just apply odd-even transposition on all 18 wires after sorting halves.
    # This is not minimal but gives a valid network.
    # Better: use a known merge network.
    
    # Let's use a simple approach: after sorting halves, do a full odd-even transposition on all 18.
    # But that adds 153 - 72 = 81 more comparators, total 153, same as seed.
    
    # Instead, let's try a better merge. The merge of two sorted sequences of length n
    # can be done with a network. For n=9, a simple approach is to use the following:
    # Compare (i, 9+i) for i=0..8, then for i=0..7, compare (i, i+1) and (9+i, 9+i+1), etc.
    
    # Actually, let's just use the seed network (odd-even transposition for 18) as the initial,
    # then try to improve it with local search.
    
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    
    return comps

def _remove_comparator(comps, idx):
    """Remove comparator at index idx."""
    return comps[:idx] + comps[idx + 1:]

def _swap_comparators(comps, i, j):
    """Swap comparators at indices i and j."""
    new_comps = comps.copy()
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    return new_comps

def construct():
    start_time = time.time()
    
    # Build initial network
    comps = _build_initial_network()
    
    # Verify initial network
    if not _verify_network(comps):
        # Should not happen, but just in case
        return comps
    
    best_comps = comps
    best_count = len(comps)
    
    # Local search: try to remove comparators
    # Use a greedy approach with backtracking
    random.seed(42)
    
    current = comps.copy()
    current_count = len(current)
    
    # Try removing comparators one by one
    # Use a heuristic: try removing from the end first, or randomly
    max_iterations = 50000
    iteration = 0
    
    while time.time() - start_time < DEADLINE:
        iteration += 1
        if iteration > max_iterations:
            break
        
        # Try to remove a random comparator
        if current_count > 0:
            idx = random.randint(0, current_count - 1)
            new_comps = _remove_comparator(current, idx)
            
            # Verify
            if _verify_network(new_comps):
                current = new_comps
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_comps = current.copy()
            else:
                # Try swapping with another comparator
                if current_count > 1:
                    idx2 = random.randint(0, current_count - 1)
                    if idx2 != idx:
                        new_comps = _swap_comparators(current, idx, idx2)
                        if _verify_network(new_comps):
                            current = new_comps
                            # Count unchanged, but structure changed
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    return best_comps
