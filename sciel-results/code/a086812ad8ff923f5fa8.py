"""Sorting network construction for N=18 wires with time-bounded search."""
import time
import random

N = 18
SEED_COMPARATORS = 153
DEADLINE_SECONDS = 85.0

def _odd_even_transposition():
    """Generate the seed odd-even transposition network for N wires."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _is_sorted_state(state):
    """Check if an 18-bit state represents a sorted sequence (0^a 1^b)."""
    # Sorted means all 0s come before all 1s (wire 0 is LSB)
    # So state must be of the form: lower a bits are 0, upper (18-a) bits are 1
    # Equivalently, state & (state + 1) == 0 is NOT the right check.
    # The valid states are: for a in 0..18: ((1 << (18-a)) - 1) << a
    # Precompute these
    # But for speed, we can check: let count_ones = popcount(state)
    # Then the sorted form with that many ones is: ((1 << count_ones) - 1) << (18 - count_ones)
    # So state is sorted iff state == ((1 << popcount(state)) - 1) << (18 - popcount(state))
    ones = bin(state).count('1')
    expected = ((1 << ones) - 1) << (N - ones) if ones > 0 else 0
    return state == expected

def _verify_network(comparators):
    """Verify that the network sorts all 2^18 inputs."""
    size = 1 << N
    for x in range(size):
        state = x
        for i, j in comparators:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state ^= (1 << i) | (1 << j)
        if not _is_sorted_state(state):
            return False
    return True

def _verify_network_fast(comparators):
    """Faster verification using precomputed bit operations."""
    size = 1 << N
    # Precompute masks
    masks = [(1 << i) | (1 << j) for i, j in comparators]
    is_list = [i for i, j in comparators]
    js_list = [j for i, j in comparators]
    
    for x in range(size):
        state = x
        for k in range(len(comparators)):
            i = is_list[k]
            j = js_list[k]
            if ((state >> i) & 1) > ((state >> j) & 1):
                state ^= masks[k]
        # Check sorted
        ones = bin(state).count('1')
        if ones > 0:
            expected = ((1 << ones) - 1) << (N - ones)
        else:
            expected = 0
        if state != expected:
            return False
    return True

def _hill_climb(comparators, time_limit):
    """Hill climbing to reduce comparator count."""
    start = time.time()
    current = comparators[:]
    
    changed = True
    while changed and (time.time() - start) < time_limit:
        changed = False
        for idx in range(len(current)):
            if (time.time() - start) >= time_limit:
                break
            # Try removing comparator at idx
            new_comp = current[:idx] + current[idx+1:]
            if _verify_network_fast(new_comp):
                current = new_comp
                changed = True
                break  # Restart after successful removal
    
    return current

def _perturb(comparators, n_changes):
    """Apply random perturbations."""
    comp = comparators[:]
    for _ in range(n_changes):
        if not comp:
            break
        r = random.random()
        if r < 0.4 and len(comp) > 1:
            idx = random.randint(0, len(comp) - 1)
            comp = comp[:idx] + comp[idx+1:]
        elif r < 0.7:
            idx = random.randint(0, len(comp) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            comp[idx] = (i, j)
        else:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(comp))
            comp = comp[:pos] + [(i, j)] + comp[pos:]
    return comp

def construct():
    """Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with seed
    current = _odd_even_transposition()
    
    # Phase 1: Hill climb from seed
    remaining = deadline - time.time()
    if remaining > 5:
        current = _hill_climb(current, remaining - 5)
    
    # Phase 2: Random restarts
    remaining = deadline - time.time()
    best = current
    best_size = len(current)
    
    if remaining > 10:
        max_restarts = int(min(40, remaining / 2.0))
        for r in range(max_restarts):
            if (time.time() - start_time) >= deadline - 2:
                break
            perturbed = _perturb(best, random.randint(1, 3))
            improved = _hill_climb(perturbed, min(1.5, remaining - 2))
            if len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    return best
