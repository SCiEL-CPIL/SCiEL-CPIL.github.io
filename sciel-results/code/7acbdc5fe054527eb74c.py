"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from functools import reduce
import operator

N = 17

def _apply_comparator(state, comp):
    """Apply a single compare-exchange to a state (tuple of values)."""
    i, j = comp
    if state[i] > state[j]:
        lst = list(state)
        lst[i], lst[j] = state[j], state[i]
        return tuple(lst)
    return state

def _is_sorted(state):
    """Check if state is non-decreasing."""
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _validate_network(comparators, timeout_iters=131072):
    """Validate that the network sorts all 2^N boolean inputs (zero-one principle)."""
    for bits in range(timeout_iters):
        state = tuple((bits >> k) & 1 for k in range(N))
        for comp in comparators:
            state = _apply_comparator(state, comp)
        if not _is_sorted(state):
            return False
    return True

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (power of 2 case, or padded)."""
    # For 17 wires, we'll use a recursive bitonic construction
    # Bitonic sort works naturally for powers of 2; for 17 we can use 32-wire version
    # but that's wasteful. Instead, let's use a known construction.
    
    # Actually, let's use a simpler approach: odd-even merge sort network
    # For 17 wires, we can build it recursively.
    
    comparators = []
    
    def bitonic_merge(lo, hi, direction):
        """Merge two sorted sequences of equal length in wires [lo, hi)."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        # Compare elements at distance half apart
        for i in range(lo, lo + half):
            a = i
            b = i + half
            if direction:  # ascending
                if a < b:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
            else:  # descending
                if a < b:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
        # Recursively merge the two halves
        bitonic_merge(lo, lo + half, direction)
        bitonic_merge(lo + half, hi, direction)
    
    def bitonic_sort(lo, hi):
        """Sort wires [lo, hi) in ascending order using bitonic sort."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        # Sort first half ascending, second half descending to create bitonic sequence
        bitonic_sort(lo, lo + half)
        # For the second half, we need to sort descending
        # We can do this by sorting ascending then reversing the order of comparators
        # But simpler: just use odd-even merge or recursive bitonic with direction
        
        # Let's use a different approach: build bitonic sequence then merge
        # This is getting complex. Let me use a simpler seed.
        pass
    
    # Simpler: just use odd-even transposition as seed, then optimize
    return None

def _make_seed_network():
    """Create a seed sorting network using odd-even transposition."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _try_remove_comparator(comparators, idx):
    """Try removing comparator at index idx and return new list if still valid."""
    new_comps = comparators[:idx] + comparators[idx+1:]
    if _validate_network(new_comps):
        return new_comps
    return None

def _local_search(comparators, max_iter=5000, timeout_frac=0.7):
    """Local search to minimize comparator count."""
    import time
    start = time.time()
    timeout = 55  # seconds, leave buffer
    
    best = comparators[:]
    best_len = len(best)
    
    for iteration in range(max_iter):
        if time.time() - start > timeout:
            break
        if len(best) <= 63:  # Below lower bound, stop
            break
        
        # Try to remove a random comparator
        if len(best) == 0:
            break
        
        idx = random.randint(0, len(best) - 1)
        new_comps = _try_remove_comparator(best, idx)
        if new_comps is not None:
            best = new_comps
            best_len = len(best)
    
    return best

def _random_perturb(comparators, n_perturb=2):
    """Randomly perturb the network by adding/removing/swapping comparators."""
    comps = comparators[:]
    for _ in range(n_perturb):
        action = random.choice(['add', 'remove', 'swap'])
        if action == 'add' and random.random() < 0.3:
            i = random.randint(0, N - 2)
            j = i + random.randint(1, N - 1 - i)
            if i > j:
                i, j = j, i
            comps.append((i, j))
        elif action == 'remove' and len(comps) > 10:
            idx = random.randint(0, len(comps) - 1)
            comps.pop(idx)
        elif action == 'swap' and len(comps) >= 2:
            i = random.randint(0, len(comps) - 2)
            j = random.randint(i + 1, len(comps) - 1)
            comps[i], comps[j] = comps[j], comps[i]
    return comps

def construct():
    """Construct a sorting network on N=17 wires with minimized comparators."""
    import time
    start = time.time()
    timeout = 58  # seconds, leave buffer
    
    # Start with odd-even transposition seed
    seed = _make_seed_network()
    current = seed[:]
    
    # Validate seed
    if not _validate_network(current):
        # Should not happen, but just in case
        raise ValueError("Seed network is invalid")
    
    best = current[:]
    best_len = len(best)
    
    # Local search with random restarts
    while time.time() - start < timeout:
        # Try local search from current
        improved = _local_search(current, max_iter=200, timeout_frac=0.3)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)
            current = improved[:]
        else:
            # Perturb and try again
            if random.random() < 0.5:
                current = _random_perturb(best, n_perturb=random.randint(1, 3))
                if not _validate_network(current):
                    current = best[:]
            else:
                # Stuck, try a fresh perturbation
                current = _random_perturb(best, n_perturb=random.randint(2, 5))
                if not _validate_network(current):
                    # Try less aggressive perturbation
                    current = _random_perturb(best, n_perturb=1)
                    if not _validate_network(current):
                        current = best[:]
    
    return best

# EVOLVE-BLOCK-END
