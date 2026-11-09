"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 17

# EVOLVE-BLOCK-START
def _verify_network(comparators):
    """Verify sorting network using zero-one principle with fast bitmask ops."""
    for x in range(1 << N):
        cur = x
        for (i, j) in comparators:
            bi = (cur >> i) & 1
            bj = (cur >> j) & 1
            if bi > bj:
                cur ^= (1 << i) | (1 << j)
        # Check sorted: 0s then 1s
        ones = 0
        tmp = cur
        while tmp:
            ones += 1
            tmp &= tmp - 1
        expected = ((1 << ones) - 1) << (N - ones)
        if cur != expected:
            return False
    return True


def _odd_even_transposition_network():
    """Generate odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def _try_remove_comparator(comparators, idx):
    """Try removing comparator at index idx and verify if still valid."""
    new_comps = comparators[:idx] + comparators[idx + 1:]
    if _verify_network(new_comps):
        return new_comps
    return None


def _greedy_reduce(comparators, deadline):
    """Greedy removal: repeatedly try to remove comparators."""
    comps = list(comparators)
    while time.time() < deadline:
        removed = False
        # Try from back to front (later comparators more likely redundant)
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() >= deadline:
                break
            new_comps = _try_remove_comparator(comps, idx)
            if new_comps is not None:
                comps = new_comps
                removed = True
                break
        if not removed:
            # Try from front
            for idx in range(len(comps)):
                if time.time() >= deadline:
                    break
                new_comps = _try_remove_comparator(comps, idx)
                if new_comps is not None:
                    comps = new_comps
                    removed = True
                    break
        if not removed:
            break
    return comps


def _can_swap(comparators, i, j):
    """Check if comparators at i and j can be swapped (disjoint wires)."""
    a = comparators[i]
    b = comparators[j]
    return len(set(a) & set(b)) == 0


def _local_search(comparators, deadline):
    """Local search: try removing and swapping."""
    comps = list(comparators)
    
    while time.time() < deadline:
        improved = False
        
        # Try removals
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() >= deadline:
                break
            new_comps = _try_remove_comparator(comps, idx)
            if new_comps is not None:
                comps = new_comps
                improved = True
                break
        if improved:
            continue
        
        # Try swaps of disjoint comparators
        found_swap = False
        for idx in range(len(comps) - 1):
            if time.time() >= deadline:
                break
            for jdx in range(idx + 1, min(idx + 3, len(comps))):
                if time.time() >= deadline:
                    break
                if _can_swap(comps, idx, jdx):
                    new_comps = list(comps)
                    new_comps[idx], new_comps[jdx] = new_comps[jdx], new_comps[idx]
                    if _verify_network(new_comps):
                        comps = new_comps
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                break
        if improved:
            continue
        
        break
    
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s margin
    
    # Start with odd-even transposition network
    comps = _odd_even_transposition_network()
    
    # Phase 1: Greedy removal
    comps = _greedy_reduce(comps, deadline)
    
    # Phase 2: Local search
    if time.time() < deadline:
        comps = _local_search(comps, deadline)
    
    # Final verification
    if not _verify_network(comps):
        comps = _odd_even_transposition_network()
    
    return comps
# EVOLVE-BLOCK-END
