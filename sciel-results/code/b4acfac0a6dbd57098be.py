import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072

def _build_input_matrix():
    """Build a (131072, 17) uint8 matrix of all Boolean inputs."""
    # Use bit extraction: input k has bit i set if (k >> i) & 1
    idx = np.arange(NUM_INPUTS, dtype=np.uint32)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = ((idx >> i) & 1).astype(np.uint8)
    return bits

def _verify_network(comparators, input_matrix):
    """Verify that the network sorts all inputs. Returns True if valid."""
    state = input_matrix.copy()
    for (i, j) in comparators:
        # min to i, max to j
        a = state[:, i]
        b = state[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        state[:, i] = mn
        state[:, j] = mx
    # Check sorted: for each row, state[:, 0] <= state[:, 1] <= ... <= state[:, N-1]
    # i.e., each row must be non-decreasing
    diffs = np.diff(state, axis=1)
    # All diffs must be >= 0
    return bool(np.all(diffs >= 0))

def _count_comparators(comparators):
    return len(comparators)

def _depth_of_network(comparators):
    """Compute parallel depth (longest path in comparator DAG)."""
    if not comparators:
        return 0
    # Simple approach: simulate layers
    # Each comparator (i,j) at position p depends on all comparators that touch i or j before it
    # Use topological depth
    last_end = [-1] * N
    depth = 0
    for (i, j) in comparators:
        d = max(last_end[i], last_end[j]) + 1
        last_end[i] = d
        last_end[j] = d
        depth = max(depth, d)
    return depth

def _odd_even_transposition():
    """Generate odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(comparators, input_matrix, idx):
    """Try removing comparator at index idx. Return new list if still valid."""
    new_comps = comparators[:idx] + comparators[idx+1:]
    if _verify_network(new_comps, input_matrix):
        return new_comps
    return None

def _try_insert(comparators, input_matrix, i, j):
    """Try inserting comparator (i,j) at every position. Return (new_list, pos) if valid."""
    for pos in range(len(comparators) + 1):
        new_comps = comparators[:pos] + [(i, j)] + comparators[pos:]
        if _verify_network(new_comps, input_matrix):
            return new_comps, pos
    return None, -1

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Build input matrix once
    input_matrix = _build_input_matrix()

    # Start from odd-even transposition
    comparators = _odd_even_transposition()
    best_count = len(comparators)
    best_comparators = list(comparators)

    # Phase 1: Greedy removal - try to remove each comparator
    # Go through multiple passes until no more can be removed
    pass_num = 0
    while time.time() - start_time < DEADLINE * 0.6:
        pass_num += 1
        removed_any = False
        # Try removing from end to beginning (less disruption)
        for idx in range(len(comparators) - 1, -1, -1):
            if time.time() - start_time > DEADLINE * 0.6:
                break
            new_comps = _try_remove(comparators, input_matrix, idx)
            if new_comps is not None:
                comparators = new_comps
                removed_any = True
                if len(comparators) < best_count:
                    best_count = len(comparators)
                    best_comparators = list(comparators)
        if not removed_any:
            break

    # Phase 2: Try to insert useful comparators to enable further removals
    # This is more complex; try adding a comparator then see if we can remove more
    if time.time() - start_time < DEADLINE * 0.75:
        # Try inserting each possible comparator at various positions
        # Then attempt removal passes again
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        # Shuffle for diversity
        import random
        random.seed(42)
        random.shuffle(all_pairs)
        
        improved = True
        while improved and time.time() - start_time < DEADLINE * 0.75:
            improved = False
            for (i, j) in all_pairs:
                if time.time() - start_time > DEADLINE * 0.75:
                    break
                # Skip if already present
                if (i, j) in comparators:
                    continue
                new_comps, pos = _try_insert(comparators, input_matrix, i, j)
                if new_comps is not None:
                    # Now try to remove comparators from the new network
                    temp_comps = new_comps
                    for _ in range(3):  # a few removal passes
                        any_removed = False
                        for idx in range(len(temp_comps) - 1, -1, -1):
                            if time.time() - start_time > DEADLINE * 0.75:
                                break
                            nc = _try_remove(temp_comps, input_matrix, idx)
                            if nc is not None:
                                temp_comps = nc
                                any_removed = True
                        if not any_removed:
                            break
                    if len(temp_comps) < len(comparators):
                        comparators = temp_comps
                        improved = True
                        if len(comparators) < best_count:
                            best_count = len(comparators)
                            best_comparators = list(comparators)

    # Final: make sure we have a valid network
    # Verify best
    if not _verify_network(best_comparators, input_matrix):
        # Fallback to original
        best_comparators = _odd_even_transposition()

    return best_comparators
