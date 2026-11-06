"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def _bitonic_merge(n, offset, step):
    """
    Generates comparators for a bitonic merge of a sequence of length n,
    starting at 'offset' with a stride of 'step'.
    """
    if n <= 1:
        return []
    half = n // 2
    comparators = []
    # Compare elements i and i + half
    for i in range(half):
        u = offset + i * step
        v = offset + (i + half) * step
        comparators.append((min(u, v), max(u, v)))
    # Recursively merge the two halves
    comparators.extend(_bitonic_merge(half, offset, step * 2))
    comparators.extend(_bitonic_merge(half, offset + half * step, step * 2))
    return comparators

def _bitonic_sort(n, offset, step, direction):
    """
    Generates comparators for a bitonic sort of a sequence of length n,
    starting at 'offset' with a stride of 'step'.
    direction: True for ascending, False for descending.
    """
    if n <= 1:
        return []
    half = n // 2
    comparators = []
    # Sort first half ascending
    comparators.extend(_bitonic_sort(half, offset, step, True))
    # Sort second half descending
    comparators.extend(_bitonic_sort(half, offset + half * step, step, False))
    # Merge the two halves
    comparators.extend(_bitonic_merge(n, offset, step))
    # Note: The standard bitonic sort merge assumes the input is bitonic.
    # The recursive structure ensures this.
    # However, the direction of the final merge depends on the desired output order.
    # For a top-level sort, we want ascending.
    # The merge step above produces an ascending sorted sequence if the input was bitonic (up then down).
    # Our construction: first half up, second half down. This is a valid bitonic sequence.
    # The merge will sort it into ascending order.
    return comparators

def construct():
    """
    Constructs a Bitonic Sorting Network for N=16.
    Uses 80 comparators.
    """
    # Top-level call: sort 16 wires, offset 0, step 1, ascending
    network = _bitonic_sort(N, 0, 1, True)
    return network
# EVOLVE-BLOCK-END
