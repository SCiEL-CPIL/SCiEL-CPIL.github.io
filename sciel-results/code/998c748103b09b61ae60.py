def _bitonic(n, direction):
    """
    Generate a bitonic sort network for n wires.
    n must be a power of 2.
    direction: True for ascending, False for descending.
    """
    if n <= 1:
        return []
    half = n // 2
    # Sort first half in ascending order
    # Sort second half in descending order
    left = _bitonic(half, True)
    right = _bitonic(half, False)
    # Offset the right half
    right = [(i + half, j + half) for (i, j) in right]
    # Merge the two halves
    merge = _bitonic_merge(n, direction)
    return left + right + merge

def _bitonic_merge(n, direction):
    """
    Generate a bitonic merge network for n wires.
    Assumes the input is bitonic (first half ascending, second half descending).
    n must be a power of 2.
    """
    if n <= 1:
        return []
    half = n // 2
    # Compare corresponding elements
    comparators = []
    for i in range(half):
        if direction:
            comparators.append((i, i + half))
        else:
            comparators.append((i + half, i))
    # Recursively merge each half
    left = _bitonic_merge(half, direction)
    right = _bitonic_merge(half, direction)
    right = [(i + half, j + half) for (i, j) in right]
    return comparators + left + right
