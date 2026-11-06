def _bitonic_merge(n, lo, reverse=False):
    """Bitonic merge for n elements starting at lo."""
    if n <= 1:
        return []
    half = n // 2
    net = []
    # Compare pairs
    for i in range(half):
        if reverse:
            net.append((lo + i + half, lo + i))
        else:
            net.append((lo + i, lo + i + half))
    # Recursively merge halves
    net += _bitonic_merge(half, lo, reverse)
    net += _bitonic_merge(half, lo + half, reverse)
    return net

def _bitonic_sort(n, lo, reverse=False):
    if n <= 1:
        return []
    half = n // 2
    net = []
    net += _bitonic_sort(half, lo, False)
    net += _bitonic_sort(half, lo + half, True)
    net += _bitonic_merge(n, lo, reverse)
    return net
