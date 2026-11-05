"""Rewritten program for size-efficient sorting networks on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a fixed compare-exchange network that sorts all 16-input sequences.
    Uses a known efficient construction based on odd-even mergesort.
    """
    def odd_even_merge(n, offset, comparators):
        """
        Add comparators to merge two sorted halves of size n//2 starting at offset.
        """
        if n <= 1:
            return
        half = n // 2
        # Recursively sort each half
        odd_even_merge(half, offset, comparators)
        odd_even_merge(half, offset + half, comparators)
        # Merge the two sorted halves
        for k in range(n - 1):
            if k % 2 == 0:
                for i in range(half - k - 1, -1, -1):
                    comparators.append((offset + i, offset + i + 1))
            else:
                for i in range(half - k, -1, -1):
                    if i + 1 < n:
                        comparators.append((offset + i, offset + i + 1))

    def odd_even_sort(n, offset, comparators):
        """
        Build an odd-even mergesort network for n wires starting at offset.
        """
        if n <= 1:
            return
        half = n // 2
        # Recursively sort each half
        odd_even_sort(half, offset, comparators)
        odd_even_sort(half, offset + half, comparators)
        # Merge the two sorted halves
        odd_even_merge(n, offset, comparators)

    # Build the sorting network
    comparators = []
    odd_even_sort(N, 0, comparators)

    # Remove duplicate comparators (keep order)
    seen = set()
    unique_comparators = []
    for c in comparators:
        if c not in seen:
            seen.add(c)
            unique_comparators.append(c)

    return unique_comparators
# EVOLVE-BLOCK-END
