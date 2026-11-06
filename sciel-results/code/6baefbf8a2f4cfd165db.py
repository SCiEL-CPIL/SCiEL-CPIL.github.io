import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = 22
    # Denominators
    denominators = list(range(1, n + 1))
    # Reciprocals for calculation
    reciprocals = [1.0 / i for i in denominators]

    # Current set of denominators to keep
    current_denoms = denominators[:]

    def find_dependence(denoms):
        """
        Find a pair of subsets of denoms with the same reciprocal sum.
        Returns the relative index of an element to remove, or None if valid.
        """
        # Generate all subset sums and their masks
        # mask is relative to the input list 'denoms'
        items = [(0.0, 0)]
        for i, d in enumerate(denoms):
            val = 1.0 / d
            bit = 1 << i
            new_items = []
            for s, m in items:
                new_items.append((s + val, m | bit))
            items.extend(new_items)

        # Sort by sum
        items.sort(key=lambda x: x[0])

        # Find adjacent equal sums
        for i in range(len(items) - 1):
            s1, m1 = items[i]
            s2, m2 = items[i + 1]
            if abs(s1 - s2) < 1e-12:
                # Found dependence
                # Return the relative index of one element in the symmetric difference
                diff_mask = m1 ^ m2
                # Get lowest set bit
                if diff_mask:
                    rel_idx = (diff_mask & -diff_mask).bit_length() - 1
                    return rel_idx
                else:
                    # m1 == m2, which means same subset, not a dependence
                    # Should not happen with distinct sums unless sum is 0?
                    # But we are looking for distinct subset sums.
                    # If m1 == m2, they are the same subset, so sum is same.
                    # This is not a collision of DIFFERENT subsets.
                    # We need m1 != m2.
                    continue

        return None

    while True:
        if time.time() > deadline:
            break

        rel_idx_to_remove = find_dependence(current_denoms)

        if rel_idx_to_remove is None:
            # Valid set found
            return current_denoms

        # Remove the element
        current_denoms.pop(rel_idx_to_remove)

    # Fallback
    # Check if current set is valid
    if find_dependence(current_denoms) is None:
        return current_denoms

    # Return seed
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
