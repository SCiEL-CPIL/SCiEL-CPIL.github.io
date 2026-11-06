N = 18

def _odd_even_merge(a, b, size):
    """
    Generate comparators to merge two sorted sequences.
    a and b are lists of wire indices (each of length `size`).
    The merged result is placed back into the same 2*size wires.
    Returns list of (i, j) comparators.
    """
    if size == 1:
        return [(a[0], b[0])]

    half = (size + 1) // 2
    # Split each sequence into even and odd indexed elements
    a_even = a[0::2]
    a_odd = a[1::2]
    b_even = b[0::2]
    b_odd = b[1::2]

    comparators = []
    # Recursively merge even-indexed elements
    comparators.extend(_odd_even_merge(a_even, b_even, len(a_even)))
    # Recursively merge odd-indexed elements
    comparators.extend(_odd_even_merge(a_odd, b_odd, len(a_odd)))

    # Now we need to interleave and fix adjacent pairs
    # After merging, the even-merged sequence occupies wires:
    #   a_even[0], b_even[0], a_even[1], b_even[1], ...
    # No, that's not right. The merged even sequence is distributed across
    # the wires in a_even + b_even. We need to track where each element goes.

    # In Batcher's odd-even merge, after the recursive merges:
    # - The merged even sequence has 2*len(a_even) elements
    # - The merged odd sequence has 2*len(a_odd) elements
    # We interleave them: E[0], O[0], E[1], O[1], ...
    # and then compare adjacent pairs to fix up.

    # The key insight: after merging a_even and b_even, the k-th smallest
    # element among all even-indexed elements is in the k-th position of
    # the combined wire set (a_even interleaved with b_even in a specific way).

    # For the final fix-up step, we compare:
    # (wire holding element 2i, wire holding element 2i+1) for all valid i
    # where element 2i is the i-th even-merged element and element 2i+1 is
    # the i-th odd-merged element.

    # Let's track the wire positions:
    # After _odd_even_merge(a_even, b_even), the result is a sorted sequence
    # of length 2*len(a_even) distributed across wires a_even + b_even.
    # The i-th element of this sorted sequence is in wire (a_even + b_even)[i].

    # Similarly for odd.

    # So for the final comparisons:
    # For i in range(max(len(a_even), len(a_odd))):
    #   if i < 2*len(a_even) and i < 2*len(a_odd):
    #     even_pos = i // 2  # index in the even-merged sequence
    #     odd_pos = i // 2
    #     # But this isn't quite right either.

    # Let me use a simpler approach: just compare all adjacent pairs
    # in the final interleaved sequence.

    # The final sequence should have 2*size elements.
    # After the recursive merges, we have:
    # - Even-merged: 2*len(a_even) sorted elements in wires a_even + b_even
    # - Odd-merged: 2*len(a_odd) sorted elements in wires a_odd + b_odd

    # We interleave: take element 0 from even-merged, element 0 from odd-merged,
    # element 1 from even-merged, element 1 from odd-merged, etc.

    # The wire for the k-th element of the even-merged sequence is:
    #   if k < len(a_even): a_even[k]
    #   else: b_even[k - len(a_even)]
    # No, that's not how it works. The merged sequence is distributed across
    # ALL wires in a_even + b_even.

    # I think the cleanest way is to note that in Batcher's merge:
    # The output wire for the k-th smallest element of the combined set
    # (a_even ∪ b_even) is the k-th wire in the list (a_even + b_even).

    # So:
    # even_merged_wires = a_even + b_even  # length 2*len(a_even)
    # odd_merged_wires = a_odd + b_odd    # length 2*len(a_odd)

    # The final output should be:
    # E[0], O[0], E[1], O[1], E[2], O[2], ...
    # where E[i] is the i-th element of the even-merged sequence
    # and O[i] is the i-th element of the odd-merged sequence.

    # So:
    # E[i] is in wire even_merged_wires[i]
    # O[i] is in wire odd_merged_wires[i]

    # We need to compare (E[i], O[i]) for all valid i, i.e.,
    # for i in range(min(len(even_merged_wires), len(odd_merged_wires))):
    #   compare (even_merged_wires[i], odd_merged_wires[i])

    # But wait, we also need to handle the case where one sequence is longer.
    # If even_merged_wires is longer, the extra elements are already in the
    # correct position (they're the largest elements and go at the end).
    # Similarly if odd_merged_wires is longer.

    even_merged_wires = a_even + b_even
    odd_merged_wires = a_odd + b_odd

    max_len = max(len(even_merged_wires), len(odd_merged_wires))
    for i in range(max_len):
        if i < len(even_merged_wires) and i < len(odd_merged_wires):
            comparators.append((even_merged_wires[i], odd_merged_wires[i]))

    return comparators

def _batcher_sort(n):
    """
    Generate Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    mid = n // 2
    left_size = mid
    right_size = n - mid

    left_comps = _batcher_sort(left_size)
    right_comps = _batcher_sort(right_size)
    right_comps = [(i + left_size, j + left_size) for (i, j) in right_comps]

    left_wires = list(range(left_size))
    right_wires = list(range(left_size, n))

    # Pad if sizes differ
    if len(left_wires) < len(right_wires):
        left_wires = left_wires + [left_size + len(left_wires)] * (len(right_wires) - len(left_wires))
    elif len(right_wires) < len(left_wires):
        right_wires = right_wires + [right_size + len(right_wires)] * (len(left_wires) - len(right_wires))

    merge_comps = _odd_even_merge(left_wires, right_wires, len(left_wires))

    return left_comps + right_comps + merge_comps

def construct():
    """
    Construct a sorting network for 18 wires using Batcher's odd-even merge sort.
    """
    return _batcher_sort(18)
