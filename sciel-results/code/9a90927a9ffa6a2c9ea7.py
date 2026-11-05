"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Batcher's odd-even merge sort network for 16 wires.
    Yields 60 comparators.
    """
    comps = []
    
    def batcher_sort(start, end):
        size = end - start
        if size <= 1:
            return
        half = size // 2
        batcher_sort(start, start + half)
        batcher_sort(start + half, end)
        # Odd-even merge
        _odd_even_merge(start, start + half, start + half, end)
    
    def _odd_even_merge(a_start, a_end, b_start, b_end):
        size_a = a_end - a_start
        size_b = b_end - b_start
        total = size_a + size_b
        if total == 2:
            comps.append((a_start, b_start))
            return
        # Split each half
        ha = total // 2
        # Merge (a_start..a_start+ha) with (b_start..b_start+ha)
        _odd_even_merge(a_start, a_start + ha, b_start, b_start + ha)
        # Merge (a_start+ha..a_end) with (b_start+ha..b_end)
        _odd_even_merge(a_start + ha, a_end, b_start + ha, b_end)
        # Odd-even cross phase
        # Compare (a_start + i) with (b_start + i) for i = 1, 3, 5, ...
        i = 1
        while i < total - 1:
            comps.append((a_start + i, b_start + i))
            i += 2
        # Compare (a_start + i) with (b_start + i) for i = 2, 4, 6, ...
        i = 2
        while i < total - 1:
            comps.append((a_start + i, b_start + i))
            i += 2
    
    batcher_sort(0, N)
    return comps
# EVOLVE-BLOCK-END
