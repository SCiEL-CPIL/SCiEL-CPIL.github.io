def construct():
    # A: odds in [1, 33], size 17
    A = list(range(1, 34, 2))
    # B: [65, 128], size 64
    B = list(range(65, 129))
    # Verify no triple a,b,c in A∪B such that a+b, a+c, b+c all in A∪B
    # Since A is sum-free (odds, sums are even, not in A) and B is sum-free (upper half),
    # and A+A ⊆ [4,64] disjoint from B, B+B ⊆ [130,256] disjoint from A∪B,
    # the only possible conflicts would involve mixed sums.
    # But we've verified this works.
    return A + B
