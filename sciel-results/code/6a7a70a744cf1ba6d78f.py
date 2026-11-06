"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17.
    # All elements must share gcd>1 with 255, so each must be divisible by 3, 5, or 17.
    # Multiples of 3: 85 elements. Multiples of 5: 51. Multiples of 17: 15.
    # Any pairwise-gcd>1 set is a clique in the graph on {i: 3|i or 5|i or 17|i}.
    # The multiples-of-3 set is an 85-clique. Can we do better by mixing?
    # Numbers divisible by both 3 and 5 (i.e., 15) bridge the 3-clique and 5-clique.
    # But any two numbers one divisible by 3 only and one by 5 only need not have gcd>1.
    # The maximum clique is exactly the multiples of 3, giving 85 elements.
    # Proof sketch: any valid set S with 255 must have each element share a factor with
    # every other. If S contains a number x not divisible by 3, then x must be divisible
    # by 5 or 17. If x is divisible by 5 but not 3, then any y in S not divisible by 5
    # must share gcd>1 with x, forcing y to share factor 17 with x. This becomes very
    # restrictive. The clean 3-clique achieves 85 which is the floor(255/3) bound.
    return [3 * k for k in range(1, 86)]
# EVOLVE-BLOCK-END
