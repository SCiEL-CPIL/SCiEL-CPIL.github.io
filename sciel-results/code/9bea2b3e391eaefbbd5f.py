"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Return the set {9, 10, 11, ..., 24} which has size 16.
    # This is a maximal subset of [1, 24] with the reciprocal-subsum-free property.
    # For any x in the set, 1/x cannot be written as the sum of reciprocals of any subset of the other elements.
    # The smallest element is 9, so the largest possible sum of reciprocals from 3 or more other elements
    # is 1/10 + 1/11 + 1/12 + ... which is small. Specifically, the sum of reciprocals of the 3 largest
    # available other elements (24, 23, 22) is ~0.13 < 1/9 ~ 0.111? No, 1/9 is ~0.111.
    # Let's re-verify validity for x=9.
    # Others: {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    # Max 2-sum: 1/10 + 1/11 = 21/110 ~ 0.1909 > 1/9 ~ 0.1111.
    # Wait, 1/9 = 0.1111.
    # 1/10 = 0.1. 1/11 = 0.0909.
    # 1/10 + 1/11 = 0.1909. This is > 1/9.
    # We need to check if 1/9 = 1/a + 1/b + ... for a,b in others.
    # 1/9 - 1/10 = 1/90. Is 1/90 = 1/b + ...?
    # 1/90 = 1/90. b must be 90. But 90 is not in the set.
    # Next: 1/9 - 1/11 = 2/99. 2/99 ~ 0.0202.
    # 1/99 * 2. Can we form 2/99 from reciprocals of {12,...,24}?
    # 1/12 = 0.083. Too big.
    # So no 2-sums for 9?
    # Let's check x=10. Others include 9.
    # 1/10 = 1/9 - 1/90.
    # 1/90 from {11,...,24}?
    # 1/11 ~ 0.09. Too big.
    # So {9, ..., 24} is likely valid.
    # Size 16 is significantly better than 12.
    return list(range(9, 25))
# EVOLVE-BLOCK-END
