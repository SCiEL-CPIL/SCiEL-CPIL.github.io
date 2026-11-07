"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set of integers from 226 to 256 inclusive has size 31.
    # It satisfies the condition that subset sums of different cardinalities are disjoint.
    # This is verified by checking that max_sum(k) < min_sum(k+1) for all k.
    # For consecutive integers L..L+n-1, the condition holds if L > n^2/4.
    # For n=31, L=226 > 31^2/4 = 240.25 is FALSE, but the specific check passed.
    # Wait, 226 is NOT > 240.
    # My previous calculation for n=31:
    # Max(S_15) = 3735. Min(S_16) = 3736. Gap = 1.
    # The formula L > n^2/4 was an approximation.
    # The exact condition for consecutive integers is:
    # G_k = L - k(n-k-1) > 0.
    # Max k(n-k-1) for n=31 is at k=15: 15*(31-15-1) = 15*15 = 225.
    # So we need L > 225.
    # L=226 works.
    # For n=32, max k(n-k-1) at k=15: 15*(32-15-1) = 15*16 = 240.
    # Need L > 240.
    # Max L for n=32 is 256-31 = 225.
    # 225 is not > 240. So n=32 is impossible.
    
    return list(range(226, 257))
# EVOLVE-BLOCK-END
