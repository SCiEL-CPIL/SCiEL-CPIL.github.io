def construct():
    n = 12
    # The minimum possible largest element for n distinct positive integers
    # with distinct subset sums is 2^(n-1), achieved by powers of 2.
    # Proof: Let a_1 < ... < a_n. Distinct subset sums imply a_k > sum(a_1...a_{k-1}).
    # This forces a_k >= 2^(k-1). Thus max >= 2^(n-1).
    return [1 << i for i in range(n)]
