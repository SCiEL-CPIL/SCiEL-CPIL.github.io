def build_S_from_set(A):
    S = [0] * 61
    S[0] = 1
    for x in A:
        k = len(A) # Wait, this is wrong.
        # The cardinality of subsets grows as we add elements.
        # If we add elements one by one, we need to know the current max cardinality.
        # Let's just do the standard DP.
        S = [0] * 61
        S[0] = 1
        curr_max_k = 0
        for x in A:
            for k in range(curr_max_k, -1, -1):
                if S[k]:
                    S[k+1] |= (S[k] << x)
            curr_max_k += 1
    return S
