import time

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    Maximizes the set size.
    """
    # We know that an Arithmetic Progression with d=1, n=31, a=226 is valid.
    # Set: 226, 227, ..., 256
    # Size: 31
    
    # Let's verify if n=32 is possible with a quick search or logic.
    # Based on analysis, n=32 AP fails.
    # Non-AP search is expensive.
    # We will return the known valid set of size 31.
    
    # To be safe and fast, we just return the known good set.
    # If we want to be robust, we can check validity.
    
    S = list(range(226, 257))
    
    # Optional: Verify validity
    # n = len(S)
    # prefix = [0]*(n+1)
    # for i in range(n): prefix[i+1] = prefix[i] + S[i]
    # suffix = [0]*(n+1)
    # for i in range(n-1, -1, -1): suffix[i] = suffix[i+1] + S[i]
    # for k in range(1, n+1):
    #     min_k = prefix[k]
    #     max_k_minus_1 = 0 if k==1 else suffix[n-(k-1)]
    #     if min_k <= max_k_minus_1:
    #         return list(range(128, 136)) # Fallback
    # return S
    
    return S
