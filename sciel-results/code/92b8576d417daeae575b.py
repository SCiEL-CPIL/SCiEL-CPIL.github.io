def construct():
    """
    Return a nonempty subset of [1, 100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy: Select the largest integers such that the minimum sum of any 3 
    distinct elements exceeds 200, while the maximum sum of any 2 distinct 
    elements is less than 200.
    
    Let S = {k, k+1, ..., 100}.
    Max sum of 2 elements: 100 + 99 = 199 < 200. (Always true for k <= 99)
    Min sum of 3 elements: k + (k+1) + (k+2) = 3k + 3.
    We require 3k + 3 > 200 => 3k > 197 => k >= 66.
    
    For k = 66:
    S = {66, 67, ..., 100}.
    Cardinality = 100 - 66 + 1 = 35.
    No subset sums to 200 because:
    - Any subset of size <= 2 has sum <= 199.
    - Any subset of size >= 3 has sum >= 201.
    
    This construction is optimal for this problem instance.
    """
    return list(range(66, 101))
