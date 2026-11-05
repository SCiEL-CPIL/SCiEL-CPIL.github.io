import time

def construct():
    # The set {66, 67, ..., 100} has cardinality 35.
    # The minimum sum of any 3 distinct elements is 66 + 67 + 68 = 201 > 200.
    # The maximum sum of any 2 distinct elements is 99 + 100 = 199 < 200.
    # No single element is 200.
    # Thus, no subset sums to 200.
    # It is conjectured/known that 35 is the maximum cardinality for this problem.
    return list(range(66, 101))
