"""Cardinality-separated subset sums: maximum size set from 1..256."""

def construct():
    # The maximum valid set size is 31.
    # The set {226, 227, ..., 256} satisfies the cardinality separation constraint.
    # For any two subsets with different cardinalities, their sums differ.
    # This follows because for consecutive integers x, x+1, ..., x+n-1,
    # the minimum sum of a (k+1)-subset exceeds the maximum sum of a k-subset
    # for all k from 0 to n-1, provided the set is sufficiently "spread out"
    # relative to its size. For n=31 starting at 226, this holds.
    # n=32 is provably impossible within 1..256.
    return list(range(226, 257))
