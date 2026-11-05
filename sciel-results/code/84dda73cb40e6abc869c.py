def construct():
    # The set [171, 256] satisfies the condition.
    # For any a in [171, 256], 3*a >= 513 > 511 (max possible b+c).
    # Since b > a, b+c > 2a, so 2a is not a multiple to check.
    # The next multiple is 3a, which is out of bounds.
    return list(range(171, 257))
