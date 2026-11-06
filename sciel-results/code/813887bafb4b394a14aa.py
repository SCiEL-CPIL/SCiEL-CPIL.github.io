def construct():
    # The set [171, 256] has size 86.
    # For any a, b, c in this set with a < b <= c:
    # b + c > 2a.
    # The next multiple of a is 3a.
    # Since a >= 171, 3a >= 513.
    # Since b, c <= 256, b + c <= 512.
    # Thus b + c cannot be a multiple of a.
    return list(range(171, 257))
