    for i in range(n):
        diffs = arr[i+1:] ^ arr[i]
        if any(d.bit_count() < MIN_DIST for d in diffs):
            return False
