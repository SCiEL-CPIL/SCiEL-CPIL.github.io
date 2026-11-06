from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    # 255 = 3 * 5 * 17
    # Any element in the set must share a factor with 255.
    # The multiples of 3 in [1, 255] form a clique of size 85.
    # 85 is the maximum possible size.
    return list(range(3, 256, 3))

# Run and verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
    print("Valid:", valid)
