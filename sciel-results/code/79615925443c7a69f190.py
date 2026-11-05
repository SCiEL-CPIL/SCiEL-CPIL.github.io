def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')
