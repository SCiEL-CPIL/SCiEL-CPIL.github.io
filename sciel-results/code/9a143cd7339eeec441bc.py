def bit_count(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')
