def batcher(n, offset=0):
    if n == 1:
        return []
    half = n // 2
    wires = list(range(offset, offset+n))
    comp = []
    comp += batcher(half, offset)
    comp += batcher(half, offset+half)
    comp += merge(wires)
    return comp
