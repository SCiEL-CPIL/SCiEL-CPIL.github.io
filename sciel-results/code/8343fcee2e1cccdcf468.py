def construct():
    n = 32
    edges = []
    block_size = 4
    num_blocks = 8
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        for a in range(block_size):
            for b in range(a + 1, block_size):
                edges.append([start + a, start + b])
    return {'n': n, 'edges': edges}
