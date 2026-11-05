def construct():
    # 8 K4 blocks + 3 extra vertices connected to full K4 blocks 0,1,2
    # Forms: K5 (block0+v32), K5 (block1+v33), K5 (block2+v34), 5xK4
    # n=35, alpha=8, C5-free
    edges = []
    # K4 blocks: block b has vertices 4*b .. 4*b+3
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    # Extra vertex 32 connected to all of block 0 (0-3)
    for v in range(4):
        edges.append([32, v])
    # Extra vertex 33 connected to all of block 1 (4-7)
    for v in range(4, 8):
        edges.append([33, v])
    # Extra vertex 34 connected to all of block 2 (8-11)
    for v in range(8, 12):
        edges.append([34, v])
    return {'n': 35, 'edges': edges}
