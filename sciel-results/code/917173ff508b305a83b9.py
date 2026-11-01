"""Construct a 4-regular graph of girth 10 with 320 vertices (the (4,10)-cage)."""
# EVOLVE-BLOCK-START
def construct(k, g):
    n_layers = 4
    per_layer = 80
    n = n_layers * per_layer
    offsets = [1, 2, 4, 8]
    
    edges = []
    for layer in range(n_layers):
        next_layer = (layer + 1) % n_layers
        for j in range(per_layer):
            v = layer * per_layer + j
            for off in offsets:
                u = next_layer * per_layer + ((j + off) % per_layer)
                edges.append((v, u))
    
    return n, edges
# EVOLVE-BLOCK-END
