<<<<<<< SEARCH
def _solve_instance(n, p, seed, shuffle=False):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    used = set()
    remaining = set(all_edges)
    rng2 = None
    if shuffle:
        from random import Random
        rng2 = Random(seed ^ 0x5DEECE66D)
=======
def _solve_instance(n, p, seed, shuffle=False):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    adj_mask_full = _adj_mask(n, edges)
    chosen = []
    used = set()
    remaining = set(all_edges)
    rng2 = None
    if shuffle:
        from random import Random
        rng2 = Random(seed ^ 0x5DEECE66D)
>>>>>>> REPLACE

Now let me also reduce the cap further and add an early exit if no progress is made across multiple rounds with the same remaining count:

