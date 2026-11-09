<<<<<<< SEARCH
def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
=======
def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _get_active_pairs(c, K):
    """Get the K closest pairs using KD-tree."""
    tree = cKDTree(c)
    # Query k=K+1 neighbors (including self)
    dists, idxs = tree.query(c, k=min(K + 1, N))
    # Collect all unique pairs
    pairs = set()
    for i in range(N):
        for j in idxs[i, 1:]:  # skip self
            if j > i:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    # Sort by distance and take top K
    pair_list = list(pairs)
    if len(pair_list) <= K:
        return pair_list
    # Compute distances for sorting
    pair_dists = [(np.linalg.norm(c[i] - c[j]), (i, j)) for i, j in pair_list]
    pair_dists.sort()
    return [p for _, p in pair_dists[:K]]
>>>>>>> REPLACE

