    # claim the core edges within this neighborhood
    for a in range(len(nbrs)):
        for b in range(a + 1, len(nbrs)):
            claimed.add(frozenset((nbrs[a], nbrs[b])))
    