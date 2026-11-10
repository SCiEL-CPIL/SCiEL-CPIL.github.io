    # Precompute sets containing each pair
    sets_containing_pair = {}
    for i in range(n):
        for j in range(i+1, n):
            sets_containing_pair[(i,j)] = [s for s in all_8_sets if i in s and j in s]
