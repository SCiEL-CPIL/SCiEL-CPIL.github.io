    # Precompute block_triplet_matrix and block_to_idx
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    block_triplet_matrix = np.zeros((n_blocks, n_triplets), dtype=bool)
    block_triplet_ids = [None] * n_blocks
    
    for i, block in enumerate(all_blocks):
        tids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            block_triplet_matrix[i, tid] = True
            tids.append(tid)
        block_triplet_ids[i] = tids
