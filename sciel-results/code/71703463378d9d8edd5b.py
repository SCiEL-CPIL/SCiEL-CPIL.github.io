<<<<<<< SEARCH
def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol

    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1

    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True

    return sol
=======
def _remove_redundant(sol, block_triple_matrix, num_triples):
    if not sol:
        return sol

    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1

    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True

    return sol
>>>>>>> REPLACE

