<<<<<<< SEARCH
def _polish(sol, num_blocks, block_triple_arrays, triple_to_blocks_arr, num_triples):
    if not sol:
        return sol
    sol_set = set(sol)
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_arrays[idx]
        if len(tris) > 0:
            counts[tris] += 1

    # Iterative redundant removal to fixpoint
    # A block is redundant if it has no unique triples
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_arrays[idx]
            if len(tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.all(counts[tris] > 1):
                counts[tris] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
=======
def _polish(sol, num_blocks, block_triple_arrays, triple_to_blocks_arr, num_triples, block_triple_lens=None):
    if not sol:
        return sol
    sol_set = set(sol)
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_arrays[idx]
        if len(tris) > 0:
            counts[tris] += 1

    # Iterative redundant removal to fixpoint
    # A block is redundant if it has no unique triples (no triple with count==1)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_arrays[idx]
            if len(tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if not np.any(counts[tris] == 1):
                counts[tris] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
>>>>>>> REPLACE

