def compute_block_rank(ni, mj, pk):
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk
