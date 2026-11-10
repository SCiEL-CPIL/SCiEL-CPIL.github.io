def _enforce_symmetry_union(cell):
    sym = np.zeros_like(cell)
    N = cell.shape[0]
    for i in range(N):
        for j in range(N):
            if cell[i, j] == 1:
                sym[i, j] = 1
                sym[i, N-1-j] = 1
                sym[N-1-i, j] = 1
                sym[N-1-i, N-1-j] = 1
                sym[j, i] = 1
                sym[j, N-1-i] = 1
                sym[N-1-j, i] = 1
                sym[N-1-j, N-1-i] = 1
    return sym

def _enforce_symmetry_intersection(cell):
    sym = np.ones_like(cell)
    N = cell.shape[0]
    for i in range(N):
        for j in range(N):
            if cell[i, j] == 0:
                sym[i, j] = 0
                sym[i, N-1-j] = 0
                sym[N-1-i, j] = 0
                sym[N-1-i, N-1-j] = 0
                sym[j, i] = 0
                sym[j, N-1-i] = 0
                sym[N-1-j, i] = 0
                sym[N-1-j, N-1-i] = 0
    return sym
