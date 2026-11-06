"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n = 20
    left = list(range(10))
    right = list(range(10, 20))
    m = 10
    K = 20  # number of available colours

    def build_edges(C):
        # C is 10x10 int matrix of colours 0..K-1
        edges = []
        for i in range(m):
            for j in range(m):
                u = left[i]
                v = right[j]
                c = int(C[i, j])
                edges.append([u, v, c])
        return edges

    def is_latin_valid(C):
        # rainbow 4-cycles iff each colour has <=1 per row and <=1 per column
        # (i.e. the matrix is a Latin square over its used colours)
        for c in range(K):
            mask = (C == c)
            if mask.any():
                row_counts = mask.sum(axis=1)
                if row_counts.max() > 1:
                    return False
                col_counts = mask.sum(axis=0)
                if col_counts.max() > 1:
                    return False
        return True

    # Fallback: star (19 edges)
    star = [[0, v, 0] for v in range(1, n)]
    best_edges = star
    best_count = len(star)

    def try_update(C):
        nonlocal best_edges, best_count
        if is_latin_valid(C):
            e = build_edges(C)
            if len(e) > best_count:
                best_count = len(e)
                best_edges = e
            return True
        return False

    # --- Approach A: pure Latin square with 10 colours (guaranteed valid, 100 edges) ---
    # Base Latin square: C[i,j] = (i + j) % m, then map to distinct colours.
    base = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            base[i, j] = (i + j) % m
    # map base symbols 0..9 to 10 of the 20 colours
    perm = list(range(K))
    random.shuffle(perm)
    C0 = np.array([[perm[int(base[i, j])] for j in range(m)] for i in range(m)], dtype=np.int64)
    try_update(C0)

    # --- Approach B: Latin square that uses up to K distinct colours ---
    # Start from base Latin square (10 symbols), then recolor individual cells to
    # new distinct colours while preserving the Latin property (one per row/col).
    # Strategy: build a random Latin square over K symbols restricted to 10x10? 
    # Simpler: start from C0 and try to introduce extra colours by swapping.

    # Better constructive approach: random Latin square of order 10 using 10 colours,
    # then for extra colours, pick a cell and recolor it to a fresh colour c that
    # is not already in its row or column (guarantees Latin property holds).
    def try_inject(C, max_inject=300):
        # C is 10x10 Latin over some colours; inject extra distinct colours
        # by recoloring a cell to a colour absent from its row and column.
        C = C.copy()
        for _ in range(max_inject):
            if time.time() > deadline:
                break
            # pick random cell
            i = random.randrange(m)
            j = random.randrange(m)
            row_cols = set(C[i, :].tolist())
            col_cols = set(C[:, j].tolist())
            used = row_cols | col_cols
            available = [c for c in range(K) if c not in used]
            if not available:
                continue
            # recolor to a fresh colour (not already used anywhere? not required)
            newc = random.choice(available)
            C[i, j] = newc
        return C

    # Try several random Latin squares of order 10 (10 colours), then inject.
    for attempt in range(4000):
        if time.time() > deadline:
            break
        # random Latin square via random permutation of columns of base
        C = np.zeros((m, m), dtype=np.int64)
        col_perm = list(range(m))
        random.shuffle(col_perm)
        for i in range(m):
            for j in range(m):
                C[i, j] = (i + col_perm[j]) % m
        # map to 10 random distinct colours
        perm = list(range(K))
        random.shuffle(perm)
        C = np.array([[perm[int(C[i, j])] for j in range(m)] for i in range(m)], dtype=np.int64)
        # inject extra colours
        C2 = try_inject(C, max_inject=120)
        try_update(C2)

    # --- Approach C: full random Latin square over K colours of order 10 ---
    # A Latin square of order 10 uses exactly 10 symbols. To use up to 20 colours
    # while staying valid, we only need each colour used at most once per row/col.
    # We can take a 10x10 array and assign colours such that rows and columns are
    # injective. Random greedy assignment:
    for attempt in range(3000):
        if time.time() > deadline:
            break
        C = np.full((m, m), -1, dtype=np.int64)
        ok = True
        # order cells randomly
        cells = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(cells)
        for (i, j) in cells:
            used = set(C[i, :].tolist()) | set(C[:, j].tolist())
            used = {c for c in used if c != -1}
            avail = [c for c in range(K) if c not in used]
            if not avail:
                ok = False
                break
            C[i, j] = random.choice(avail)
        if ok:
            try_update(C)

    return best_edges
# EVOLVE-BLOCK-END
