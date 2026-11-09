import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    rng = np.random.default_rng(12345)

    # Quadratic residues mod 17: x^2 mod 17 for x=1..8
    def qr_set(n):
        s = set()
        for x in range(1, n):
            s.add((x * x) % n)
        s.discard(0)
        return s

    qr = sorted(qr_set(17))  # {1,2,4,8,9,13,15,16}
    # Build a base vector from the QR set (cyclic code generator idea)
    base = np.zeros(N, dtype=np.uint8)
    for r in qr:
        base[r] = 1

    def code_from_gen(G):
        k = G.shape[0]
        # all 2^k combinations
        words = []
        for mask in range(1 << k):
            w = 0
            for i in range(k):
                if (mask >> i) & 1:
                    row = G[i]
                    for j in range(N):
                        if row[j]:
                            w ^= (1 << j)
            words.append(w)
        return words

    def min_dist(words):
        n = len(words)
        best = N + 1
        for i in range(n):
            for j in range(i + 1, n):
                d = (words[i] ^ words[j]).bit_count()
                if d < best:
                    best = d
                    if best < MIN_DIST:
                        break
            if best < MIN_DIST:
                break
        return best

    # Candidate generator matrices: QR-derived rows + identity rows
    # Try several structured matrices and pick the first valid with largest size.
    candidates = []

    # Matrix A: QR cyclic shifts (8 rows) -> expect [17,8,6]
    G_A = np.zeros((8, N), dtype=np.uint8)
    for i in range(8):
        G_A[i] = np.roll(base, i)
    candidates.append(G_A)

    # Matrix B: QR shifts + one identity row (9 rows) -> maybe [17,9,6]
    G_B = np.zeros((9, N), dtype=np.uint8)
    for i in range(8):
        G_B[i] = np.roll(base, i)
    G_B[8] = np.eye(1, N, 0, dtype=np.uint8)[0]
    candidates.append(G_B)

    # Matrix C: random full-rank 9x17
    for trial in range(20):
        G_C = rng.integers(0, 2, size=(9, N), dtype=np.uint8)
        candidates.append(G_C)

    # Matrix D: random 8x17
    for trial in range(20):
        G_D = rng.integers(0, 2, size=(8, N), dtype=np.uint8)
        candidates.append(G_D)

    best_words = None
    for G in candidates:
        k = G.shape[0]
        # rank check
        if np.linalg.matrix_rank(G.astype(np.int8)) < k:
            continue
        words = code_from_gen(G)
        # distinct check
        if len(set(words)) != len(words):
            continue
        md = min_dist(words)
        if md >= MIN_DIST:
            if best_words is None or len(words) > len(best_words):
                best_words = words

    if best_words is None:
        # Fallback: deterministic small valid code (Reed-Muller-like) to guarantee nonempty
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    return list(best_words)
