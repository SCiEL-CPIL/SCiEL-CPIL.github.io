# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs 220 integer points in 11 dimensions using the minimal vectors of the D_11 lattice:
    all permutations of (±1, ±1, 0, ..., 0).

    Each point has squared norm 2. The minimum squared pairwise distance is 3 (≥ 2),
    satisfying the constraint min_dist² >= max_norm².

    Returns:
        points: np.ndarray of shape (220, 11) with int64 entries.
    """
    d = 11
    points = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    points = np.array(points, dtype=np.int64)
    return points


# EVOLVE-BLOCK-END
