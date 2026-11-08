<<<<<<< SEARCH
import time
import math
import random
import numpy as np

def _score_of(U):
    n = len(U)
    if n < 2:
        return 0.0
    a = np.asarray(U, dtype=np.int64)
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns = len(np.unique(s))
    nd = len(np.unique(d))
    if ns <= 0:
        return 0.0
    ratio = nd / ns
    if ratio <= 0:
        return 0.0
    maxU = a.max()
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _score_fast(U):
    """Vectorized score, handles small-to-moderate sets."""
    a = np.asarray(U, dtype=np.int64)
    n = a.size
    if n < 2:
        return 0.0
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns = s.size
    nd = d.size
    # unique counts via sorting (fast for small)
    ns_u = len(np.unique(s))
    nd_u = len(np.unique(d))
    if ns_u <= 0:
        return 0.0
    ratio = nd_u / ns_u
    if ratio <= 0:
        return 0.0
    maxU = int(a.max())
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom
=======
import time
import math
import random
import numpy as np

def _score_of(U):
    n = len(U)
    if n < 2:
        return 0.0
    a = np.asarray(U, dtype=np.int64)
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns = len(np.unique(s))
    nd = len(np.unique(d))
    if ns <= 0:
        return 0.0
    ratio = nd / ns
    if ratio <= 0:
        return 0.0
    maxU = a.max()
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _score_fast(U):
    """Vectorized score, handles small-to-moderate sets."""
    a = np.asarray(U, dtype=np.int64)
    n = a.size
    if n < 2:
        return 0.0
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns_u = len(np.unique(s))
    nd_u = len(np.unique(d))
    if ns_u <= 0:
        return 0.0
    ratio = nd_u / ns_u
    if ratio <= 0:
        return 0.0
    maxU = int(a.max())
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _score_fast_from_arrays(p, x):
    """Fast score when U = concat(prefix, [x]) with prefix already as array."""
    U = np.concatenate([p, np.asarray([x], dtype=np.int64)])
    s = U[:, None] + U[None, :]
    d = U[:, None] - U[None, :]
    ns_u = len(np.unique(s))
    nd_u = len(np.unique(d))
    if ns_u <= 0:
        return 0.0
    ratio = nd_u / ns_u
    if ratio <= 0:
        return 0.0
    maxU = int(U.max())
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom
>>>>>>> REPLACE

