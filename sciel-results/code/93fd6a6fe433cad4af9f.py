"""Maximum Cut: vectorized greedy local search with random restarts and a time guard."""
import time
import numpy as np

_DEADLINE_SECONDS = 120.0  # safety margin under the 150 s wall-clock kill
_SAFETY_MARGIN = 5.0       # stop this many seconds before the deadline


def _cut_size(A, s):
    # s in {+1,-1}; cut = 0.5 * sum A[i,j]*(1 - s[i]*s[j])
    return 0.5 * (A.sum() - s @ (A @ s))


def _local_search(A, s, deadline):
    """Greedy flip-improvement. Returns best (cut, s) seen during this run."""
    n = A.shape[0]
    cur_cut = _cut_size(A, s)
    best_cut = cur_cut
    best_s = s.copy()
    As = A @ s  # A @ s, shape (n,)
    # gain of flipping v: sum_j A[v,j]*(2*s[v]*s[j]-1) = 2*s[v]*As[v] - 2*deg[v]
    deg = A.sum(axis=1)
    while True:
        if time.time() >= deadline:
            break
        gain = 2.0 * s * As - 2.0 * deg
        # only flips with strictly positive gain improve the cut
        idx = np.argmax(gain)
        g = gain[idx]
        if g <= 0:
            break
        v = int(idx)
        # flip v
        s[v] = -s[v]
        cur_cut += 0.5 * g
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_s = s.copy()
        # update As: As[j] += A[j,v] * (s_new[v] - s_old[v]) = A[j,v]*(-2*s_old[v])
        delta = -2.0 * (-s[v])  # s_old = -s[v] after flip; new - old = 2*new = -2*old
        # simpler: new_s = s[v]; old_s = -s[v]; new - old = 2*new_s = 2*s[v]
        delta = 2.0 * s[v]
        As += A[:, v] * delta
    return best_cut, best_s


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    start = time.time()
    deadline = start + _DEADLINE_SECONDS - _SAFETY_MARGIN

    rng = np.random.default_rng(0)
    best_cut = -np.inf
    best_labels = None

    # First: a deterministic alternating start, then random restarts.
    s0 = np.ones(n, dtype=np.float64)
    s0[1::2] = -1.0
    cut, s = _local_search(A, s0.copy(), deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = s

    # Random restarts while time remains.
    while time.time() < deadline:
        s = rng.choice([-1.0, 1.0], size=n)
        cut, s = _local_search(A, s, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = s

    if best_labels is None:
        # Fallback (shouldn't happen): alternating
        labels = np.zeros(n, dtype=int)
        labels[1::2] = 1
        return labels

    # Convert ±1 to 0/1
    labels = ((best_labels + 1.0) / 2.0).astype(int)
    return labels
