"""Improved Maximum Cut: spectral init + fast incremental local search + restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(labels, A):
    # count edges crossing: A[i,j] with labels[i] != labels[j]; symmetric -> /2
    return int(np.sum(A[labels == 0][:, labels == 1]))


def _local_search(labels, A, row_deg, deadline, max_sweeps=200):
    # Incremental 1-flip local search.
    # For vertex v, current same-side neighbors = s[v]; other = deg[v] - s[v].
    # Flipping v is beneficial iff s[v] > deg[v] - s[v], i.e. s[v] > deg[v]/2.
    # On flip, for each neighbor u: if labels[u] == labels[v] (before flip),
    #   after flip they differ, so s[u] decreases by 1 (they no longer count as same-side).
    #   if labels[u] != labels[v] (before), after flip they become same, s[u] increases by 1.
    n = len(labels)
    labels = labels.copy()
    deg = row_deg
    # s[v] = number of neighbors of v with same label as v
    same = np.zeros(n, dtype=np.int64)
    # compute initial s efficiently: for each v, count neighbors with same label
    # s[v] = sum_j A[v,j] * (labels[j]==labels[v])
    for v in range(n):
        same[v] = A[v][labels == labels[v]].sum()

    improved = True
    sweep = 0
    while improved and sweep < max_sweeps:
        if time.time() > deadline:
            break
        improved = False
        sweep += 1
        for v in range(n):
            if time.time() > deadline:
                break
            s = same[v]
            d = deg[v]
            if s > d - s:  # flipping reduces same-side edges -> increases cut
                # flip v
                old = labels[v]
                labels[v] = 1 - old
                improved = True
                # update same[] for neighbors
                nbrs = np.nonzero(A[v])[0]
                for u in nbrs:
                    if u == v:
                        continue
                    if labels[u] == old:
                        # before flip u was same as v, now different
                        same[u] -= 1
                    else:
                        # before flip u was different from v, now same
                        same[u] += 1
                # update same[v]
                same[v] = d - s
    return labels


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.int8)
    n = len(A)
    deg = A.sum(axis=1).astype(np.int64)

    # deadline: be safe, leave margin
    start = time.time()
    deadline = start + 110.0  # 110s of 120s budget

    best_labels = None
    best_cut = -1

    def consider(labels):
        nonlocal best_labels, best_cut
        c = _cut_size(labels, A)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()

    # --- Spectral initialization (Fiedler vector of L = D - A) ---
    try:
        L = np.diag(deg.astype(np.float64)) - A.astype(np.float64)
        # use scipy if available for faster eig, else numpy
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            Ls = csr_matrix(L)
            # Fiedler vector: second smallest eigenvalue
            vals, vecs = eigsh(Ls, k=2, which='SM')
            # eigsh returns ascending; index 1 is Fiedler
            fiedler = vecs[:, 1]
        except Exception:
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]
        labels = (fiedler > 0).astype(np.int8)
        # ensure both partitions non-trivial
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        labels = _local_search(labels, A, deg, deadline)
        consider(labels)
    except Exception:
        pass

    # --- Random restarts with local search ---
    rng = np.random.default_rng(0)
    # number of restarts: try to do as many as time allows
    # Estimate time per local search; adapt.
    for restart in range(200):
        if time.time() > deadline:
            break
        if restart % 5 == 0:
            # random init
            labels = rng.integers(0, 2, size=n).astype(np.int8)
        else:
            # perturb best
            if best_labels is not None:
                labels = best_labels.copy()
                # flip a random fraction
                k = max(1, n // 10)
                idx = rng.choice(n, size=k, replace=False)
                labels[idx] = 1 - labels[idx]
            else:
                labels = rng.integers(0, 2, size=n).astype(np.int8)
        labels = _local_search(labels, A, deg, deadline)
        consider(labels)
        if time.time() > deadline:
            break

    if best_labels is None:
        best_labels = np.arange(n) % 2

    # Final polish pass with tighter budget if time remains
    if time.time() < deadline - 2.0:
        labels = _local_search(best_labels, A, deg, deadline)
        consider(labels)

    return best_labels
# EVOLVE-BLOCK-END
