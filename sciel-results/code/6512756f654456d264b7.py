# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Compute priority scores for vertices for the fixed greedy independent-set decoder.
    Higher score => selected earlier.

    Strategy:
    1. Fast structural fallback score (cheap, always available).
    2. If there is budget, run a local "value" simulation: for a set of sampled
       candidate roots, run the greedy decoder and record the marginal gain each
       vertex contributes (the independent-set size realized with that vertex forced
       in early vs. without it). Sum these gains over roots to get a per-vertex
       value score that directly targets the metric (greedy independent-set size).
    3. Blend the value score with the structural score for robustness.
    """
    start_time = time.time()
    DEADLINE = 135.0  # safety margin well under 150s

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]

    if n == 0:
        return np.array([])

    # ---- Structural baseline (cheap) -------------------------------------
    degrees = adj.sum(axis=1).astype(np.float64)
    adj_f = adj.astype(np.float64)

    neighbor_degree_sums = adj_f @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = np.where(
            degrees > 0, neighbor_degree_sums / np.maximum(degrees, 1.0), 0.0
        )

    # Structural score: reward low own degree + high-degree neighbors
    struct = (mean_neighbor_degrees - degrees) + 0.5 / (1.0 + degrees)

    # Spectral feature: Fiedler vector (2nd-smallest eigenvector of graph Laplacian).
    # Vertices with large |fiedler| are at the "periphery" of the main component —
    # less central to connectivity, so safer to include in an independent set.
    fiedler_mag = np.zeros(n, dtype=np.float64)
    if n >= 3:
        try:
            if time.time() - start_time < DEADLINE - 5.0:
                # Build Laplacian: L = D - A
                deg_int = degrees.astype(np.int64)
                L = csr_matrix(
                    (deg_int - adj_f), shape=(n, n)
                )
                # eigsh for smallest algebraic eigenvalues (shift-invert via sigma)
                # For a connected graph, smallest eigenvalue is 0; we want the 2nd.
                # Use sigma=0 with which='LA' on (L - sigma*I) won't work directly.
                # Instead, find smallest-magnitude eigenvalues: k=2, which='LM' on L
                # but L is PSD so smallest magnitude = smallest algebraic.
                # eigsh with which='SM' can be slow; use shift-invert: sigma=-1
                try:
                    vals, vecs = eigsh(L, k=2, sigma=-1.0, which='LM')
                    # Sort by eigenvalue
                    idx = np.argsort(vals)
                    # Fiedler is the eigenvector for the 2nd smallest eigenvalue
                    # (the smallest is ~0 for connected graphs)
                    if len(idx) >= 2:
                        fiedler_vec = vecs[:, idx[1]]
                        fiedler_mag = np.abs(fiedler_vec)
                except Exception:
                    # Fallback: which='SM'
                    try:
                        vals, vecs = eigsh(L, k=2, which='SM')
                        idx = np.argsort(vals)
                        if len(idx) >= 2:
                            fiedler_vec = vecs[:, idx[1]]
                            fiedler_mag = np.abs(fiedler_vec)
                    except Exception:
                        pass
        except Exception:
            pass

    if time.time() - start_time > DEADLINE:
        return struct

    # ---- Local value simulation ------------------------------------------
    # Greedy decoder simulation that returns per-vertex marginal gain.
    # We approximate marginal gain of vertex v as:
    #   gain[v] = (number of vertices the greedy decoder would still pick
    #             if v is forced selected first) - (number it picks without v)
    # Doing this exactly for every v is O(n^2) greedy runs; instead we sample
    # roots and accumulate how each vertex's selection helps.

    # Precompute neighbor lists as boolean rows (adjacency) for fast masking.
    # Use a heap-free greedy: repeatedly pick the highest-priority available vertex.
    # Priority for the inner simulation uses the structural score (fixed ordering).

    # To keep it fast, we simulate a few greedy runs with different "forced" starts
    # and track, for each run, which vertices got selected. A vertex's total
    # "presence" across runs where it was selected early indicates value.
    # Better: directly measure marginal contribution.

    # We'll do K sampled greedy decodes; for each, we also compute the set of
    # vertices that were selected. Then value[v] = average over samples of
    #   (size of greedy set when v is removed from availability before start)
    #   vs (size when v is allowed).
    # Simpler and effective: run greedy with a randomized tie-break / perturbed
    # priority and accumulate selection frequency weighted by the marginal effect.

    # Most direct: for each sampled root r, run greedy forcing r first, record set S_r.
    # Then for any vertex v, an estimate of its value is the increase in |S| when v
    # is included. We approximate value[v] by the frequency that v is in S_r for
    # roots r that are "near" v, plus a baseline.

    # Given time, use a pragmatic approach: simulate greedy with the structural
    # priority, and compute for each vertex the number of other vertices it
    # "unblocks" / the size of the resulting set. Use multiple priority orderings
    # (perturbed) to get a robust value.

    value = np.zeros(n, dtype=np.float64)

    # Number of simulation runs based on budget
    # Each run is O(n log n) with a simple selection; budget ~120s.
    # Estimate per-run cost empirically with a small trial.
    def greedy_set(order):
        """Given a fixed ordering (indices), return the set (bool) selected."""
        selected = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        for v in order:
            if not remaining[v]:
                continue
            # select v
            selected[v] = True
            # remove v and its neighbors from remaining
            remaining[v] = False
            nb = adj[v]
            remaining[nb] = False
        return selected

    # Build a base ordering by structural score (descending).
    base_order = np.argsort(-struct)

    # Trial run to estimate cost
    t0 = time.time()
    greedy_set(base_order)
    per_run = max(time.time() - t0, 1e-6)

    # Decide number of runs we can afford
    remaining_time = DEADLINE - (time.time() - start_time)
    max_runs = int(max(1, remaining_time / (per_run * 1.5)))
    max_runs = min(max_runs, 400)

    rng = np.random.default_rng(12345)

    for it in range(max_runs):
        if time.time() - start_time > DEADLINE:
            break
        # Perturb the structural score to get diverse orderings.
        # Add small noise so ties break differently and we explore the landscape.
        noise = rng.standard_normal(n) * (0.05 * (struct.std() + 1e-9))
        perturbed = struct + noise
        order = np.argsort(-perturbed)
        sel = greedy_set(order)
        # Accumulate: vertices selected get credit proportional to how "early"
        # they were in the order (earlier => more decisive).
        # Rank position of each selected vertex in this order:
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        # Credit = (n - rank) for selected vertices (earlier = bigger credit)
        credit = np.zeros(n, dtype=np.float64)
        credit[sel] = (n - rank[sel]).astype(np.float64)
        value += credit

    # Normalize value to comparable scale with struct and spectral
    vstd = value.std()
    sstd = struct.std()
    fstd = fiedler_mag.std()

    if vstd > 1e-12 and sstd > 1e-12:
        value_norm = (value - value.mean()) / vstd
        struct_norm = (struct - struct.mean()) / sstd
        if fstd > 1e-12:
            fiedler_norm = (fiedler_mag - fiedler_mag.mean()) / fstd
            # Blend: value (simulation) + structural + spectral periphery
            scores = 0.55 * value_norm + 0.25 * struct_norm + 0.20 * fiedler_norm
        else:
            scores = 0.7 * value_norm + 0.3 * struct_norm
    else:
        if fstd > 1e-12:
            fiedler_norm = (fiedler_mag - fiedler_mag.mean()) / fstd
            scores = 0.6 * struct_norm + 0.4 * fiedler_norm
        else:
            scores = struct

    return scores
# EVOLVE-BLOCK-END
