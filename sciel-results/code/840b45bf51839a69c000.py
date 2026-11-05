"""Program for clique partition of two dense random graphs.

Strategy: spectral clustering on the Laplacian to identify dense regions,
then greedy maximum-clique extraction prioritizing eigenvector centrality.
Every edge is placed in exactly one clique; each clique has >= 2 vertices.
Minimizes total number of cliques.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def _random_graph(n, p, seed):
    """Build the edge list exactly as the verifier does."""
    from random import Random
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return edges


def _eigenvector_centrality(adj, n, tol=1e-10, max_iter=200):
    """Power iteration for eigenvector centrality."""
    deg = np.asarray(adj.sum(axis=1), dtype=float)
    deg[deg == 0] = 1.0
    x = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        x_new = (adj @ x) / deg
        x_new = x_new / (np.linalg.norm(x_new) + 1e-30)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def _spectral_clusters(adj, n, k=4):
    """Cluster vertices using the Fiedler vector(s) of the normalized Laplacian."""
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return list(range(n))
    # Use the smallest eigenvectors (excluding the trivial constant one)
    n_use = min(k, n - 1)
    if n_use < 1:
        return list(range(n))
    vecs = eigvecs[:, 1:1 + n_use]
    # Simple k-means on the embedding
    # initialize with k far-apart points
    pts = vecs
    idx = np.arange(n)
    # pick initial centroids
    centroids = [pts[0]]
    for _ in range(1, n_use):
        d = np.min([np.sum((pts - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(pts[idx[np.argmax(d)]])
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        d = np.stack([np.sum((pts - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(n_use):
            mask = labels == ci
            if mask.any():
                centroids[ci] = pts[mask].mean(axis=0)
    return labels.tolist()


def _greedy_clique_partition(adj, n, order):
    """
    Greedily partition edges into cliques.
    `order` is a permutation of vertices to try as seeds (high priority first).
    Returns a list of cliques (each a list of vertex indices, size >= 2).
    """
    # remaining edges as a set of frozensets for O(1) check
    edge_set = set()
    for u in range(n):
        for v in adj[u, :]:
            pass
    # Build edge set from adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edge_set.add((u, v))

    # remaining degree per vertex (number of incident remaining edges)
    rem_deg = [0] * n
    for (u, v) in edge_set:
        rem_deg[u] += 1
        rem_deg[v] += 1

    cliques = []

    # For fast "is (u,v) still an edge" we keep edge_set
    # We'll process vertices in priority order; but a vertex may be used in
    # multiple cliques. Instead, we repeatedly pick the best seed among vertices
    # that still have remaining edges, using the given priority (centrality).

    # Precompute a priority rank: lower rank = higher priority
    rank = {v: i for i, v in enumerate(order)}

    while edge_set:
        # choose seed: vertex with remaining edges and best (lowest) rank
        seed = None
        best_rank = None
        for v in range(n):
            if rem_deg[v] > 0:
                r = rank[v]
                if best_rank is None or r < best_rank:
                    best_rank = r
                    seed = v
        if seed is None:
            break

        # Build a clique starting from seed.
        clique = [seed]
        # Candidate neighbors: vertices adjacent to all current clique members,
        # with at least one remaining edge to the clique.
        # Greedily add the candidate that maximizes remaining edges to clique.
        while True:
            # candidates
            cands = []
            for v in range(n):
                if v in clique:
                    continue
                if rem_deg[v] == 0:
                    continue
                ok = True
                for c in clique:
                    if c < v:
                        e = (c, v)
                    else:
                        e = (v, c)
                    if e not in edge_set:
                        ok = False
                        break
                if ok:
                    cands.append(v)
            if not cands:
                break
            # pick candidate with max remaining degree (tie-break by rank)
            best = None
            best_key = None
            for v in cands:
                key = (rem_deg[v], -rank[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            clique.append(best)

        # Now remove all edges among clique members that still exist
        added = False
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    edge_set.remove(e)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                    added = True
        # Only record clique if it actually covered an edge (size >= 2 guaranteed
        # because seed had rem_deg>0 and we added at least one neighbor, or if
        # no neighbor was added we must not record a single-vertex clique).
        if len(clique) >= 2 and added:
            cliques.append(clique)

    return cliques


def _partition_one(n, p, seed):
    edges = _random_graph(n, p, seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1

    # Eigenvector centrality for priority ordering
    try:
        evc = _eigenvector_centrality(adj, n)
    except Exception:
        evc = np.arange(n, dtype=float)

    # Spectral clusters to process dense regions first
    try:
        labels = _spectral_clusters(adj, n, k=4)
    except Exception:
        labels = list(range(n))

    # Build an order: vertices sorted by (cluster density desc, centrality desc)
    # Compute per-cluster remaining-edge density as a proxy
    cluster_size = defaultdict(int)
    cluster_edges = defaultdict(int)
    for v in range(n):
        cluster_size[labels[v]] += 1
    for (u, v) in edges:
        if labels[u] == labels[v]:
            cluster_edges[labels[u]] += 1

    cluster_score = {}
    for c in cluster_size:
        sz = cluster_size[c]
        cluster_score[c] = (cluster_edges[c] / (sz * sz) if sz > 0 else 0.0, -evc.mean() if False else 0.0)

    # Order vertices: group by cluster (densest first), within cluster by centrality desc
    clusters_sorted = sorted(cluster_size.keys(), key=lambda c: (cluster_edges[c] / max(1, cluster_size[c] ** 2)), reverse=True)
    order = []
    for c in clusters_sorted:
        vs = [v for v in range(n) if labels[v] == c]
        vs.sort(key=lambda v: -evc[v])
        order.extend(vs)

    cliques = _greedy_clique_partition(adj, n, order)

    # Convert to lists of vertex lists
    return [[int(x) for x in cl] for cl in cliques]


def construct():
    res1 = _partition_one(32, 0.6, 601)
    res2 = _partition_one(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
