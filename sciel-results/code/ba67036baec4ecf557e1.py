import numpy as np
import time
import itertools
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    D = 11

    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330

    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)

    # Add 22 weight-1 vectors (unit vectors with signs)
    # These are compatible with all weight-4 vectors (dot product is at most 1 in absolute value)
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = sign
            all_vectors.append(v)

    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280 + 22 = 5302

    # Build adjacency list for vectors
    # Conflict if dot > 2
    # For weight-4 vectors: conflict if they share 3+ elements with same signs
    # For weight-1 vectors: no conflicts with weight-4 (|dot| <= 1)
    # For weight-1 vs weight-1: conflict if same position and same sign (dot = 1 > 2? No, 1 <= 2)
    # Actually, dot > 2 means: for weight-4, need 3 shared positions with same sign
    # For weight-1, max dot with weight-4 is 1, so no conflict
    # For weight-1 vs weight-1: dot is 0 or ±1, never > 2, so no conflicts

    # So we only need to build conflicts among weight-4 vectors
    # Weight-1 vectors (indices 5280 to 5301) have no conflicts

    # Build conflict graph for weight-4 vectors only
    n_w4 = 5280

    # Build adjacency for weight-4 vectors
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)

    adj_w4 = [[] for _ in range(n_w4)]

    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]

                base1 = s1 * 16
                base2 = s2 * 16

                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]

                d = v1 @ v2.T

                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj_w4[idx1].append(idx2)
                    adj_w4[idx2].append(idx1)

    # Full adjacency: weight-1 vectors have no neighbors
    adj = adj_w4 + [[] for _ in range(n_total - n_w4)]

    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_k = 0
    best_selected = None

    max_restarts = 300

    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break

        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)

        # Greedy Minimum Degree MIS
        # Use a priority queue for efficiency
        # But simpler: just iterate and pick min degree each time

        while avail.any():
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break

            degs = deg[avail_indices]
            min_deg = np.min(degs)

            min_indices_rel = np.where(degs == min_deg)[0]

            pick_rel = min_indices_rel[rng.integers(len(min_indices_rel))]
            u = avail_indices[pick_rel]

            selected[u] = True
            avail[u] = False

            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1

        k = np.sum(selected)

        # Local search: try swaps to improve
        # For each selected node, try swapping with a non-selected node that's not a neighbor
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            selected_indices = np.where(selected)[0]
            non_selected_indices = np.where(~selected)[0]

            # Try random swaps
            for _ in range(100):
                if (time.time() - start_time) > deadline - 5:
                    break

                # Pick a random selected node to remove
                if len(selected_indices) == 0:
                    break
                i = selected_indices[rng.integers(len(selected_indices))]

                # Find non-selected nodes that are not neighbors of i
                # and whose addition doesn't conflict with other selected nodes
                # Simplified: just try to add a non-selected node that has no selected neighbors
                # Actually, we need to remove i and add j where j conflicts with no selected node except i

                # Check if removing i allows adding some j
                # j must not conflict with any selected node other than i
                # And j must conflict with i (otherwise we could have added j without removing i)

                # This is complex. Let's use a simpler heuristic:
                # Try to add a random non-selected node that has few selected neighbors
                # If it has 0 selected neighbors, add it (improvement)
                # If it has 1 selected neighbor, remove that neighbor and add j (swap)

                if len(non_selected_indices) == 0:
                    break

                j = non_selected_indices[rng.integers(len(non_selected_indices))]

                # Count selected neighbors of j
                sel_neighbors = 0
                sel_neighbor_idx = -1
                for v in adj[j]:
                    if selected[v]:
                        sel_neighbors += 1
                        sel_neighbor_idx = v
                        if sel_neighbors > 1:
                            break

                if sel_neighbors == 0:
                    # Can add j directly
                    selected[j] = True
                    k += 1
                    improved = True
                    # Update non_selected_indices
                    non_selected_indices = np.where(~selected)[0]
                    selected_indices = np.where(selected)[0]
                elif sel_neighbors == 1:
                    # Swap: remove sel_neighbor_idx, add j
                    # But only if j conflicts with sel_neighbor_idx
                    # (which it must, since sel_neighbor_idx is a neighbor)
                    # Check that removing sel_neighbor_idx doesn't break anything
                    # (it's already removed, so no issue)
                    # Add j
                    selected[sel_neighbor_idx] = False
                    selected[j] = True
                    # k stays the same
                    improved = True
                    non_selected_indices = np.where(~selected)[0]
                    selected_indices = np.where(selected)[0]

        k = np.sum(selected)

        if k > best_k:
            best_k = k
            best_selected = selected.copy()

    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)

    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
