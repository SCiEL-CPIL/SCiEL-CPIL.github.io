import time
import random
import numpy as np

N = 20

# Precompute triples for each vertex
triples_for_vertex = [[] for _ in range(N)]
triples_indices = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            t_idx = len(triples_indices)
            triples_indices.append((i, j, k))
            triples_for_vertex[i].append((t_idx, j, k))
            triples_for_vertex[j].append((t_idx, i, k))
            triples_for_vertex[k].append((t_idx, i, j))

total_triples = len(triples_indices)
all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
num_all_edges = len(all_edges)

# Precompute for each vertex, the list of (t_idx, other1, other2) as numpy arrays for speed
triples_np = []
for i in range(N):
    if triples_for_vertex[i]:
        arr = np.array(triples_for_vertex[i], dtype=np.int32)  # shape (k, 3): [t_idx, x, y]
        triples_np.append(arr)
    else:
        triples_np.append(np.empty((0, 3), dtype=np.int32))

# Precompute edge index mapping for O(1) edge lookup
edge_to_idx = {}
for idx, (u, v) in enumerate(all_edges):
    edge_to_idx[(u, v)] = idx
    edge_to_idx[(v, u)] = idx


def can_add_edge_fast(M, C, u, v):
    """Vectorized check if adding edge (u,v) keeps K3,3-free."""
    arr_u = triples_np[u]
    if len(arr_u) > 0:
        t_idx_u = arr_u[:, 0]
        x_u = arr_u[:, 1]
        y_u = arr_u[:, 2]
        mask = (M[x_u, v] == 1) & (M[y_u, v] == 1) & (C[t_idx_u] == 2)
        if np.any(mask):
            return False

    arr_v = triples_np[v]
    if len(arr_v) > 0:
        t_idx_v = arr_v[:, 0]
        x_v = arr_v[:, 1]
        y_v = arr_v[:, 2]
        mask = (M[x_v, u] == 1) & (M[y_v, u] == 1) & (C[t_idx_v] == 2)
        if np.any(mask):
            return False

    return True


def get_missing_edges(M):
    """Return list of missing edges (u,v) where M[u,v]==0 and u<v."""
    missing = []
    for u, v in all_edges:
        if M[u, v] == 0:
            missing.append((u, v))
    return missing


def add_edge(M, C, edges, u, v):
    M[u, v] = 1
    M[v, u] = 1
    edges.append([u, v])
    arr_u = triples_np[u]
    if len(arr_u) > 0:
        t_idx_u = arr_u[:, 0]
        x_u = arr_u[:, 1]
        y_u = arr_u[:, 2]
        mask = (M[x_u, v] == 1) & (M[y_u, v] == 1)
        C[t_idx_u[mask]] += 1
    arr_v = triples_np[v]
    if len(arr_v) > 0:
        t_idx_v = arr_v[:, 0]
        x_v = arr_v[:, 1]
        y_v = arr_v[:, 2]
        mask = (M[x_v, u] == 1) & (M[y_v, u] == 1)
        C[t_idx_v[mask]] += 1


def remove_edge(M, C, edges, u, v):
    # Compute masks BEFORE modifying M
    arr_u = triples_np[u]
    if len(arr_u) > 0:
        t_idx_u = arr_u[:, 0]
        x_u = arr_u[:, 1]
        y_u = arr_u[:, 2]
        mask_u = (M[x_u, v] == 1) & (M[y_u, v] == 1)
    else:
        mask_u = None

    arr_v = triples_np[v]
    if len(arr_v) > 0:
        t_idx_v = arr_v[:, 0]
        x_v = arr_v[:, 1]
        y_v = arr_v[:, 2]
        mask_v = (M[x_v, u] == 1) & (M[y_v, u] == 1)
    else:
        mask_v = None

    # Now modify M
    M[u, v] = 0
    M[v, u] = 0

    # Remove from edges list
    for i, e in enumerate(edges):
        if e[0] == u and e[1] == v:
            edges.pop(i)
            break

    # Decrement C using pre-computed masks
    if mask_u is not None:
        C[t_idx_u[mask_u]] -= 1
    if mask_v is not None:
        C[t_idx_v[mask_v]] -= 1


def local_search_fast(M, C, edges, deadline):
    """Fast local search: add edges, then do 1-for-2 swaps."""
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase 1: Try to add any missing edge
        for u, v in all_edges:
            if M[u, v] == 0:
                if can_add_edge_fast(M, C, u, v):
                    add_edge(M, C, edges, u, v)
                    improved = True
                    break

        # Phase 2: If stuck, try 1-for-2 swaps
        if not improved:
            found_swap = False
            edge_list = list(edges)
            random.shuffle(edge_list)
            for e1 in edge_list:
                if time.time() > deadline:
                    break
                u1, v1 = e1[0], e1[1]
                remove_edge(M, C, edges, u1, v1)

                added = []
                for u2, v2 in all_edges:
                    if M[u2, v2] == 0:
                        if can_add_edge_fast(M, C, u2, v2):
                            add_edge(M, C, edges, u2, v2)
                            added.append((u2, v2))
                            if len(added) == 2:
                                break

                if len(added) >= 2:
                    found_swap = True
                    improved = True
                    break
                else:
                    for u2, v2 in added:
                        remove_edge(M, C, edges, u2, v2)
                    add_edge(M, C, edges, u1, v1)

            if not found_swap:
                break

    return edges


def hill_climbing_2swap_fast(M, C, edges, deadline):
    """2-for-3 swaps with better edge selection."""
    improved = True
    attempts = 0
    max_attempts = 10000
    while improved and time.time() < deadline and attempts < max_attempts:
        improved = False
        attempts += 1

        n_edges = len(edges)
        if n_edges < 2:
            break

        # Sample a random pair of edges to remove
        i = random.randint(0, n_edges - 1)
        j = random.randint(0, n_edges - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        u1, v1 = edges[i][0], edges[i][1]
        u2, v2 = edges[j][0], edges[j][1]

        # Remove both
        remove_edge(M, C, edges, u1, v1)
        remove_edge(M, C, edges, u2, v2)

        # Try to add 3 new edges
        added = []
        for u3, v3 in all_edges:
            if len(added) == 3:
                break
            if M[u3, v3] == 0:
                if can_add_edge_fast(M, C, u3, v3):
                    add_edge(M, C, edges, u3, v3)
                    added.append((u3, v3))

        if len(added) >= 3:
            improved = True
        else:
            # Undo
            for u3, v3 in added:
                remove_edge(M, C, edges, u3, v3)
            # Restore original edges
            add_edge(M, C, edges, u1, v1)
            add_edge(M, C, edges, u2, v2)

    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_edges = None
    best_count = 0

    for restart in range(500):
        if time.time() > deadline:
            break

        # Create diverse initial order
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)

        M = np.zeros((N, N), dtype=np.int8)
        C = np.zeros(total_triples, dtype=np.int8)
        edges = []

        # Greedy construction
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            if can_add_edge_fast(M, C, u, v):
                add_edge(M, C, edges, u, v)

        # Fast local search
        if time.time() < deadline:
            edges = local_search_fast(M, C, edges, deadline)

        # 2-swap polish if time permits
        if time.time() < deadline - 3:
            edges = hill_climbing_2swap_fast(M, C, edges, deadline)

        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = [e[:] for e in edges]

        # Early stop if we've found a very good solution
        if best_count >= 78:
            break

    if best_edges is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return best_edges
