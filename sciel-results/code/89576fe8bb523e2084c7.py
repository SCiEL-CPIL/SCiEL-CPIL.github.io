def construct():
    import time
    import random
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    start_time = time.time()
    deadline = start_time + 115.0

    d = 7
    nverts = 1 << d  # 128

    # 1. Build edges
    edges = []
    edge_map = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx

    num_edges = len(edges)  # 448

    # 2. Build faces
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    face_vertices_set = set()

    for i in range(d):
        for j in range(i + 1, d):
            for u in range(nverts):
                if (u & (1 << i)) != 0:
                    continue
                if (u & (1 << j)) != 0:
                    continue

                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)

                face_key = tuple(sorted([v1, v2, v3, v4]))
                if face_key in face_vertices_set:
                    continue

                face_vertices_set.add(face_key)
                f_idx = len(face_edges_list)

                e1 = edge_map[(min(v1, v2), max(v1, v2))]
                e2 = edge_map[(min(v1, v3), max(v1, v3))]
                e3 = edge_map[(min(v2, v4), max(v2, v4))]
                e4 = edge_map[(min(v3, v4), max(v3, v4))]

                face_edges_list.append((e1, e2, e3, e4))

                edge_faces[e1].append(f_idx)
                edge_faces[e2].append(f_idx)
                edge_faces[e3].append(f_idx)
                edge_faces[e4].append(f_idx)

    num_faces = len(face_edges_list)  # 336 faces

    # Build numpy arrays
    face_edge_arr = np.zeros((num_faces, num_edges), dtype=np.int8)
    for f in range(num_faces):
        for e in face_edges_list[f]:
            face_edge_arr[f, e] = 1

    edge_faces_arr = [np.array(edge_faces[e], dtype=np.int32) for e in range(num_edges)]

    # Precompute: for each edge, the list of other edges sharing a face
    edge_neighbors = []
    for e in range(num_edges):
        s = set()
        for f in edge_faces[e]:
            for fe in face_edges_list[f]:
                s.add(fe)
        s.discard(e)
        edge_neighbors.append(s)

    # Spectral initialization: build signed adjacency of line graph
    # A[e][f] = -1 if e and f share a face (conflict), 0 otherwise
    # Maximizing x^T A x over x in {0,1} is related to C4-free edge sets
    # The top eigenvector of A gives a soft assignment
    row_idx = []
    col_idx = []
    data = []
    for e in range(num_edges):
        for f in edge_neighbors[e]:
            row_idx.append(e)
            col_idx.append(f)
            data.append(-1)

    A = csr_matrix((data, (row_idx, col_idx)), shape=(num_edges, num_edges))

    # Get top eigenvector (most negative eigenvalue of A corresponds to max cut-like)
    # We want to maximize sum of -A[e][f]*x[e]*x[f] = minimize A*x*x
    # Top eigenvector of -A gives the direction
    try:
        eigvals, eigvecs = eigsh(-A, k=3, which='LM')
        # Use the eigenvector for the largest eigenvalue of -A
        top_eigvec = eigvecs[:, 0]
        # Make it non-negative by flipping if needed
        if np.sum(top_eigvec) < 0:
            top_eigvec = -top_eigvec
    except:
        top_eigvec = np.ones(num_edges)

    # Greedy initialization
    def random_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        selected = np.zeros(num_edges, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        order = np.random.permutation(num_edges)
        count = 0
        for e in order:
            faces_e = edge_faces_arr[e]
            if np.all(face_count[faces_e] < 3):
                selected[e] = True
                count += 1
                face_count[faces_e] += 1
        return selected, face_count, count

    def spectral_greedy():
        """Use spectral ordering to greedily select edges."""
        selected = np.zeros(num_edges, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        # Order edges by spectral component (descending)
        order = np.argsort(-top_eigvec)
        count = 0
        for e in order:
            faces_e = edge_faces_arr[e]
            if np.all(face_count[faces_e] < 3):
                selected[e] = True
                count += 1
                face_count[faces_e] += 1
        return selected, face_count, count

    # Try multiple initializations
    best_selected = None
    best_face_count = None
    best_count = 0

    # Spectral initialization
    s, fc, c = spectral_greedy()
    if c > best_count:
        best_count = c
        best_selected = s
        best_face_count = fc

    # Random greedy
    t0 = time.time()
    while time.time() - t0 < 4.0:
        s, fc, c = random_greedy()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    selected = best_selected.copy()
    face_count = best_face_count.copy()
    count = best_count
    best_found = count
    best_selected_copy = selected.copy()
    best_face_count_copy = face_count.copy()

    # Local search helpers
    def try_add_edge(e):
        nonlocal count, best_found, best_selected_copy, best_face_count_copy
        faces_e = edge_faces_arr[e]
        if np.any(face_count[faces_e] == 3):
            return False
        selected[e] = True
        count += 1
        face_count[faces_e] += 1
        if count > best_found:
            best_found = count
            best_selected_copy = selected.copy()
            best_face_count_copy = face_count.copy()
        return True

    def remove_edge(e):
        nonlocal count
        selected[e] = False
        count -= 1
        faces_e = edge_faces_arr[e]
        face_count[faces_e] -= 1
        return True

    def can_add(e):
        faces_e = edge_faces_arr[e]
        return not np.any(face_count[faces_e] == 3)

    def find_addable_edge(exclude=None):
        """Find an unselected edge that can be added. Returns index or -1."""
        for e in range(num_edges):
            if selected[e]:
                continue
            if exclude is not None and e in exclude:
                continue
            if can_add(e):
                return e
        return -1

    # Main local search loop
    neutral_swap_count = 0
    max_neutral_swaps = 5000
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        # Phase 1: Greedy Add - try to add any edge that doesn't violate constraints
        added = False
        for e in range(num_edges):
            if selected[e]:
                continue
            if can_add(e):
                try_add_edge(e)
                added = True
                neutral_swap_count = 0
                no_improve_count = 0
                break

        if added:
            continue

        # Phase 2: Try swaps
        e = random.randint(0, num_edges - 1)
        if selected[e]:
            # Pick a random selected edge to remove
            sel_indices = np.where(selected)[0]
            if len(sel_indices) > 0:
                r = sel_indices[random.randint(0, len(sel_indices) - 1)]
                remove_edge(r)
                add_e = find_addable_edge(exclude={r})
                if add_e != -1:
                    try_add_edge(add_e)
                    no_improve_count = 0
                else:
                    try_add_edge(r)
                    neutral_swap_count += 1
            continue

        # e is unselected
        b_faces = []
        for f in edge_faces[e]:
            if face_count[f] == 3:
                b_faces.append(f)

        if not b_faces:
            try_add_edge(e)
            neutral_swap_count = 0
            no_improve_count = 0
            continue

        # Find edges to remove that are in blocking faces
        candidate_r = None
        if len(b_faces) == 1:
            f0 = b_faces[0]
            sel_in_f0 = [fe for fe in face_edges_list[f0] if selected[fe]]
            if sel_in_f0:
                candidate_r = random.choice(sel_in_f0)
        else:
            # Multiple blocking faces - find intersection
            common = set(face_edges_list[b_faces[0]])
            for f in b_faces[1:]:
                common &= set(face_edges_list[f])
            common_sel = [fe for fe in common if selected[fe]]
            if common_sel:
                candidate_r = random.choice(common_sel)
            else:
                # No single edge in all blocking faces. Try 2-for-1: remove 2 edges
                all_sel_in_bf = set()
                for f in b_faces:
                    for fe in face_edges_list[f]:
                        if selected[fe]:
                            all_sel_in_bf.add(fe)

                pair_found = False
                sel_list = list(all_sel_in_bf)
                random.shuffle(sel_list)
                for i in range(min(len(sel_list), 50)):
                    for j in range(i + 1, min(len(sel_list), 50)):
                        r1, r2 = sel_list[i], sel_list[j]
                        remove_edge(r1)
                        remove_edge(r2)
                        if can_add(e):
                            try_add_edge(e)
                            pair_found = True
                            break
                        else:
                            try_add_edge(r1)
                            try_add_edge(r2)
                        if pair_found:
                            break
                    if pair_found:
                        break

                if pair_found:
                    no_improve_count = 0
                    continue

                # If no pair works, try single removal with best alternative add
                if sel_list:
                    r = sel_list[0]
                    remove_edge(r)
                    add_e = find_addable_edge(exclude={r})
                    if add_e != -1:
                        try_add_edge(add_e)
                        no_improve_count = 0
                    else:
                        try_add_edge(r)
                        neutral_swap_count += 1
                else:
                    neutral_swap_count += 1
                continue

        # We have a candidate_r
        remove_edge(candidate_r)
        if can_add(e):
            try_add_edge(e)
            no_improve_count = 0
        else:
            add_e = find_addable_edge(exclude={candidate_r})
            if add_e != -1:
                try_add_edge(add_e)
            else:
                try_add_edge(candidate_r)
            neutral_swap_count += 1

        if count <= best_found:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if neutral_swap_count > max_neutral_swaps or no_improve_count > max_no_improve:
            t0 = time.time()
            while time.time() - t0 < 2.0:
                s, fc, c = random_greedy()
                if c > best_count:
                    best_count = c
                    best_selected = s
                    best_face_count = fc
            selected = best_selected.copy()
            face_count = best_face_count.copy()
            count = best_count
            neutral_swap_count = 0
            no_improve_count = 0

    # Final polish - multiple passes
    for _ in range(3):
        for e in range(num_edges):
            if selected[e]:
                continue
            if can_add(e):
                try_add_edge(e)

    result = []
    for i in range(num_edges):
        if best_selected_copy[i]:
            result.append(list(edges[i]))

    return result
