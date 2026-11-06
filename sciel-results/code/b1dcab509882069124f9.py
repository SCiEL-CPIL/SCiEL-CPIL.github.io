def construct():
    import time
    import random
    import numpy as np

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
    face_edges_set = [set(f) for f in face_edges_list]

    # Precompute for each face the list of edges
    # Each edge belongs to exactly 2 faces

    # Greedy initialization with multiple strategies
    def random_greedy(seed=None):
        if seed is not None:
            random.seed(seed)

        selected = [False] * num_edges
        face_count = [0] * num_faces
        order = list(range(num_edges))
        random.shuffle(order)

        count = 0
        for e in order:
            ok = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1

        return selected, face_count, count

    def greedy_by_degree(seed=None):
        """Greedy that prioritizes edges with fewer conflicting faces already saturated."""
        if seed is not None:
            random.seed(seed)

        selected = [False] * num_edges
        face_count = [0] * num_faces
        remaining = list(range(num_edges))
        random.shuffle(remaining)

        count = 0
        while remaining:
            # Pick the edge that has the most "room" (fewest faces at count 3)
            best_idx = 0
            best_room = -1
            for idx in range(min(50, len(remaining))):
                e = remaining[idx]
                room = 0
                for f in edge_faces[e]:
                    if face_count[f] < 3:
                        room += 1
                if room > best_room:
                    best_room = room
                    best_idx = idx
                if best_room == len(edge_faces[e]):
                    break

            e = remaining.pop(best_idx)
            ok = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1

        return selected, face_count, count

    # Try multiple random starts with different strategies
    best_selected = None
    best_face_count = None
    best_count = 0
    t0 = time.time()
    while time.time() - t0 < 4.0:
        s, fc, c = random_greedy()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    # Also try degree-based greedy
    for _ in range(5):
        s, fc, c = greedy_by_degree()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count
    best_found = count
    best_selected_copy = selected[:]
    best_face_count_copy = face_count[:]

    # Local search with simulated annealing
    def try_add_edge(e):
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        selected[e] = True
        count += 1
        for f in edge_faces[e]:
            face_count[f] += 1
        if count > best_found:
            best_found = count
            best_selected_copy = selected[:]
            best_face_count_copy = face_count[:]
        return True

    def remove_edge(e):
        nonlocal selected, face_count, count
        selected[e] = False
        count -= 1
        for f in edge_faces[e]:
            face_count[f] -= 1
        return True

    # Main local search loop with improved swap strategy
    neutral_swap_count = 0
    max_neutral_swaps = 3000
    temperature = 1.0
    cooling_rate = 0.99995
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        # Phase 1: Greedy Add - try to add any edge that doesn't violate constraints
        added = False
        for e in range(num_edges):
            if selected[e]:
                continue
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add_edge(e)
                added = True
                neutral_swap_count = 0
                no_improve_count = 0
                break

        if added:
            continue

        # Phase 2: Try 1-for-1 swaps
        # Pick a random unselected edge
        e = random.randint(0, num_edges - 1)
        if selected[e]:
            # Try to remove a selected edge and add another
            # Pick a random selected edge to potentially remove
            sel_edges = [i for i in range(num_edges) if selected[i]]
            if sel_edges:
                r = random.choice(sel_edges)
                # Find an unselected edge that could replace r
                # We need an edge that, when added, doesn't create a 4-face
                # after removing r
                remove_edge(r)
                # Now try to add some edge
                best_add = None
                for add_e in range(num_edges):
                    if selected[add_e]:
                        continue
                    can_add = True
                    for f in edge_faces[add_e]:
                        if face_count[f] == 3:
                            can_add = False
                            break
                    if can_add:
                        best_add = add_e
                        break
                if best_add is not None:
                    try_add_edge(best_add)
                    no_improve_count = 0
                else:
                    # Revert
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

        # Find a candidate edge to remove that is in all blocking faces
        candidate_r = None
        if b_faces:
            # Get edges in the first blocking face that are selected
            sel_in_first = [re for re in face_edges_list[b_faces[0]] if selected[re]]
            for r in sel_in_first:
                ok = True
                for f in b_faces[1:]:
                    if r not in face_edges_set[f]:
                        ok = False
                        break
                if ok:
                    candidate_r = r
                    break

        if candidate_r is None:
            # Try a more aggressive approach: remove any edge from one blocking face
            # and see if we can add e or some other edge
            first_f = b_faces[0]
            sel_in_first = [re for re in face_edges_list[first_f] if selected[re]]
            if sel_in_first:
                r = random.choice(sel_in_first)
                remove_edge(r)
                # Try to add e
                can_add_e = True
                for f in edge_faces[e]:
                    if face_count[f] == 3:
                        can_add_e = False
                        break
                if can_add_e:
                    try_add_edge(e)
                    no_improve_count = 0
                else:
                    # Try to add some other edge
                    best_add = None
                    for add_e in range(num_edges):
                        if selected[add_e]:
                            continue
                        can_add = True
                        for f in edge_faces[add_e]:
                            if face_count[f] == 3:
                                can_add = False
                                break
                        if can_add:
                            best_add = add_e
                            break
                    if best_add is not None:
                        try_add_edge(best_add)
                    else:
                        # Revert
                        try_add_edge(r)
                    neutral_swap_count += 1
            else:
                neutral_swap_count += 1

        else:
            # Try the swap: remove candidate_r, add e
            remove_edge(candidate_r)
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add_edge(e)
                no_improve_count = 0
            else:
                # Revert
                try_add_edge(candidate_r)
                neutral_swap_count += 1

        # Check for stagnation
        if count <= best_found:
            no_improve_count += 1
        else:
            no_improve_count = 0

        # Restart if stuck
        if neutral_swap_count > max_neutral_swaps or no_improve_count > max_no_improve:
            # Restart with random greedy
            t0 = time.time()
            while time.time() - t0 < 1.5:
                s, fc, c = random_greedy()
                if c > best_count:
                    best_count = c
                    best_selected = s
                    best_face_count = fc
            selected = best_selected[:]
            face_count = best_face_count[:]
            count = best_count
            neutral_swap_count = 0
            no_improve_count = 0
            temperature = 1.0

        # Cool down
        temperature *= cooling_rate

    # Final polish: try to add any remaining edges
    for e in range(num_edges):
        if selected[e]:
            continue
        can_add = True
        for f in edge_faces[e]:
            if face_count[f] == 3:
                can_add = False
                break
        if can_add:
            try_add_edge(e)

    # Return the best solution found
    result = []
    for i in range(num_edges):
        if best_selected_copy[i]:
            result.append(list(edges[i]))

    return result
