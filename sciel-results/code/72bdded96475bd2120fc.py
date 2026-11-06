def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 112  # safety margin vs 150s kill

    d = 7
    nverts = 1 << d  # 128

    # ---- Build edges ----
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

    # ---- Build faces (336) ----
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
                face_key = (v1, v2, v3, v4)
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

    num_faces = len(face_edges_list)
    face_edges_set = [set(f) for f in face_edges_list]

    # Precompute, for each edge, the list of faces it belongs to (already edge_faces).

    # ---- Hitting-set state helpers ----
    # face_count[f] = number of selected edges in face f (0..4)
    # A face is "saturated" if face_count[f] == 3 (adding the 4th would violate).
    # An edge e is addable iff for all f in edge_faces[e], face_count[f] < 3.
    # Removing edge e: for each f in edge_faces[e], face_count[f] -= 1.
    #   If face_count[f] was 3, that face is no longer saturated -> we "hit" it.

    def random_greedy():
        selected = [False] * num_edges
        face_count = [0] * num_faces
        # Damage-aware ordering: process edges that saturate fewer faces first,
        # with random tie-breaking to diversify.
        order = list(range(num_edges))
        random.shuffle(order)
        count = 0
        for e in order:
            ok = True
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1
        return selected, face_count, count

    def greedy_fill(selected, face_count):
        """Repeatedly add any addable edge until none remain. Returns new count."""
        count = sum(1 for x in selected if x)
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if selected[e]:
                    continue
                ok = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        ok = False
                        break
                if ok:
                    selected[e] = True
                    count += 1
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    changed = True
        return count

    def is_addable(e, face_count):
        for f in edge_faces[e]:
            if face_count[f] >= 3:
                return False
        return True

    def add_edge(e, selected, face_count):
        selected[e] = True
        for f in edge_faces[e]:
            face_count[f] += 1

    def remove_edge(e, selected, face_count):
        selected[e] = False
        for f in edge_faces[e]:
            face_count[f] -= 1

    def count_saturation(face_count):
        return sum(1 for c in face_count if c >= 3)

    # ---- Main search ----
    best_selected = None
    best_face_count = None
    best_count = 0

    def consider(selected, face_count, count):
        nonlocal best_selected, best_face_count, best_count
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_face_count = face_count[:]

    # Phase A: many random greedy starts + greedy fill
    t0 = time.time()
    while time.time() - t0 < 3.0:
        s, fc, c = random_greedy()
        c = greedy_fill(s, fc)
        consider(s, fc, c)

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count

    # Phase B: local search — add (damage-aware) / hitting-set remove / swap
    # We maintain best_found globally.
    neutral_streak = 0
    max_neutral = 2000

    while time.time() < deadline - 5:
        # --- Try to add an edge, preferring the one with least new saturation ---
        addable = []
        for e in range(num_edges):
            if selected[e]:
                continue
            if is_addable(e, face_count):
                # damage = number of faces that go from 2->3 (become saturated)
                dmg = 0
                for f in edge_faces[e]:
                    if face_count[f] == 2:
                        dmg += 1
                addable.append((dmg, e))
        if addable:
            # Pick lowest damage; random tie-break
            addable.sort(key=lambda x: (x[0], random.random()))
            e = addable[0][1]
            add_edge(e, selected, face_count)
            count += 1
            neutral_streak = 0
            consider(selected, face_count, count)
            continue

        # --- No addable edge: try a hitting-set removal to create room ---
        # Find the edge whose removal reduces saturation the most.
        # For each selected edge e, compute how many of its faces are currently saturated.
        best_r = -1
        best_gain = 0
        # Sample a subset of selected edges to keep it fast
        sel_edges = [e for e in range(num_edges) if selected[e]]
        # If too many, sample
        if len(sel_edges) > 64:
            sample = random.sample(sel_edges, 64)
        else:
            sample = sel_edges
        for e in sample:
            gain = 0
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = e

        if best_r == -1 or best_gain == 0:
            # No beneficial removal found via sampling; do a random neutral swap
            # to perturb, or restart.
            # Random neutral swap: remove a random selected edge, add a random addable one.
            removed = random.choice(sel_edges)
            remove_edge(removed, selected, face_count)
            count -= 1
            # Now add back the best addable edge (greedy fill one step)
            filled = False
            for e in range(num_edges):
                if selected[e]:
                    continue
                if is_addable(e, face_count):
                    add_edge(e, selected, face_count)
                    count += 1
                    filled = True
                    break
            if not filled:
                # restore to keep size
                add_edge(removed, selected, face_count)
                count += 1
            neutral_streak += 1
            consider(selected, face_count, count)
            if neutral_streak > max_neutral:
                # restart from best
                selected = best_selected[:]
                face_count = best_face_count[:]
                count = best_count
                neutral_streak = 0
            continue

        # Perform the hitting removal, then greedy-fill to recover
        remove_edge(best_r, selected, face_count)
        count -= 1
        new_count = greedy_fill(selected, face_count)
        count = new_count
        if count >= best_count:
            neutral_streak = 0
        else:
            neutral_streak += 1
        consider(selected, face_count, count)
        if neutral_streak > max_neutral:
            selected = best_selected[:]
            face_count = best_face_count[:]
            count = best_count
            neutral_streak = 0

    # Phase C: final polish — a few more greedy-fill passes and targeted swaps
    if time.time() < deadline - 2:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # add any addable
            progress = False
            for e in range(num_edges):
                if selected[e]:
                    continue
                if is_addable(e, face_count):
                    add_edge(e, selected, face_count)
                    count += 1
                    progress = True
            consider(selected, face_count, count)
            if not progress:
                break

    # ---- Build result from best found ----
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append([edges[i][0], edges[i][1]])
    return result
