import random
import time
import sys

def construct():
    d = 7
    n_v = 1 << d  # 128
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build edges: 448 edges
    edges = []  # (u, v, dim)
    edge_idx = {}
    for x in range(n_v):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_idx[(x, y)] = idx

    n_edges = len(edges)  # 448

    # Build faces: 672 faces, each with 4 edge indices
    faces = []
    edge_to_faces = [[] for _ in range(n_edges)]

    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << (d - 2)):
                base = 0
                bit_pos = 0
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if mask & (1 << bit_pos):
                        base |= (1 << k)
                    bit_pos += 1
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                e01 = edge_idx[(min(v0, v1), max(v0, v1))]
                e02 = edge_idx[(min(v0, v2), max(v0, v2))]
                e13 = edge_idx[(min(v1, v3), max(v1, v3))]
                e23 = edge_idx[(min(v2, v3), max(v2, v3))]
                face_edges = [e01, e02, e13, e23]
                fid = len(faces)
                faces.append(face_edges)
                for e in face_edges:
                    edge_to_faces[e].append(fid)

    n_faces = len(faces)  # 672

    # Precompute for fast face violation checking
    # For each edge, the list of faces it belongs to is already in edge_to_faces

    def count_violations(selected):
        """Count faces with all 4 edges selected."""
        fc = [0] * n_faces
        for e in range(n_edges):
            if selected[e]:
                for f in edge_to_faces[e]:
                    fc[f] += 1
        v = 0
        for f in range(n_faces):
            if fc[f] == 4:
                v += 1
        return v

    def is_valid(selected):
        """Check if no face has all 4 edges selected."""
        fc = [0] * n_faces
        for e in range(n_edges):
            if selected[e]:
                for f in edge_to_faces[e]:
                    fc[f] += 1
                    if fc[f] == 4:
                        return False
        return True

    def greedy_random(seed_val):
        """Greedy: add edges in random order if valid."""
        rng = random.Random(seed_val)
        order = list(range(n_edges))
        rng.shuffle(order)
        selected = [False] * n_edges
        fc = [0] * n_faces
        count = 0
        for e in order:
            ok = True
            for f in edge_to_faces[e]:
                if fc[f] >= 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_to_faces[e]:
                    fc[f] += 1
        return selected, fc, count

    def local_search(selected, fc, count, time_limit, seed_val):
        """
        Local search: try to add edges, and do swaps.
        """
        rng = random.Random(seed_val)
        non_selected = [e for e in range(n_edges) if not selected[e]]
        rng.shuffle(non_selected)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            if len(non_selected) > 0:
                rng.shuffle(non_selected)

            for idx, e_new in enumerate(non_selected):
                if time.time() >= time_limit:
                    break

                # Check if e_new can be added directly
                blocking = []
                for f in edge_to_faces[e_new]:
                    if fc[f] >= 3:
                        blocking.append(f)

                if not blocking:
                    # Add directly
                    selected[e_new] = True
                    count += 1
                    for f in edge_to_faces[e_new]:
                        fc[f] += 1
                    non_selected[idx] = non_selected[-1]
                    non_selected.pop()
                    improved = True
                    break

                # Try swaps: for each blocking face, try removing a selected edge from it
                for f in blocking:
                    if time.time() >= time_limit:
                        break
                    for e_old in faces[f]:
                        if not selected[e_old]:
                            continue
                        # Simulate: remove e_old, add e_new
                        ok = True
                        for f2 in edge_to_faces[e_new]:
                            if f2 in edge_to_faces[e_old]:
                                if fc[f2] >= 4:
                                    ok = False
                                    break
                            else:
                                if fc[f2] >= 3:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        # Perform swap
                        selected[e_old] = False
                        selected[e_new] = True
                        for f2 in edge_to_faces[e_old]:
                            fc[f2] -= 1
                        for f2 in edge_to_faces[e_new]:
                            fc[f2] += 1
                        non_selected[idx] = non_selected[-1]
                        non_selected.pop()
                        improved = True
                        break
                    if improved:
                        break

        return selected, fc, count

    def simulated_annealing(selected, fc, count, time_limit, seed_val):
        """
        Simulated annealing: allow moves that decrease count temporarily.
        """
        rng = random.Random(seed_val)
        best_count = count
        best_selected = selected[:]
        best_fc = fc[:]

        T = 50.0
        T_min = 0.01
        cooling = 0.995

        while time.time() < time_limit and T > T_min:
            move_type = rng.random()
            if move_type < 0.5:
                unselected = [e for e in range(n_edges) if not selected[e]]
                if not unselected:
                    T *= cooling
                    continue
                e = rng.choice(unselected)
                blocking = sum(1 for f in edge_to_faces[e] if fc[f] >= 3)
                if blocking == 0:
                    selected[e] = True
                    count += 1
                    for f in edge_to_faces[e]:
                        fc[f] += 1
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_fc = fc[:]
            elif move_type < 0.75:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                if not selected_edges:
                    T *= cooling
                    continue
                e = rng.choice(selected_edges)
                selected[e] = False
                count -= 1
                for f in edge_to_faces[e]:
                    fc[f] -= 1
            else:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                unselected = [e for e in range(n_edges) if not selected[e]]
                if not selected_edges or not unselected:
                    T *= cooling
                    continue
                e_out = rng.choice(selected_edges)
                e_in = rng.choice(unselected)
                if e_out == e_in:
                    T *= cooling
                    continue
                ok = True
                for f2 in edge_to_faces[e_in]:
                    if f2 in edge_to_faces[e_out]:
                        if fc[f2] >= 4:
                            ok = False
                            break
                    else:
                        if fc[f2] >= 3:
                            ok = False
                            break
                if ok:
                    selected[e_out] = False
                    selected[e_in] = True
                    for f2 in edge_to_faces[e_out]:
                        fc[f2] -= 1
                    for f2 in edge_to_faces[e_in]:
                        fc[f2] += 1
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_fc = fc[:]

            T *= cooling

        return best_selected, best_fc, best_count

    def tabu_search(selected, fc, count, time_limit, seed_val):
        """
        Tabu search: maintain a tabu list of recently moved edges to avoid cycling.
        """
        rng = random.Random(seed_val)
        best_count = count
        best_selected = selected[:]
        best_fc = fc[:]

        tabu_list = {}  # edge -> iteration when it became non-tabu
        tabu_tenure = 20
        max_iterations = 5000

        for iteration in range(max_iterations):
            if time.time() >= time_limit:
                break

            # Pick a random move
            move_type = rng.random()
            if move_type < 0.4:
                unselected = [e for e in range(n_edges) if not selected[e] and tabu_list.get(e, 0) <= iteration]
                if not unselected:
                    unselected = [e for e in range(n_edges) if not selected[e]]
                if not unselected:
                    continue
                e = rng.choice(unselected)
                blocking = sum(1 for f in edge_to_faces[e] if fc[f] >= 3)
                if blocking == 0:
                    selected[e] = True
                    count += 1
                    for f in edge_to_faces[e]:
                        fc[f] += 1
                    tabu_list[e] = iteration + tabu_tenure
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_fc = fc[:]
            elif move_type < 0.7:
                selected_edges = [e for e in range(n_edges) if selected[e] and tabu_list.get(e, 0) <= iteration]
                if not selected_edges:
                    selected_edges = [e for e in range(n_edges) if selected[e]]
                if not selected_edges:
                    continue
                e = rng.choice(selected_edges)
                selected[e] = False
                count -= 1
                for f in edge_to_faces[e]:
                    fc[f] -= 1
                tabu_list[e] = iteration + tabu_tenure
            else:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                unselected = [e for e in range(n_edges) if not selected[e]]
                if not selected_edges or not unselected:
                    continue
                e_out = rng.choice(selected_edges)
                e_in = rng.choice(unselected)
                if e_out == e_in:
                    continue
                ok = True
                for f2 in edge_to_faces[e_in]:
                    if f2 in edge_to_faces[e_out]:
                        if fc[f2] >= 4:
                            ok = False
                            break
                    else:
                        if fc[f2] >= 3:
                            ok = False
                            break
                if ok:
                    selected[e_out] = False
                    selected[e_in] = True
                    for f2 in edge_to_faces[e_out]:
                        fc[f2] -= 1
                    for f2 in edge_to_faces[e_in]:
                        fc[f2] += 1
                    tabu_list[e_out] = iteration + tabu_tenure
                    tabu_list[e_in] = iteration + tabu_tenure
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_fc = fc[:]

        return best_selected, best_fc, best_count

    # Main loop: multiple restarts with different strategies
    best_count = 0
    best_selected = None
    best_fc = None

    # Phase 1: Greedy with different seeds
    t_start = time.time()
    n_greedy = 30
    for i in range(n_greedy):
        if time.time() > deadline:
            break
        selected, fc, count = greedy_random(i * 12345 + 67890)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_fc = fc[:]

    # Phase 2: Local search on best greedy result
    if time.time() < deadline:
        remaining = deadline - time.time()
        ls_time = time.time() + remaining * 0.3
        selected, fc, count = local_search(best_selected, best_fc, best_count, ls_time, 42)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_fc = fc[:]

    # Phase 3: More local search with different seeds
    if time.time() < deadline:
        remaining = deadline - time.time()
        n_ls = min(5, int(remaining / 10))
        for i in range(n_ls):
            if time.time() > deadline:
                break
            ls_time = time.time() + min(remaining * 0.15, 15)
            selected, fc, count = local_search(best_selected, best_fc, best_count, ls_time, 1000 + i * 777)
            if count > best_count:
                best_count = count
                best_selected = selected[:]
                best_fc = fc[:]

    # Phase 4: Simulated annealing on best result
    if time.time() < deadline:
        remaining = deadline - time.time()
        sa_time = time.time() + remaining * 0.25
        selected, fc, count = simulated_annealing(best_selected, best_fc, best_count, sa_time, 999)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_fc = fc[:]

    # Phase 5: Tabu search on best result
    if time.time() < deadline:
        remaining = deadline - time.time()
        ts_time = time.time() + remaining * 0.25
        selected, fc, count = tabu_search(best_selected, best_fc, best_count, ts_time, 7777)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_fc = fc[:]

    # Phase 6: Final local search
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = time.time() + remaining * 0.5
            selected, fc, count = local_search(best_selected, best_fc, best_count, ls_time, 55555)
            if count > best_count:
                best_count = count
                best_selected = selected[:]
                best_fc = fc[:]

    # Build output
    out = []
    for e in range(n_edges):
        if best_selected[e]:
            x, y, i_dim = edges[e]
            out.append([x, y])

    return out
