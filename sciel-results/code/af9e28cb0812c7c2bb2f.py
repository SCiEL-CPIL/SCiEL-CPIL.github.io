import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2  # 448

    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # Precompute faces: each face is a tuple of 4 edge indices
    faces_idx = []
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)

                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))

                face_edges = [e1, e2, e3, e4]
                idxs = tuple(edge_to_index[e] for e in face_edges)
                faces_idx.append(idxs)

    # Map edge index to list of face indices it belongs to
    edge_to_faces_idx = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            edge_to_faces_idx[e_idx].append(f_idx)

    # For fast validity checking: for each face, track count of selected edges
    face_count = [0] * len(faces_idx)
    selected = bytearray(total_edges)  # 0 or 1

    def is_valid_addition(e_idx):
        """Check if adding edge e_idx keeps all faces C4-free (no face has 4 selected)."""
        for f_idx in edge_to_faces_idx[e_idx]:
            if face_count[f_idx] == 3:
                return False
        return True

    def add_edge(e_idx):
        selected[e_idx] = 1
        for f_idx in edge_to_faces_idx[e_idx]:
            face_count[f_idx] += 1

    def remove_edge(e_idx):
        selected[e_idx] = 0
        for f_idx in edge_to_faces_idx[e_idx]:
            face_count[f_idx] -= 1

    def count_selected():
        return sum(selected)

    def greedy_random():
        """Random greedy: shuffle edges, add if valid."""
        nonlocal face_count, selected
        selected = bytearray(total_edges)
        face_count = [0] * len(faces_idx)
        
        indices = list(range(total_edges))
        random.shuffle(indices)
        count = 0
        for e_idx in indices:
            if is_valid_addition(e_idx):
                add_edge(e_idx)
                count += 1
        return count

    def local_search(time_limit):
        """
        Local search on the current selection using the face_count array.
        Tries:
          1. Add any valid missing edge
          2. 1-for-1 swaps (remove b, add e) to escape local optima
          3. 2-for-1 swaps (remove b, add e1 and e2) to increase size
          4. Perturbation (remove random edge, greedy repair)
        """
        start_time = time.time()
        
        def try_add_all():
            added = False
            for e_idx in range(total_edges):
                if not selected[e_idx] and is_valid_addition(e_idx):
                    add_edge(e_idx)
                    added = True
            return added

        # Initial: add all possible
        try_add_all()

        while time.time() - start_time < time_limit:
            improved = False

            # 1. Try to add any missing edge
            if try_add_all():
                improved = True
                continue

            # 2. Compute unblockers for missing edges
            # For each missing edge e, find which selected edges, if removed, would allow e to be added
            missing = [i for i in range(total_edges) if not selected[i]]
            if not missing:
                break

            # single_unblockers[e] = set of b in selected such that removing b makes e addable
            single_unblockers = {}
            for e in missing:
                bad_faces = []
                for f_idx in edge_to_faces_idx[e]:
                    if face_count[f_idx] == 3:
                        # Find which 3 edges in this face are selected (excluding e)
                        face = faces_idx[f_idx]
                        others = [idx for idx in face if idx != e and selected[idx]]
                        bad_faces.append(others)
                
                if not bad_faces:
                    continue  # Should have been added

                # The set of edges that must be removed to unblock e is the intersection of all bad face "others"
                if len(bad_faces) == 1:
                    common = set(bad_faces[0])
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    if not common:
                        continue
                single_unblockers[e] = common

            # 3. Try 2-for-1 swaps
            # Build map: b -> list of edges it unblocks
            unblocks = {}
            for e, blockers in single_unblockers.items():
                for b in blockers:
                    if selected[b]:
                        if b not in unblocks:
                            unblocks[b] = []
                        unblocks[b].append(e)

            # Find a b that unblocks >= 2 edges, try the swap
            found_swap = False
            b_list = [b for b in range(total_edges) if selected[b]]
            random.shuffle(b_list)

            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                candidates = unblocks.get(b, [])
                if len(candidates) < 2:
                    continue

                # Try pairs from candidates (limit to keep fast)
                n_try = min(len(candidates), 20)
                if n_try >= 2:
                    sample = random.sample(candidates, n_try)
                    for i in range(len(sample)):
                        for j in range(i + 1, len(sample)):
                            e1, e2 = sample[i], sample[j]
                            # Temporarily remove b
                            remove_edge(b)
                            ok1 = is_valid_addition(e1)
                            if ok1:
                                add_edge(e1)
                                ok2 = is_valid_addition(e2)
                                if ok2:
                                    add_edge(e2)
                                    # Keep the swap
                                    found_swap = True
                                    break
                                else:
                                    remove_edge(e1)
                            else:
                                pass
                            # If we didn't keep, restore b
                            if not found_swap:
                                add_edge(b)
                        if found_swap:
                            break
                if found_swap:
                    break

            if found_swap:
                improved = True
                try_add_all()
                continue

            # 4. Try 1-for-1 swaps
            did_1for1 = False
            if single_unblockers:
                e_candidates = list(single_unblockers.keys())
                random.shuffle(e_candidates)
                for e in e_candidates[:50]:  # limit work
                    if time.time() - start_time > time_limit:
                        break
                    blockers = single_unblockers[e]
                    if not blockers:
                        continue
                    b = random.choice(list(blockers))
                    if not selected[b]:
                        continue
                    # Perform swap
                    remove_edge(b)
                    if is_valid_addition(e):
                        add_edge(e)
                        did_1for1 = True
                        # Try to add more after swap
                        try_add_all()
                        break
                    else:
                        add_edge(b)  # restore
                if did_1for1:
                    improved = True
                    continue

            # 5. Perturbation: remove a random selected edge, then greedy repair
            sel_list = [i for i in range(total_edges) if selected[i]]
            if sel_list:
                b = random.choice(sel_list)
                remove_edge(b)
                try_add_all()

        return count_selected()

    best_count = 0
    best_selected = bytearray(total_edges)
    best_face_count = [0] * len(faces_idx)

    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    # Phase 1: Multiple random greedy restarts
    while time.time() < deadline - 30:  # Leave 30s for local search
        greedy_random()
        cur_count = count_selected()
        if cur_count > best_count:
            best_count = cur_count
            best_selected = selected[:]
            best_face_count = face_count[:]

    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5:
        # Restore best state
        selected = best_selected[:]
        face_count = best_face_count[:]
        
        cur_count = local_search(deadline - time.time() - 5)
        if cur_count > best_count:
            best_count = cur_count
            best_selected = selected[:]
            best_face_count = face_count[:]

    # Phase 3: If time remains, do another round of restarts + local search
    if time.time() < deadline - 15:
        while time.time() < deadline - 10:
            greedy_random()
            cur_count = count_selected()
            if cur_count > best_count:
                best_count = cur_count
                best_selected = selected[:]
                best_face_count = face_count[:]
        
        if time.time() < deadline - 5:
            selected = best_selected[:]
            face_count = best_face_count[:]
            cur_count = local_search(deadline - time.time() - 5)
            if cur_count > best_count:
                best_count = cur_count
                best_selected = selected[:]
                best_face_count = face_count[:]

    # Build result
    result = []
    for e_idx in range(total_edges):
        if best_selected[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])

    return result
