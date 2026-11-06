import random
import time
import math

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

    # Precompute faces
    faces = []
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
                faces.append(face_edges)

    num_faces = len(faces)

    # Store faces as tuples of edge indices
    faces_idx = []
    for f in faces:
        idxs = tuple(edge_to_index[e] for e in f)
        faces_idx.append(idxs)

    # Map edge index to list of face indices
    edge_to_faces_idx = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            edge_to_faces_idx[e_idx].append(f_idx)

    # Helper: fast validity check using face_counts array
    def is_valid_addition(face_counts, e_idx):
        for f_idx in edge_to_faces_idx[e_idx]:
            if face_counts[f_idx] == 3:
                return False
        return True

    def greedy_initialization():
        chosen = set()
        face_counts = [0] * num_faces
        indices = list(range(total_edges))
        random.shuffle(indices)

        for e_idx in indices:
            if is_valid_addition(face_counts, e_idx):
                chosen.add(e_idx)
                for f_idx in edge_to_faces_idx[e_idx]:
                    face_counts[f_idx] += 1

        return chosen, face_counts

    def try_add_all(chosen, face_counts):
        missing = [i for i in range(total_edges) if i not in chosen]
        random.shuffle(missing)
        added = 0
        for e_idx in missing:
            if is_valid_addition(face_counts, e_idx):
                chosen.add(e_idx)
                for f_idx in edge_to_faces_idx[e_idx]:
                    face_counts[f_idx] += 1
                added += 1
        return added

    def local_search_with_swaps(chosen, face_counts, time_limit):
        """
        Local search with 2-for-1 swaps and 1-for-1 perturbations.
        """
        start_time = time.time()
        current_size = len(chosen)

        while time.time() - start_time < time_limit:
            # First, try to add any missing edge
            added = try_add_all(chosen, face_counts)
            current_size = len(chosen)

            if added > 0:
                continue

            # Compute single unblockers for missing edges
            missing_edges = [i for i in range(total_edges) if i not in chosen]
            if not missing_edges:
                break

            single_unblockers = {}
            for e in missing_edges:
                bad_faces = []
                for f_idx in edge_to_faces_idx[e]:
                    if face_counts[f_idx] == 3:
                        # Find which 3 edges are in chosen
                        others = [idx for idx in faces_idx[f_idx] if idx != e and idx in chosen]
                        bad_faces.append(others)

                if not bad_faces:
                    continue

                if len(bad_faces) == 1:
                    single_unblockers[e] = set(bad_faces[0])
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    if common:
                        single_unblockers[e] = common
                    else:
                        single_unblockers[e] = set()

            # Map b -> list of e's it unblocks
            unblocks = {b: [] for b in chosen}
            for e, blockers in single_unblockers.items():
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)

            # Try 2-for-1 swaps
            found_swap = False
            b_list = list(chosen)
            random.shuffle(b_list)

            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                if b not in chosen:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try random pairs
                    pairs_to_try = random.sample(candidates, min(10, len(candidates)))
                    if len(pairs_to_try) < 2:
                        continue
                    for i in range(len(pairs_to_try)):
                        for j in range(i + 1, len(pairs_to_try)):
                            e1, e2 = pairs_to_try[i], pairs_to_try[j]
                            # Check validity by temporarily removing b
                            # Save state
                            saved_counts = face_counts[:]
                            chosen.discard(b)
                            for f_idx in edge_to_faces_idx[b]:
                                face_counts[f_idx] -= 1

                            # Try adding e1
                            if is_valid_addition(face_counts, e1):
                                chosen.add(e1)
                                for f_idx in edge_to_faces_idx[e1]:
                                    face_counts[f_idx] += 1
                                # Try adding e2
                                if is_valid_addition(face_counts, e2):
                                    chosen.add(e2)
                                    for f_idx in edge_to_faces_idx[e2]:
                                        face_counts[f_idx] += 1
                                    current_size += 1
                                    found_swap = True
                                    break
                                else:
                                    # Revert e1
                                    chosen.discard(e1)
                                    for f_idx in edge_to_faces_idx[e1]:
                                        face_counts[f_idx] -= 1
                            else:
                                # Revert b removal
                                pass

                            # Revert b removal
                            chosen.add(b)
                            for f_idx in edge_to_faces_idx[b]:
                                face_counts[f_idx] += 1

                            if found_swap:
                                break
                        if found_swap:
                            break

                if found_swap:
                    break

            if found_swap:
                # Recalculate and continue
                try_add_all(chosen, face_counts)
                continue

            # If no 2-for-1 found, try perturbation
            # 1-for-1 swap or kick
            if random.random() < 0.3:
                # 1-for-1
                if missing_edges:
                    e = random.choice(missing_edges)
                    blockers = single_unblockers.get(e, set())
                    if blockers:
                        b = random.choice(list(blockers))
                        # Swap: remove b, add e
                        chosen.discard(b)
                        for f_idx in edge_to_faces_idx[b]:
                            face_counts[f_idx] -= 1
                        chosen.add(e)
                        for f_idx in edge_to_faces_idx[e]:
                            face_counts[f_idx] += 1
            else:
                # Kick: remove random edges, then repair
                kick_size = random.randint(3, 10)
                removal_candidates = list(chosen)
                random.shuffle(removal_candidates)
                to_remove = set(removal_candidates[:kick_size])

                for e_idx in to_remove:
                    chosen.discard(e_idx)
                    for f_idx in edge_to_faces_idx[e_idx]:
                        face_counts[f_idx] -= 1

                try_add_all(chosen, face_counts)

        return chosen, face_counts

    def simulated_annealing(deadline):
        start_time = time.time()

        # Initial state
        current_chosen, current_counts = greedy_initialization()
        current_size = len(current_chosen)

        best_chosen = set(current_chosen)
        best_size = current_size

        # Temperature schedule
        T = 3.0
        cooling_rate = 0.9997
        min_T = 0.1

        # Kick size range
        min_kick = 4
        max_kick = 20

        # Phase 1: Simulated Annealing
        while time.time() < deadline - 20:
            # 1. Try to fill up
            try_add_all(current_chosen, current_counts)
            current_size = len(current_chosen)

            if current_size > best_size:
                best_size = current_size
                best_chosen = set(current_chosen)

            # 2. Generate neighbor via Kick
            kick_size = random.randint(min_kick, max_kick)
            if kick_size > current_size:
                kick_size = max(1, current_size // 3)

            removal_candidates = list(current_chosen)
            random.shuffle(removal_candidates)
            to_remove = set(removal_candidates[:kick_size])

            # Apply removal
            next_chosen = set(current_chosen)
            next_counts = current_counts[:]

            for e_idx in to_remove:
                next_chosen.discard(e_idx)
                for f_idx in edge_to_faces_idx[e_idx]:
                    next_counts[f_idx] -= 1

            # Greedy Repair
            try_add_all(next_chosen, next_counts)
            next_size = len(next_chosen)

            # 3. Acceptance criterion
            delta = next_size - current_size
            if delta >= 0:
                current_chosen = next_chosen
                current_counts = next_counts
                current_size = next_size
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_chosen = next_chosen
                    current_counts = next_counts
                    current_size = next_size

            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T

        # Phase 2: Local search on the best set found
        if time.time() < deadline - 2:
            ls_chosen, ls_counts = local_search_with_swaps(
                set(best_chosen),
                [0] * num_faces,
                deadline - time.time() - 2
            )
            # Rebuild face_counts for ls_chosen
            ls_counts = [0] * num_faces
            for e_idx in ls_chosen:
                for f_idx in edge_to_faces_idx[e_idx]:
                    ls_counts[f_idx] += 1
            try_add_all(ls_chosen, ls_counts)

            if len(ls_chosen) > best_size:
                best_chosen = ls_chosen
                best_size = len(best_chosen)

        # Phase 3: Another round of local search on the overall best
        if time.time() < deadline - 1:
            ls2_chosen, ls2_counts = local_search_with_swaps(
                set(best_chosen),
                [0] * num_faces,
                deadline - time.time() - 1
            )
            ls2_counts = [0] * num_faces
            for e_idx in ls2_chosen:
                for f_idx in edge_to_faces_idx[e_idx]:
                    ls2_counts[f_idx] += 1
            try_add_all(ls2_chosen, ls2_counts)

            if len(ls2_chosen) > best_size:
                best_chosen = ls2_chosen
                best_size = len(best_chosen)

        return best_chosen

    # Main execution
    start = time.time()
    deadline = start + 115  # 35s safety margin

    # Run multiple SA restarts if time permits
    best_set = set()
    best_size = 0

    # First SA run
    sa_result = simulated_annealing(deadline)
    best_set = sa_result
    best_size = len(best_set)

    # If time remains, do another quick SA restart
    if time.time() < deadline - 5:
        # Quick restart with different seed
        quick_deadline = min(deadline - 2, time.time() + 30)
        sa_result2 = simulated_annealing(quick_deadline)
        if len(sa_result2) > best_size:
            best_set = sa_result2
            best_size = len(best_set)

    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])

    return result
