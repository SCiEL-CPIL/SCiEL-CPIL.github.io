"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_edge_data():
    """Precompute all edges and face constraints for Q7."""
    d = 7
    n = 1 << d  # 128
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_index[(x, y)] = idx
    E = len(edges)  # 448

    # Build faces: C(7,2) * 2^5 = 21 * 32 = 672 faces
    face_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_ij
                    e1 = edge_index[(min(v0, v1), max(v0, v1))]
                    e2 = edge_index[(min(v0, v2), max(v0, v2))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v3), max(v2, v3))]
                    face_edges.append([e1, e2, e3, e4])
    F = len(face_edges)  # 672

    # For each edge, list of faces it belongs to
    edge_to_faces = [[] for _ in range(E)]
    for fid, fedges in enumerate(face_edges):
        for eidx in fedges:
            edge_to_faces[eidx].append(fid)

    return edges, E, face_edges, F, edge_to_faces, d, n


def _complementary_deletion(face_edges, F, edge_to_faces, E, rng, deadline):
    """Start with all edges, remove edges to break all C4s.
    Greedy: while there's a face with all 4 edges, remove one of its edges.
    Use smart choice: remove the edge that appears in the most full faces.
    """
    selected = np.ones(E, dtype=np.int8)
    face_count = np.zeros(F, dtype=np.int8)
    for fid in range(F):
        face_count[fid] = 4

    count = E

    # Build list of full faces
    full_faces = list(range(F))
    rng.shuffle(full_faces)

    # Process full faces iteratively
    while full_faces:
        if time.time() > deadline:
            break

        # Rebuild: keep only truly full faces
        full_faces = [f for f in full_faces if face_count[f] == 4]
        if not full_faces:
            break

        # For each full face, count how many full faces each of its edges is in
        edge_full_count = {}
        for fid in full_faces:
            for eidx in face_edges[fid]:
                if selected[eidx]:
                    edge_full_count[eidx] = edge_full_count.get(eidx, 0) + 1

        if not edge_full_count:
            break

        # Pick the edge with max count, break ties randomly
        max_count = max(edge_full_count.values())
        candidates = [e for e, c in edge_full_count.items() if c == max_count]
        rem_eidx = rng.choice(candidates)

        # Remove this edge
        selected[rem_eidx] = 0
        count -= 1
        for fid in edge_to_faces[rem_eidx]:
            face_count[fid] -= 1

    return selected, face_count, count


def _try_add_edges(selected, face_count, count, E, edge_to_faces, rng, deadline):
    """Try to add back unselected edges that don't complete any face."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for eidx in range(E):
            if selected[eidx]:
                continue
            can_add = True
            for fid in edge_to_faces[eidx]:
                if face_count[fid] == 3:
                    can_add = False
                    break
            if can_add:
                selected[eidx] = 1
                count += 1
                for fid in edge_to_faces[eidx]:
                    face_count[fid] += 1
                improved = True
    return selected, face_count, count


def _local_search_swap(selected, face_count, count, E, edge_to_faces, rng, deadline):
    """Local search: remove one edge, try to add multiple others."""
    start_time = time.time()
    improved = True

    while improved and (time.time() - start_time) < (deadline - start_time):
        improved = False
        selected_list = [i for i in range(E) if selected[i] == 1]
        rng.shuffle(selected_list)

        for rem_eidx in selected_list:
            if time.time() > deadline:
                break
            if not selected[rem_eidx]:
                continue

            # Temporarily remove
            selected[rem_eidx] = 0
            count -= 1
            for fid in edge_to_faces[rem_eidx]:
                face_count[fid] -= 1

            # Try to add as many unselected edges as possible
            added = 0
            new_adds = []
            for eidx in range(E):
                if selected[eidx]:
                    continue
                can_add = True
                for fid in edge_to_faces[eidx]:
                    if face_count[fid] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[eidx] = 1
                    count += 1
                    added += 1
                    new_adds.append(eidx)
                    for fid in edge_to_faces[eidx]:
                        face_count[fid] += 1

            if added <= 1:
                # Revert
                for eidx in new_adds:
                    selected[eidx] = 0
                    count -= 1
                    for fid in edge_to_faces[eidx]:
                        face_count[fid] -= 1
                selected[rem_eidx] = 1
                count += 1
                for fid in edge_to_faces[rem_eidx]:
                    face_count[fid] += 1
            else:
                improved = True

    return selected, face_count, count


def _multi_removal_search(selected, face_count, count, E, edge_to_faces, rng, deadline):
    """Try removing 2-4 edges and adding back more."""
    start_time = time.time()

    while time.time() < deadline:
        selected_list = [i for i in range(E) if selected[i] == 1]
        if len(selected_list) < 2:
            break
        rng.shuffle(selected_list)

        num_remove = rng.randint(2, 4)
        removed = selected_list[:num_remove]

        # Temporarily remove all
        for e in removed:
            selected[e] = 0
            count -= 1
            for fid in edge_to_faces[e]:
                face_count[fid] -= 1

        # Try to add edges
        added = 0
        new_adds = []
        for eidx in range(E):
            if selected[eidx]:
                continue
            can_add = True
            for fid in edge_to_faces[eidx]:
                if face_count[fid] == 3:
                    can_add = False
                    break
            if can_add:
                selected[eidx] = 1
                count += 1
                added += 1
                new_adds.append(eidx)
                for fid in edge_to_faces[eidx]:
                    face_count[fid] += 1

        if added <= num_remove:
            # Revert
            for eidx in new_adds:
                selected[eidx] = 0
                count -= 1
                for fid in edge_to_faces[eidx]:
                    face_count[fid] -= 1
            for e in removed:
                selected[e] = 1
                count += 1
                for fid in edge_to_faces[e]:
                    face_count[fid] += 1
        else:
            # Accept improvement
            return selected, face_count, count

    return selected, face_count, count


def _greedy_add(selected, face_count, E, edge_to_faces, rng, deadline):
    """Add any edge that fits, in random order."""
    unselected = [e for e in range(E) if not selected[e]]
    rng.shuffle(unselected)
    added = 0
    for eidx in unselected:
        if time.time() > deadline:
            break
        if selected[eidx]:
            continue
        can_add = True
        for fid in edge_to_faces[eidx]:
            if face_count[fid] == 3:
                can_add = False
                break
        if can_add:
            selected[eidx] = 1
            added += 1
            for fid in edge_to_faces[eidx]:
                face_count[fid] += 1
    return added


def _is_valid(selected, face_edges, F):
    """Verify no face has all 4 edges selected."""
    for fid in range(F):
        cnt = 0
        for e in face_edges[fid]:
            if selected[e]:
                cnt += 1
        if cnt == 4:
            return False
    return True


def _pair_removal_search(selected, face_count, count, E, edge_to_faces, rng, deadline):
    """Try removing 2 edges at a time and adding back more."""
    start_time = time.time()
    improved = True

    while improved and (time.time() - start_time) < (deadline - start_time):
        improved = False
        selected_list = [i for i in range(E) if selected[i] == 1]
        if len(selected_list) < 2:
            break
        rng.shuffle(selected_list)

        # Try pairs from the first 30 selected edges
        for i in range(min(30, len(selected_list))):
            if time.time() > deadline:
                break
            e1 = selected_list[i]
            if not selected[e1]:
                continue
            for j in range(i + 1, min(i + 15, len(selected_list))):
                e2 = selected_list[j]
                if not selected[e2]:
                    continue

                # Temporarily remove both
                selected[e1] = 0
                selected[e2] = 0
                count -= 2
                for fid in edge_to_faces[e1]:
                    face_count[fid] -= 1
                for fid in edge_to_faces[e2]:
                    face_count[fid] -= 1

                # Try to add edges
                added = 0
                new_adds = []
                for eidx in range(E):
                    if selected[eidx]:
                        continue
                    can_add = True
                    for fid in edge_to_faces[eidx]:
                        if face_count[fid] == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[eidx] = 1
                        count += 1
                        added += 1
                        new_adds.append(eidx)
                        for fid in edge_to_faces[eidx]:
                            face_count[fid] += 1

                if added > 2:
                    improved = True
                    break
                else:
                    # Revert
                    for eidx in new_adds:
                        selected[eidx] = 0
                        count -= 1
                        for fid in edge_to_faces[eidx]:
                            face_count[fid] -= 1
                    selected[e1] = 1
                    selected[e2] = 1
                    count += 2
                    for fid in edge_to_faces[e1]:
                        face_count[fid] += 1
                    for fid in edge_to_faces[e2]:
                        face_count[fid] += 1

            if improved:
                break

    return selected, face_count, count


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    edges, E, face_edges, F, edge_to_faces, d, n = _build_edge_data()

    rng = random.Random(42)

    best_count = 0
    best_selected = None
    best_face_count = None

    # Phase 1: Complementary deletion (start dense, remove to break C4s)
    for restart in range(15):
        if time.time() > deadline - 35:
            break
        sel, fc, cnt = _complementary_deletion(face_edges, F, edge_to_faces, E, rng, deadline - 35)
        # Try to add back edges
        sel, fc, cnt = _try_add_edges(sel, fc, cnt, E, edge_to_faces, rng, deadline - 25)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
            best_face_count = fc.copy()

    # Phase 2: Greedy construction (start empty, add edges)
    for restart in range(40):
        if time.time() > deadline - 30:
            break
        sel = np.zeros(E, dtype=np.int8)
        fc = np.zeros(F, dtype=np.int8)
        order = list(range(E))
        rng.shuffle(order)
        for e in order:
            if time.time() > deadline - 25:
                break
            can_add = True
            for fid in edge_to_faces[e]:
                if fc[fid] == 3:
                    can_add = False
                    break
            if can_add:
                sel[e] = 1
                for fid in edge_to_faces[e]:
                    fc[fid] += 1
        cnt = int(sel.sum())
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
            best_face_count = fc.copy()

    # Phase 3: Local search with 1-edge removal + greedy repair
    if best_selected is not None:
        sel = best_selected.copy()
        fc = best_face_count.copy()
        cnt = best_count

        for rnd in range(200):
            if time.time() > deadline - 15:
                break

            # Try removing 1 edge and adding back
            selected_list = [i for i in range(E) if sel[i] == 1]
            rng.shuffle(selected_list)
            improved = False

            for rem_e in selected_list[:25]:
                if time.time() > deadline - 12:
                    break
                # Remove
                sel[rem_e] = 0
                cnt -= 1
                for fid in edge_to_faces[rem_e]:
                    fc[fid] -= 1

                # Try to add
                added = 0
                new_adds = []
                for eidx in range(E):
                    if sel[eidx]:
                        continue
                    can_add = True
                    for fid in edge_to_faces[eidx]:
                        if fc[fid] == 3:
                            can_add = False
                            break
                    if can_add:
                        sel[eidx] = 1
                        cnt += 1
                        added += 1
                        new_adds.append(eidx)
                        for fid in edge_to_faces[eidx]:
                            fc[fid] += 1

                if added > 1:
                    improved = True
                    break
                else:
                    # Revert
                    for eidx in new_adds:
                        sel[eidx] = 0
                        cnt -= 1
                        for fid in edge_to_faces[eidx]:
                            fc[fid] -= 1
                    sel[rem_e] = 1
                    cnt += 1
                    for fid in edge_to_faces[rem_e]:
                        fc[fid] += 1

            if cnt > best_count:
                best_count = cnt
                best_selected = sel.copy()
                best_face_count = fc.copy()

            if not improved:
                # Try 2-edge removal
                if len(selected_list) >= 2:
                    pairs = [(selected_list[i], selected_list[j])
                             for i in range(min(12, len(selected_list)))
                             for j in range(i + 1, min(12, len(selected_list)))]
                    rng.shuffle(pairs)
                    for e1, e2 in pairs[:15]:
                        if time.time() > deadline - 12:
                            break
                        sel[e1] = 0
                        sel[e2] = 0
                        cnt -= 2
                        for fid in edge_to_faces[e1]:
                            fc[fid] -= 1
                        for fid in edge_to_faces[e2]:
                            fc[fid] -= 1

                        added = 0
                        new_adds = []
                        for eidx in range(E):
                            if sel[eidx]:
                                continue
                            can_add = True
                            for fid in edge_to_faces[eidx]:
                                if fc[fid] == 3:
                                    can_add = False
                                    break
                            if can_add:
                                sel[eidx] = 1
                                cnt += 1
                                added += 1
                                new_adds.append(eidx)
                                for fid in edge_to_faces[eidx]:
                                    fc[fid] += 1

                        if added > 2:
                            improved = True
                            break
                        else:
                            for eidx in new_adds:
                                sel[eidx] = 0
                                cnt -= 1
                                for fid in edge_to_faces[eidx]:
                                    fc[fid] -= 1
                            sel[e1] = 1
                            sel[e2] = 1
                            cnt += 2
                            for fid in edge_to_faces[e1]:
                                fc[fid] += 1
                            for fid in edge_to_faces[e2]:
                                fc[fid] += 1

                    if cnt > best_count:
                        best_count = cnt
                        best_selected = sel.copy()
                        best_face_count = fc.copy()

    # Phase 4: Multi-edge removal search (3-5 edges)
    if best_selected is not None and time.time() < deadline - 10:
        sel = best_selected.copy()
        fc = best_face_count.copy()
        cnt = best_count

        for attempt in range(100):
            if time.time() > deadline - 8:
                break

            selected_list = [i for i in range(E) if sel[i] == 1]
            rng.shuffle(selected_list)
            num_remove = rng.randint(3, 5)
            removed = selected_list[:num_remove]

            for e in removed:
                sel[e] = 0
                cnt -= 1
                for fid in edge_to_faces[e]:
                    fc[fid] -= 1

            added = 0
            new_adds = []
            for eidx in range(E):
                if sel[eidx]:
                    continue
                can_add = True
                for fid in edge_to_faces[eidx]:
                    if fc[fid] == 3:
                        can_add = False
                        break
                if can_add:
                    sel[eidx] = 1
                    cnt += 1
                    added += 1
                    new_adds.append(eidx)
                    for fid in edge_to_faces[eidx]:
                        fc[fid] += 1

            if added > num_remove:
                if cnt > best_count:
                    best_count = cnt
                    best_selected = sel.copy()
                    best_face_count = fc.copy()
            else:
                for eidx in new_adds:
                    sel[eidx] = 0
                    cnt -= 1
                    for fid in edge_to_faces[eidx]:
                        fc[fid] -= 1
                for e in removed:
                    sel[e] = 1
                    cnt += 1
                    for fid in edge_to_faces[e]:
                        fc[fid] += 1

    # Phase 5: Final polish - single edge removals
    if best_selected is not None and time.time() < deadline - 5:
        sel = best_selected.copy()
        fc = best_face_count.copy()
        cnt = best_count

        for rnd in range(50):
            if time.time() > deadline - 4:
                break
            selected_list = [i for i in range(E) if sel[i] == 1]
            rng.shuffle(selected_list)
            improved = False
            for rem_e in selected_list[:20]:
                if time.time() > deadline - 4:
                    break
                sel[rem_e] = 0
                cnt -= 1
                for fid in edge_to_faces[rem_e]:
                    fc[fid] -= 1

                added = 0
                new_adds = []
                for eidx in range(E):
                    if sel[eidx]:
                        continue
                    can_add = True
                    for fid in edge_to_faces[eidx]:
                        if fc[fid] == 3:
                            can_add = False
                            break
                    if can_add:
                        sel[eidx] = 1
                        cnt += 1
                        added += 1
                        new_adds.append(eidx)
                        for fid in edge_to_faces[eidx]:
                            fc[fid] += 1

                if added > 1:
                    improved = True
                    break
                else:
                    for eidx in new_adds:
                        sel[eidx] = 0
                        cnt -= 1
                        for fid in edge_to_faces[eidx]:
                            fc[fid] -= 1
                    sel[rem_e] = 1
                    cnt += 1
                    for fid in edge_to_faces[rem_e]:
                        fc[fid] += 1

            if cnt > best_count:
                best_count = cnt
                best_selected = sel.copy()
                best_face_count = fc.copy()
            if not improved:
                break

    # Final verification
    final_sel = best_selected
    if not _is_valid(final_sel, face_edges, F):
        # Fallback: rebuild
        rebuild = np.zeros(E, dtype=np.int8)
        rfc = np.zeros(F, dtype=np.int8)
        for e in range(E):
            can_add = True
            for fid in edge_to_faces[e]:
                if rfc[fid] == 3:
                    can_add = False
                    break
            if can_add:
                rebuild[e] = 1
                for fid in edge_to_faces[e]:
                    rfc[fid] += 1
        final_sel = rebuild
        best_count = int(rebuild.sum())

    out = []
    for idx in range(E):
        if final_sel[idx]:
            out.append(list(edges[idx][:2]))

    return out

# EVOLVE-BLOCK-END
