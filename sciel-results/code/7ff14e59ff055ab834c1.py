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

    # Build faces: 21 * 32 = 672 faces, each with 4 edges
    face_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            for fixed in range(1 << (d - 2)):
                base = 0
                bit_pos = 0
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if fixed & (1 << bit_pos):
                        base |= (1 << k)
                    bit_pos += 1
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                e1 = edge_index[(min(v00, v10), max(v00, v10))]
                e2 = edge_index[(min(v00, v01), max(v00, v01))]
                e3 = edge_index[(min(v10, v11), max(v10, v11))]
                e4 = edge_index[(min(v01, v11), max(v01, v11))]
                face_edges.append([e1, e2, e3, e4])
    F = len(face_edges)  # 672

    # For each edge, list of faces it belongs to
    edge_to_faces = [[] for _ in range(E)]
    for fid, fedges in enumerate(face_edges):
        for eidx in fedges:
            edge_to_faces[eidx].append(fid)

    # For each face, the 4 edges (for quick checking)
    return edges, E, face_edges, F, edge_to_faces, d, n


def _complementary_deletion(face_edges, F, edge_to_faces, E, rng):
    """Start with all edges, remove edges to break all C4s.
    
    Greedy: while there's a face with all 4 edges, remove one of its edges.
    Use a smart choice: remove the edge that appears in the most full faces.
    """
    selected = np.ones(E, dtype=np.int8)
    face_count = np.zeros(F, dtype=np.int8)
    for fid in range(F):
        face_count[fid] = 4  # all 4 edges selected initially

    count = E

    # Find all full faces
    full_faces = list(range(F))
    rng.shuffle(full_faces)

    # Process full faces: for each, pick an edge to remove
    # Use a priority: remove edge that's in most full faces
    while full_faces:
        # Rebuild full_faces (some may no longer be full)
        full_faces = [f for f in full_faces if face_count[f] == 4]
        if not full_faces:
            break

        # For each full face, count how many full faces each of its edges is in
        # Pick the edge that appears in the most full faces
        edge_full_count = {}
        for fid in full_faces:
            for eidx in face_edges[fid]:
                if selected[eidx]:
                    edge_full_count[eidx] = edge_full_count.get(eidx, 0) + 1

        # Pick the edge with max count, break ties randomly
        max_count = max(edge_full_count.values())
        candidates = [e for e, c in edge_full_count.items() if c == max_count]
        rem_eidx = rng.choice(candidates)

        # Remove this edge
        selected[rem_eidx] = 0
        count -= 1
        for fid in edge_to_faces[rem_eidx]:
            face_count[fid] -= 1
            if face_count[fid] == 3 and fid in full_faces:
                full_faces.remove(fid)

    return selected, face_count, count


def _try_add_edges(selected, face_count, count, E, face_edges, F, edge_to_faces, rng):
    """Try to add back unselected edges that don't complete any face."""
    improved = True
    while improved:
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


def _local_search(selected, face_count, count, E, face_edges, F, edge_to_faces, rng, time_limit):
    """Local search with swaps: remove one edge, try to add multiple others."""
    start_time = time.time()

    # First, try to add all possible edges
    selected, face_count, count = _try_add_edges(selected, face_count, count, E, face_edges, F, edge_to_faces, rng)

    # Now try swaps: remove an edge, see if we can add 2+ others
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        selected_list = [i for i in range(E) if selected[i] == 1]
        rng.shuffle(selected_list)

        for rem_eidx in selected_list:
            if not selected[rem_eidx]:
                continue
            if time.time() > start_time + time_limit:
                break

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


def _multi_removal_search(selected, face_count, count, E, face_edges, F, edge_to_faces, rng, time_limit):
    """Try removing 2-3 edges and adding back more."""
    start_time = time.time()
    improved = True

    while improved and (time.time() - start_time) < time_limit:
        improved = False
        selected_list = [i for i in range(E) if selected[i] == 1]
        rng.shuffle(selected_list)

        # Try pairs
        for i in range(min(len(selected_list), 100)):
            if time.time() > start_time + time_limit:
                break
            rem1 = selected_list[i]
            if not selected[rem1]:
                continue
            for j in range(i + 1, min(i + 20, len(selected_list))):
                rem2 = selected_list[j]
                if not selected[rem2]:
                    continue

                # Temporarily remove both
                selected[rem1] = 0
                selected[rem2] = 0
                count -= 2
                for fid in edge_to_faces[rem1]:
                    face_count[fid] -= 1
                for fid in edge_to_faces[rem2]:
                    face_count[fid] -= 1

                # Try to add
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

                if added <= 2:
                    # Revert
                    for eidx in new_adds:
                        selected[eidx] = 0
                        count -= 1
                        for fid in edge_to_faces[eidx]:
                            face_count[fid] -= 1
                    selected[rem1] = 1
                    selected[rem2] = 1
                    count += 2
                    for fid in edge_to_faces[rem1]:
                        face_count[fid] += 1
                    for fid in edge_to_faces[rem2]:
                        face_count[fid] += 1
                else:
                    improved = True

    return selected, face_count, count


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    edges, E, face_edges, F, edge_to_faces, d, n = _build_edge_data()

    best_count = 0
    best_selected = None

    # Multiple random restarts with complementary deletion + local search
    for attempt in range(200):
        if time.time() > deadline - 5:
            break

        rng = random.Random(attempt * 7919 + 13)

        # Complementary deletion: start full, remove to break C4s
        selected, face_count, count = _complementary_deletion(face_edges, F, edge_to_faces, E, rng)

        # Try to add back edges
        selected, face_count, count = _try_add_edges(selected, face_count, count, E, face_edges, F, edge_to_faces, rng)

        # Local search with swaps
        remaining = deadline - time.time() - 2
        if remaining > 1:
            selected, face_count, count = _local_search(
                selected, face_count, count, E, face_edges, F, edge_to_faces, rng, min(remaining, 2.0)
            )

        # Multi-removal search if time permits
        remaining = deadline - time.time() - 2
        if remaining > 1:
            selected, face_count, count = _multi_removal_search(
                selected, face_count, count, E, face_edges, F, edge_to_faces, rng, min(remaining, 3.0)
            )

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Convert to edge list
    if best_selected is not None:
        result = []
        for i in range(E):
            if best_selected[i]:
                u, v, dim = edges[i]
                result.append([u, v])
        return result
    else:
        # Fallback: simple greedy
        result = []
        chosen = set()
        for x in range(n):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c, e = x ^ (1 << j), y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        result.append([x, y])
        return result

# EVOLVE-BLOCK-END
