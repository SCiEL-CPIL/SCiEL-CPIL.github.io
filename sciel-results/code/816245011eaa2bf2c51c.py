import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0

    d = 7
    n_vertices = 1 << d
    m = n_vertices * d // 2  # 448

    # Precompute edges: index -> (u, v), and (u,v) -> index
    edges = []
    edge_map = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1

    n_edges = len(edges)

    # Precompute conflicts: for each edge, list of 3-tuples of other edges
    # in each C4 it belongs to
    conflicts = [[] for _ in range(n_edges)]
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            conflicts[idx].append((idx1, idx2, idx3))

    # For each edge, also precompute which faces (as frozensets of 4 edge indices) contain it
    # face_to_edges: list of frozensets, each with 4 edge indices
    # edge_faces: list of lists of face indices
    face_to_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    seen_faces = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(u, v2), max(u, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                face = frozenset([e1, e2, e3, e4])
                if face not in seen_faces:
                    seen_faces.add(face)
                    face_idx = len(face_to_edges)
                    face_to_edges.append(face)
                    for e in face:
                        edge_faces[edge_map[e]].append(face_idx)

    def is_valid_add(edge_idx, selected):
        for t in conflicts[edge_idx]:
            if t[0] in selected and t[1] in selected and t[2] in selected:
                return False
        return True

    def greedy_fill(selected, available_indices):
        for idx in available_indices:
            if idx not in selected:
                if is_valid_add(idx, selected):
                    selected.add(idx)
        return selected

    all_indices = list(range(n_edges))
    best_set = set()
    best_size = 0

    seeds = [42, 123, 456, 789, 1024, 2048, 333, 777, 999, 111, 222, 333, 444, 555, 666, 777, 888, 999]

    for seed in seeds:
        if time.time() - start_time > time_limit:
            break

        random.seed(seed)
        current_set = set()

        # Greedy fill with shuffled order
        shuffled = list(all_indices)
        random.shuffle(shuffled)
        greedy_fill(current_set, shuffled)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

        # Local search with save/restore
        # Track for each edge how many of its C4s are "saturated" (all 3 other edges present)
        # sat_count[e] = number of C4s where the other 3 edges are all in selected
        # An edge can be added iff sat_count[e] == 0
        # When we add edge e, for each C4 containing e, increment sat_count of the other 3 edges
        # When we remove edge e, for each C4 containing e, decrement sat_count of the other 3 edges

        sat_count = [0] * n_edges
        for e in current_set:
            for t in conflicts[e]:
                # For each C4 containing e, check if the other 3 are in selected
                # Actually, we need to count for each other edge how many C4s with e are saturated
                pass
        # Better: recompute sat_count from scratch
        for e in range(n_edges):
            cnt = 0
            for t in conflicts[e]:
                if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                    cnt += 1
            sat_count[e] = cnt

        # List of edges that can be added (sat_count == 0 and not in current_set)
        addable = [e for e in range(n_edges) if e not in current_set and sat_count[e] == 0]

        iterations = 0
        max_iterations = 500000
        no_improve = 0

        while iterations < max_iterations and time.time() - start_time < time_limit:
            iterations += 1

            # Try to add a random addable edge
            if addable:
                idx = random.choice(addable)
                if idx not in current_set and sat_count[idx] == 0:
                    # Add it
                    current_set.add(idx)
                    addable.remove(idx)
                    # Update sat_count for edges in conflicts of idx
                    for t in conflicts[idx]:
                        for other in t:
                            if other not in current_set:
                                # Check if after adding idx, the C4 is now saturated for 'other'
                                # The C4 has edges {idx, t[0], t[1], t[2]}
                                # For 'other' to be blocked, the other 3 edges (including idx) must be in current_set
                                # So we need to check if the other 2 (besides idx and other) are in current_set
                                other2 = [x for x in t if x != other and x != idx]
                                if len(other2) == 2:
                                    if other2[0] in current_set and other2[1] in current_set:
                                        sat_count[other] += 1
                                        if sat_count[other] == 1 and other in addable:
                                            addable.remove(other)
                            else:
                                # other is in current_set, no change to its sat_count needed for addability
                                pass
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    no_improve = 0
                    continue

            # Try to remove an edge and add another
            # Find edges whose removal would unblock some addable edges
            # For each edge e in current_set, compute how many addable edges would become unblocked
            # This is expensive, so do it randomly
            if current_set:
                # Pick a random edge to try removing
                # Use a list for fast random access
                if not hasattr(construct, '_cur_list'):
                    pass
                cur_list = list(current_set)
                remove_idx = random.choice(cur_list)

                # Save state
                saved_set = set(current_set)
                saved_sat = list(sat_count)
                saved_addable = list(addable)

                # Remove the edge
                current_set.remove(remove_idx)
                if remove_idx in addable:
                    addable.remove(remove_idx)

                # Update sat_count: for each C4 containing remove_idx, decrement sat_count of others
                for t in conflicts[remove_idx]:
                    for other in t:
                        if other not in current_set:
                            # Check if the C4 was saturated before removal
                            other2 = [x for x in t if x != other and x != remove_idx]
                            if len(other2) == 2:
                                if other2[0] in current_set and other2[1] in current_set:
                                    sat_count[other] -= 1
                                    if sat_count[other] == 0 and other not in addable:
                                        addable.append(other)
                        # If other is in current_set, no change needed

                # Now try to add an edge from the newly available set
                added = False
                if addable:
                    # Prefer edges that were just unblocked
                    new_candidates = [e for e in addable if e not in saved_addable]
                    if new_candidates:
                        candidates = new_candidates
                    else:
                        candidates = addable
                    for _ in range(20):
                        if not candidates:
                            break
                        add_idx = random.choice(candidates)
                        if add_idx not in current_set and sat_count[add_idx] == 0:
                            current_set.add(add_idx)
                            if add_idx in addable:
                                addable.remove(add_idx)
                            for t in conflicts[add_idx]:
                                for other in t:
                                    if other not in current_set:
                                        other2 = [x for x in t if x != other and x != add_idx]
                                        if len(other2) == 2:
                                            if other2[0] in current_set and other2[1] in current_set:
                                                sat_count[other] += 1
                                                if sat_count[other] == 1 and other in addable:
                                                    addable.remove(other)
                            added = True
                            break

                if not added:
                    # Revert
                    current_set = saved_set
                    sat_count = saved_sat
                    addable = saved_addable
                    no_improve += 1
                    if no_improve > 5000:
                        break
                else:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    no_improve = 0

    # Final greedy fill
    remaining = [i for i in all_indices if i not in best_set]
    random.shuffle(remaining)
    greedy_fill(best_set, remaining)

    result = []
    for idx in best_set:
        result.append(list(edges[idx]))

    return result
