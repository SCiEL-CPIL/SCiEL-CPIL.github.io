import random
import time

def construct():
    d = 7
    n = 2 ** d
    start = time.time()
    DEADLINE = 145.0

    # Build edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_idx[e] = len(edges)
                edges.append(e)
    m = len(edges)  # 448

    # Build faces
    face_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                fixed = {}
                for bit, k in enumerate(others):
                    fixed[k] = (mask >> bit) & 1
                base = 0
                for k, v in fixed.items():
                    if v:
                        base |= (1 << k)
                v00 = base
                v10 = base ^ (1 << i)
                v01 = base ^ (1 << j)
                v11 = base ^ (1 << i) ^ (1 << j)
                e_a = (min(v00, v10), max(v00, v10))
                e_b = (min(v01, v11), max(v01, v11))
                e_c = (min(v00, v01), max(v00, v01))
                e_d = (min(v10, v11), max(v10, v11))
                idxs = [edge_idx[e_a], edge_idx[e_b], edge_idx[e_c], edge_idx[e_d]]
                face_edges.append(tuple(idxs))

    F = len(face_edges)

    # edge_faces[e] = list of face indices
    edge_faces = [[] for _ in range(m)]
    for fi, face in enumerate(face_edges):
        for ei in face:
            edge_faces[ei].append(fi)

    # face_edge_list[fi] = list of 4 edge indices
    face_edge_list = [list(f) for f in face_edges]

    def can_add(e, face_count):
        for f in edge_faces[e]:
            if face_count[f] >= 3:
                return False
        return True

    def add_edge(e, chosen, face_count):
        chosen[e] = True
        for f in edge_faces[e]:
            face_count[f] += 1

    def remove_edge(e, chosen, face_count):
        chosen[e] = False
        for f in edge_faces[e]:
            face_count[f] -= 1

    rng = random.Random(42)
    best_count = 0
    best_chosen = None

    while time.time() < DEADLINE:
        # Random greedy init
        chosen = [False] * m
        face_count = [0] * F
        order = list(range(m))
        rng.shuffle(order)
        for e in order:
            if can_add(e, face_count):
                add_edge(e, chosen, face_count)

        current_count = sum(chosen)
        if current_count > best_count:
            best_count = current_count
            best_chosen = chosen[:]

        # Local search: try removing one edge and adding one or more
        for _ in range(3000):
            if time.time() > DEADLINE:
                break

            e_rem = rng.randrange(m)
            if not chosen[e_rem]:
                continue

            # Remove
            remove_edge(e_rem, chosen, face_count)

            # Find addable edges in affected faces
            addable = set()
            for f in edge_faces[e_rem]:
                if face_count[f] <= 2:
                    for e in face_edge_list[f]:
                        if not chosen[e] and can_add(e, face_count):
                            addable.add(e)

            if not addable:
                add_edge(e_rem, chosen, face_count)
                continue

            # Try each addable edge
            found_improvement = False
            for e_add in addable:
                add_edge(e_add, chosen, face_count)

                # Try to add more
                extra_added = []
                for f in edge_faces[e_add]:
                    if face_count[f] <= 2:
                        for e in face_edge_list[f]:
                            if not chosen[e] and can_add(e, face_count):
                                add_edge(e, chosen, face_count)
                                extra_added.append(e)

                net = 1 + len(extra_added) - 1
                if net >= 1:
                    current_count += net
                    if current_count > best_count:
                        best_count = current_count
                        best_chosen = chosen[:]
                    found_improvement = True
                    break
                else:
                    # Revert
                    for e in extra_added:
                        remove_edge(e, chosen, face_count)
                    remove_edge(e_add, chosen, face_count)

            if not found_improvement:
                # Revert removal
                add_edge(e_rem, chosen, face_count)

    out = []
    for e in range(m):
        if best_chosen[e]:
            out.append(list(edges[e]))
    return out
