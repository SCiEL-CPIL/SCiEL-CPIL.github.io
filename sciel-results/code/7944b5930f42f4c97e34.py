import random
import time

def construct():
    d = 7
    n = 2 ** d
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s

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

    # For each edge, the 6 faces (as frozenset of 4 edge indices) containing it
    # Build faces: for each pair (i,j) and each assignment of other 5 bits
    edge_faces = [[] for _ in range(m)]
    face_edges = []  # list of tuples of 4 edge indices
    for i in range(d):
        for j in range(i+1, d):
            # other dims
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                # fixed values for others
                fixed = {}
                for bit, k in enumerate(others):
                    fixed[k] = (mask >> bit) & 1
                # base vertex with i=0,j=0, others=fixed
                base = 0
                for k, v in fixed.items():
                    if v:
                        base |= (1 << k)
                v00 = base
                v10 = base ^ (1 << i)
                v01 = base ^ (1 << j)
                v11 = base ^ (1 << i) ^ (1 << j)
                # 4 edges
                e_a = (min(v00, v10), max(v00, v10))  # along i at j=0
                e_b = (min(v01, v11), max(v01, v11))  # along i at j=1
                e_c = (min(v00, v01), max(v00, v01))  # along j at i=0
                e_d = (min(v10, v11), max(v10, v11))  # along j at i=1
                idxs = [edge_idx[e_a], edge_idx[e_b], edge_idx[e_c], edge_idx[e_d]]
                face_edges.append(tuple(idxs))
                for idx in idxs:
                    edge_faces[idx].append(len(face_edges)-1)

    F = len(face_edges)  # 672

    def greedy_random(rng):
        chosen = [False] * m
        face_count = [0] * F
        order = list(range(m))
        rng.shuffle(order)
        for e in order:
            ok = True
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    ok = False
                    break
            if ok:
                chosen[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
        return chosen, face_count

    def count_chosen(chosen):
        return sum(chosen)

    def try_add(e, chosen, face_count):
        for f in edge_faces[e]:
            if face_count[f] >= 3:
                return False
        chosen[e] = True
        for f in edge_faces[e]:
            face_count[f] += 1
        return True

    def do_remove(e, chosen, face_count):
        chosen[e] = False
        for f in edge_faces[e]:
            face_count[f] -= 1

    def is_addable(e, chosen, face_count):
        if chosen[e]:
            return False
        for f in edge_faces[e]:
            if face_count[f] >= 3:
                return False
        return True

    rng = random.Random(12345)
    best_count = 0
    best_chosen = None

    # Initial greedy
    chosen, face_count = greedy_random(rng)
    # Fill to stuck
    changed = True
    while changed:
        changed = False
        for e in range(m):
            if is_addable(e, chosen, face_count):
                try_add(e, chosen, face_count)
                changed = True
    c = count_chosen(chosen)
    if c > best_count:
        best_count = c
        best_chosen = chosen[:]

    # Local search with swaps
    while time.time() < DEADLINE:
        # random restart occasionally
        if rng.random() < 0.15:
            chosen, face_count = greedy_random(rng)
            changed = True
            while changed:
                changed = False
                for e in range(m):
                    if is_addable(e, chosen, face_count):
                        try_add(e, chosen, face_count)
                        changed = True
            c = count_chosen(chosen)
            if c > best_count:
                best_count = c
                best_chosen = chosen[:]
            continue

        # Do a batch of swaps
        improved = False
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            # pick a random edge to remove
            e_rem = rng.randrange(m)
            if not chosen[e_rem]:
                continue
            # find a face with count==3 containing e_rem
            candidates = [f for f in edge_faces[e_rem] if face_count[f] == 3]
            if not candidates:
                # removing e_rem from a face with count<3 is fine but may allow adds
                # just remove and try to add
                do_remove(e_rem, chosen, face_count)
                # try adds
                for e in range(m):
                    if is_addable(e, chosen, face_count):
                        try_add(e, chosen, face_count)
                improved = True
                break
            f = rng.choice(candidates)
            # pick another edge in face f to remove? No - we want to remove e_rem and add something
            # After removing e_rem, face f count becomes 2, so e_rem becomes addable again (no gain)
            # Instead: remove e_rem, then try to add edges
            do_remove(e_rem, chosen, face_count)
            added = 0
            for e in range(m):
                if is_addable(e, chosen, face_count):
                    try_add(e, chosen, face_count)
                    added += 1
            if added >= 1:
                improved = True
                break
            else:
                # revert: put e_rem back
                for ff in edge_faces[e_rem]:
                    face_count[ff] += 1
                chosen[e_rem] = True

        c = count_chosen(chosen)
        if c > best_count:
            best_count = c
            best_chosen = chosen[:]
            improved = True

        if not improved:
            # stuck, do a few random swaps to escape
            for _ in range(50):
                e_rem = rng.randrange(m)
                if not chosen[e_rem]:
                    continue
                do_remove(e_rem, chosen, face_count)
                # try to add 2 edges
                added = 0
                for e in range(m):
                    if is_addable(e, chosen, face_count):
                        try_add(e, chosen, face_count)
                        added += 1
                        if added >= 2:
                            break
                if added < 1:
                    for ff in edge_faces[e_rem]:
                        face_count[ff] += 1
                    chosen[e_rem] = True
                c2 = count_chosen(chosen)
                if c2 > best_count:
                    best_count = c2
                    best_chosen = chosen[:]
                break

    # Build output
    out = []
    for e in range(m):
        if best_chosen[e]:
            out.append(list(edges[e]))
    return out
