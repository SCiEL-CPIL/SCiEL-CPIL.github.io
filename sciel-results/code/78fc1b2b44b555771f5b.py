"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 130  # safety margin before 150s kill

    # Precompute all edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)

    n_faces = len(face_edges)  # 112

    # State
    chosen = [False] * m
    face_count = [0] * n_faces
    cur_size = 0

    def add_edge(ei):
        nonlocal cur_size
        chosen[ei] = True
        cur_size += 1
        for fid in edge_faces[ei]:
            face_count[fid] += 1

    def remove_edge(ei):
        nonlocal cur_size
        chosen[ei] = False
        cur_size -= 1
        for fid in edge_faces[ei]:
            face_count[fid] -= 1

    def can_add(ei):
        for fid in edge_faces[ei]:
            if face_count[fid] >= 3:
                return False
        return True

    random.seed(12345)
    best_size = 0
    best_state = None

    # Multiple restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        # Reset
        chosen = [False] * m
        face_count = [0] * n_faces
        cur_size = 0

        # Greedy init
        order = list(range(m))
        random.shuffle(order)
        for ei in order:
            if can_add(ei):
                add_edge(ei)

        # Local search
        stuck_count = 0
        while time.time() < deadline and stuck_count < 500:
            # Try to add edges
            improved = False
            for ei in range(m):
                if not chosen[ei] and can_add(ei):
                    add_edge(ei)
                    improved = True

            if improved:
                stuck_count = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_state = chosen[:]
            else:
                stuck_count += 1
                # Kick: remove a random chosen edge, add a random unchosen edge
                chosen_edges = [ei for ei in range(m) if chosen[ei]]
                unchosen_edges = [ei for ei in range(m) if not chosen[ei]]
                if not chosen_edges or not unchosen_edges:
                    break
                rem = random.choice(chosen_edges)
                add_candidate = random.choice(unchosen_edges)
                remove_edge(rem)
                if can_add(add_candidate):
                    add_edge(add_candidate)
                else:
                    # Try a few more
                    for _ in range(10):
                        cand = random.choice(unchosen_edges)
                        if can_add(cand):
                            add_edge(cand)
                            break
                    else:
                        # Revert
                        add_edge(rem)

        if cur_size > best_size:
            best_size = cur_size
            best_state = chosen[:]

    if best_state is None:
        # Fallback: greedy only
        chosen = [False] * m
        face_count = [0] * n_faces
        cur_size = 0
        order = list(range(m))
        random.shuffle(order)
        for ei in order:
            if can_add(ei):
                add_edge(ei)
        best_state = chosen[:]

    # Build output
    out = []
    for ei in range(m):
        if best_state[ei]:
            x, y = all_edges[ei]
            out.append([x, y])
    return out
# EVOLVE-BLOCK-END
