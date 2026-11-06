import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    d = 7
    n_vertices = 1 << d
    m = n_vertices * d // 2  # 448

    # Precompute edges: index -> (u, v)
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = idx
                idx += 1

    # Precompute faces: each face = (e0, e1, e2, e3)
    face_edges = []
    edge_face_info = [[] for _ in range(m)]  # edge -> list of face_ids

    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dd = x ^ (1 << i) ^ (1 << j)

                    e_ab = edge_map[(min(a, b), max(a, b))]
                    e_bd = edge_map[(min(b, dd), max(b, dd))]
                    e_dc = edge_map[(min(dd, c), max(dd, c))]
                    e_ca = edge_map[(min(c, a), max(c, a))]

                    fid = len(face_edges)
                    face_edges.append((e_ab, e_bd, e_dc, e_ca))

                    for e in (e_ab, e_bd, e_dc, e_ca):
                        edge_face_info[e].append(fid)

    n_faces = len(face_edges)

    def greedy_fill(selected, face_counts, candidate_order):
        """Add as many edges as possible from candidate_order (in that order)."""
        for idx in candidate_order:
            if idx not in selected:
                # Check if adding this edge completes any face
                can_add = True
                for fid in edge_face_info[idx]:
                    if face_counts[fid] == 3:
                        can_add = False
                        break
                if can_add:
                    selected.add(idx)
                    for fid in edge_face_info[idx]:
                        face_counts[fid] += 1

    def removal_gain(edge_idx, selected, face_counts):
        """Number of currently-forbidden edges that would become addable if edge_idx were removed."""
        gain = 0
        for fid in edge_face_info[edge_idx]:
            # If exactly 3 of the 4 edges in this face are selected, 
            # and edge_idx is one of them, then removing it makes the 4th addable
            if face_counts[fid] == 3:
                gain += 1
        return gain

    all_indices = list(range(m))
    best_set = set()
    best_size = 0
    best_face_counts = None

    # Phase 1: Multiple random greedy fills to find a good starting point
    for seed in [42, 123, 456, 789, 1024, 2048, 333, 777, 999, 111, 2222, 3333, 4444, 5555]:
        if time.time() - start_time > time_limit:
            break
        random.seed(seed)
        current = set()
        face_counts = [0] * n_faces
        order = list(all_indices)
        random.shuffle(order)
        greedy_fill(current, face_counts, order)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            best_face_counts = list(face_counts)

    # Phase 2: Local search
    current = set(best_set)
    face_counts = list(best_face_counts)
    cur_size = len(current)

    # SA parameters
    T0 = 2.0
    T_min = 0.05
    cooling = 0.995
    T = T0
    no_improve = 0
    max_no_improve = 500

    while time.time() - start_time < time_limit:
        if not current:
            break

        # Save state
        saved = set(current)
        saved_face_counts = list(face_counts)
        saved_size = len(current)

        # Choose an edge to remove: prefer high-gain, with some randomness
        # Sample up to 100 candidates for speed
        candidates = list(current)
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        best_e = None
        best_g = -1
        for e in candidates:
            g = removal_gain(e, current, face_counts)
            if g > best_g:
                best_g = g
                best_e = e

        if best_e is None:
            best_e = random.choice(list(current))

        # Remove best_e
        current.discard(best_e)
        for fid in edge_face_info[best_e]:
            face_counts[fid] -= 1

        # Greedily add back: try all edges not in current in random order
        remaining = [i for i in all_indices if i not in current]
        random.shuffle(remaining)
        greedy_fill(current, face_counts, remaining)

        new_size = len(current)

        # SA accept/reject
        if new_size > cur_size:
            cur_size = new_size
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_set = set(current)
                best_face_counts = list(face_counts)
        elif new_size == cur_size:
            # Accept with moderate probability (random walk)
            if random.random() < 0.5:
                cur_size = new_size
            else:
                current = set(saved)
                face_counts = list(saved_face_counts)
        else:
            # new_size < cur_size: accept with probability exp(-(cur_size - new_size)/T)
            delta = cur_size - new_size
            prob = min(1.0, 2.718281828 ** (-delta / max(T, 1e-6)))
            if random.random() < prob:
                cur_size = new_size
            else:
                current = set(saved)
                face_counts = list(saved_face_counts)

        # Cooling
        T = max(T_min, T * cooling)

        # If stuck for too long, do a kick: remove a few random edges and refill
        no_improve += 1
        if no_improve > max_no_improve:
            # Kick: remove 3 random edges, refill
            for _ in range(3):
                if current:
                    e = random.choice(list(current))
                    current.discard(e)
                    for fid in edge_face_info[e]:
                        face_counts[fid] -= 1
            remaining = [i for i in all_indices if i not in current]
            random.shuffle(remaining)
            greedy_fill(current, face_counts, remaining)
            cur_size = len(current)
            no_improve = 0
            T = max(T, T0 * 0.5)  # reheat a bit

    # Final polish: try to add any remaining edge
    remaining = [i for i in all_indices if i not in best_set]
    random.shuffle(remaining)
    face_counts = list(best_face_counts)
    greedy_fill(best_set, face_counts, remaining)

    # Convert to result
    result = []
    for idx in best_set:
        result.append(list(edges[idx]))

    return result
