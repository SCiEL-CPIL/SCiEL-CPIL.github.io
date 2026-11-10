import random
import time

def construct():
    d = 7
    n = 1 << d
    m = d * (1 << (d - 1))  # 448 edges

    # --- build edge list and face (4-cycle) incidence ---
    edges = []
    edge_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_idx[(u, v)] = idx
                edge_idx[(v, u)] = idx

    # faces: each 2D face is determined by (x, i, j) with i<j;
    # the 4 edges are (x,x^i),(x^j,x^(i|j)),(x^i,x^(i|j)),(x,x^j)
    # We'll precompute for each edge the list of face-ids it belongs to,
    # and for each face the 4 edge-ids.
    face_edges = []   # list of 4 edge-ids per face
    edge_faces = [[] for _ in range(m)]  # edge -> list of face ids

    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                a = x
                b = x ^ (1 << i)
                c = x ^ (1 << j)
                e = x ^ (1 << i) ^ (1 << j)
                # 4 edges: (a,b), (b,e), (e,c), (c,a)
                eid0 = edge_idx[(min(a, b), max(a, b))]
                eid1 = edge_idx[(min(b, e), max(b, e))]
                eid2 = edge_idx[(min(e, c), max(e, c))]
                eid3 = edge_idx[(min(c, a), max(c, a))]
                fid = len(face_edges)
                face_edges.append((eid0, eid1, eid2, eid3))
                for eid in (eid0, eid1, eid2, eid3):
                    edge_faces[eid].append(fid)

    # face_conflict_count[fid] = number of selected edges in that face (0..4)
    # edge_conflict[eid] = number of faces where adding this edge would make count==4
    # i.e., faces where currently 3 of the 4 edges are selected and this edge is the 4th

    deadline = time.time() + 130.0

    def greedy_random_seed(rng):
        """Random-order greedy: add edge if it doesn't complete a face."""
        sel = [False] * m
        face_cnt = [0] * len(face_edges)
        order = list(range(m))
        rng.shuffle(order)
        for eid in order:
            # check if adding eid would complete any face
            ok = True
            for fid in edge_faces[eid]:
                if face_cnt[fid] == 3:
                    ok = False
                    break
            if ok:
                sel[eid] = True
                for fid in edge_faces[eid]:
                    face_cnt[fid] += 1
        return sel, face_cnt

    def compute_edge_conflicts(sel, face_cnt):
        """For each edge, count how many faces it would complete if added."""
        conf = [0] * m
        for fid, (e0, e1, e2, e3) in enumerate(face_edges):
            cnt = face_cnt[fid]
            if cnt == 3:
                # exactly one edge is missing; find it
                for eid in (e0, e1, e2, e3):
                    if not sel[eid]:
                        conf[eid] += 1
                        break
        return conf

    def try_add(sel, face_cnt, eid):
        sel[eid] = True
        for fid in edge_faces[eid]:
            face_cnt[fid] += 1

    def try_remove(sel, face_cnt, eid):
        sel[eid] = False
        for fid in edge_faces[eid]:
            face_cnt[fid] -= 1

    def count_edges(sel):
        return sum(sel)

    def local_search(sel, face_cnt, rng, time_limit):
        """Iteratively add best edge or remove worst edge until stuck or time up."""
        conf = compute_edge_conflicts(sel, face_cnt)
        best_score = -1
        while time.time() < time_limit:
            # Find best move: add edge with conf==0, or remove edge that unblocks most
            best_move = None
            best_gain = 0

            # Try adding: any edge with conf==0 gives +1
            # Try removing: removing edge e gives gain = (number of edges that become addable) - 1
            # Approximate: gain = sum over faces containing e of (4 - face_cnt[fid] - 1 if face_cnt==3) ... 
            # Simpler: just try adding first (cheap), then removals

            # Add: pick any edge with conf==0
            added = False
            for eid in range(m):
                if not sel[eid] and conf[eid] == 0:
                    try_add(sel, face_cnt, eid)
                    conf = compute_edge_conflicts(sel, face_cnt)
                    added = True
                    break

            if not added:
                # No edge can be added. Try removing the edge whose removal
                # unblocks the most other edges.
                # For each selected edge, compute how many faces it's in with count==4 or 3
                best_remove = -1
                best_unblock = 0
                for eid in range(m):
                    if sel[eid]:
                        unblock = 0
                        for fid in edge_faces[eid]:
                            if face_cnt[fid] == 4:
                                # removing this edge makes face_cnt 3, so the other 3 remain
                                # but we need to check which other edges could be added
                                # Actually if face_cnt==4, removing one edge makes it 3,
                                # and the "missing" edge (the one we removed) is now addable again
                                # But more importantly, other edges in that face that were blocked
                                # by this face being full can now potentially be added
                                unblock += 1
                        if unblock > best_unblock:
                            best_unblock = unblock
                            best_remove = eid
                if best_remove >= 0 and best_unblock >= 1:
                    try_remove(sel, face_cnt, best_remove)
                    conf = compute_edge_conflicts(sel, face_cnt)
                else:
                    break  # stuck

        return sel, face_cnt

    best_sel = None
    best_count = 0
    rng = random.Random(42)

    # Multiple restarts with different random seeds
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        sel, face_cnt = greedy_random_seed(rng)
        cnt = count_edges(sel)
        if cnt > best_count:
            best_count = cnt
            best_sel = sel[:]

        # Local search
        time_left = deadline - time.time()
        if time_left > 3.0:
            sel, face_cnt = local_search(sel, face_cnt, rng, deadline - 2.0)
            cnt = count_edges(sel)
            if cnt > best_count:
                best_count = cnt
                best_sel = sel[:]

    if best_sel is None:
        # Fallback: simple greedy
        sel = [False] * m
        face_cnt = [0] * len(face_edges)
        for eid in range(m):
            ok = True
            for fid in edge_faces[eid]:
                if face_cnt[fid] == 3:
                    ok = False
                    break
            if ok:
                sel[eid] = True
                for fid in edge_faces[eid]:
                    face_cnt[fid] += 1
        best_sel = sel

    return [[edges[eid][0], edges[eid][1]] for eid in range(m) if best_sel[eid]]
# EVOLVE-BLOCK-END
