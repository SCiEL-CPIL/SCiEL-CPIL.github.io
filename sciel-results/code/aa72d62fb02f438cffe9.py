import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Build Q3 (3-cube) adjacency
    q3_adj = {i: set() for i in range(8)}
    for i in range(8):
        for j in range(i + 1, 8):
            diff = i ^ j
            if diff & (diff - 1) == 0:  # differ in exactly one bit
                q3_adj[i].add(j)
                q3_adj[j].add(i)

    # Start with K3,13: vertices 0,1,2 on one side, 3..15 on the other
    edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            edge_set.add((a, b))

    # Candidate edges to try adding: within {0,1,2} and within {3..15}
    candidates = []
    for i in range(3):
        for j in range(i + 1, 3):
            candidates.append((i, j))
    for i in range(3, 16):
        for j in range(i + 1, 16):
            candidates.append((i, j))

    # Try to add edges greedily, checking Q3-freedom after each addition
    def is_q3_free(edge_set):
        # Check all 8-vertex subsets for Q3 subgraph
        for subset in itertools.combinations(range(16), 8):
            if time.time() > deadline:
                return False  # Assume not Q3-free to be safe on timeout
            verts = list(subset)
            # Count edges in induced subgraph
            cnt = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = verts[i], verts[j]
                    if (u, v) in edge_set:
                        cnt += 1
            if cnt < 12:
                continue
            # Build local adjacency
            adj = {v: set() for v in verts}
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = verts[i], verts[j]
                    if (u, v) in edge_set:
                        adj[u].add(v)
                        adj[v].add(u)
            # Backtracking to find Q3 isomorphism
            mapping = {}
            def bt(q3_v, used):
                if q3_v == 8:
                    return True
                for gv in verts:
                    if gv in used:
                        continue
                    ok = True
                    for nb in q3_adj[q3_v]:
                        if nb in mapping and gv not in adj[mapping[nb]]:
                            ok = False
                            break
                    if not ok:
                        continue
                    mapping[q3_v] = gv
                    if bt(q3_v + 1, used | {gv}):
                        return True
                    del mapping[q3_v]
                return False
            for v0 in verts:
                mapping[0] = v0
                if bt(1, {v0}):
                    return False  # Q3 found
                del mapping[0]
        return True

    # Greedy edge addition
    for edge in candidates:
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        if not is_q3_free(edge_set):
            edge_set.remove((u, v))

    # Convert to list of [u, v]
    result = []
    for u, v in edge_set:
        result.append([min(u, v), max(u, v)])
    return result
