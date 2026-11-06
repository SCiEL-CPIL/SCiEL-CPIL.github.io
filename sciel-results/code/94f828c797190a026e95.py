import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill

    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    best_n, best_edges = get_seed_graph()

    # Try to improve beyond the seed. Targets descending.
    targets = [42, 41, 40, 39, 38]
    for target_n in targets:
        if time.time() > deadline:
            break
        time_budget = deadline - time.time()
        if time_budget < 15:
            break
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                if n >= 42:
                    break
    return best_n, best_edges


def count_violations_full(n, adj):
    """Return (total, k5_per_vertex, ik5_per_vertex). O(n^5) but tiny constant."""
    k5_pv = np.zeros(n, dtype=np.int32)
    ik5_pv = np.zeros(n, dtype=np.int32)
    # For each 5-set, count as K5 or IK5, add 1 to each of its 5 vertices
    for combo in itertools.combinations(range(n), 5):
        # check all 10 edges
        all_edge = True
        all_nonedge = True
        for a, b in itertools.combinations(combo, 2):
            if adj[a, b]:
                all_nonedge = False
            else:
                all_edge = False
            if not all_edge and not all_nonedge:
                break
        if all_edge:
            for v in combo:
                k5_pv[v] += 1
        elif all_nonedge:
            for v in combo:
                ik5_pv[v] += 1
    return int(np.sum(k5_pv) + np.sum(ik5_pv)), k5_pv, ik5_pv


def solve_for_n(n, time_budget):
    start = time.time()
    deadline = start + time_budget
    random.seed(42 + n)
    np.random.seed(42 + n)

    # ---- Build an initial graph by extending the 37-vertex Paley seed ----
    base_q = 37
    residues = {(x * x) % base_q for x in range(1, base_q)}
    adj = np.zeros((n, n), dtype=np.int8)
    # Place seed on first base_q vertices
    for i in range(base_q):
        for j in range(i + 1, base_q):
            if (j - i) % base_q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    # Extend: new vertices connect to seed by random, and to each other randomly
    for i in range(base_q, n):
        for j in range(base_q):
            if random.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1

    # Precompute, for each vertex v, the list of 3-subsets of (V \ {v})
    # used to test "does adding v to a triangle form a K4/IK4 with v".
    # For incremental counting of K5/IK5 we need, for an edge (u,v) flip,
    # the 5-sets containing both u and v.

    def full_state():
        total, k5_pv, ik5_pv = count_violations_full(n, adj)
        return total, k5_pv, ik5_pv

    total, k5_pv, ik5_pv = full_state()
    best_total = total

    # Precompute for each unordered pair (u,v): list of triples w,x,y
    # such that {u,v,w,x,y} is a 5-set. Actually we need 5-sets containing
    # both u and v: choose 3 more from the other n-2 vertices.
    # That's C(n-2, 3). For each such triple, flipping (u,v) changes the
    # 5-set's edge-status of (u,v). We recompute whether the 5-set is
    # all-edge or all-nonedge.
    # To keep it fast, for a flip we recompute the affected 5-sets.

    def affected_5sets(u, v):
        """Yield 5-tuples (as tuple) that contain both u and v."""
        others = [w for w in range(n) if w != u and w != v]
        for t in itertools.combinations(others, 3):
            yield (u, v) + t

    def is_all_edge(c):
        for a, b in itertools.combinations(c, 2):
            if not adj[a, b]:
                return False
        return True

    def is_all_nonedge(c):
        for a, b in itertools.combinations(c, 2):
            if adj[a, b]:
                return False
        return True

    max_steps = 4000
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if total == 0:
            edges_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges_list.append((i, j))
            return n, edges_list

        # Pick a random 5-set that is a violation; if none found by sampling,
        # fall back to a random edge flip.
        # Find a violation by sampling a few random 5-sets.
        found_combo = None
        for _ in range(200):
            combo = tuple(random.sample(range(n), 5))
            if is_all_edge(combo) or is_all_nonedge(combo):
                found_combo = combo
                break

        if found_combo is None:
            # Could not find a violation by sampling; do a random flip.
            u, v = random.sample(range(n), 2)
            if u > v:
                u, v = v, u
            # Apply flip with incremental update
            new_total, k5_pv, ik5_pv = do_flip(u, v, total, k5_pv, ik5_pv)
            total = new_total
            if total < best_total:
                best_total = total
            continue

        # Try flipping each pair in the violating combo; pick the best.
        pairs = list(itertools.combinations(found_combo, 2))
        random.shuffle(pairs)
        best_flip = None
        best_new_total = total + 1000
        for u, v in pairs[:10]:
            new_total, k5_pv_new, ik5_pv_new = do_flip(u, v, total, k5_pv, ik5_pv)
            if new_total < best_new_total:
                best_new_total = new_total
                best_flip = (u, v, new_total, k5_pv_new, ik5_pv_new)

        if best_flip:
            u, v, new_total, k5_pv_new, ik5_pv_new = best_flip
            total = new_total
            k5_pv = k5_pv_new
            ik5_pv = ik5_pv_new
            if total < best_total:
                best_total = total
        else:
            # Random escape
            u, v = random.sample(range(n), 2)
            if u > v:
                u, v = v, u
            total, k5_pv, ik5_pv = do_flip(u, v, total, k5_pv, ik5_pv)
            if total < best_total:
                best_total = total

    # Did we get to 0?
    if total == 0:
        edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges_list.append((i, j))
        return n, edges_list
    return None


def do_flip(u, v, total, k5_pv, ik5_pv):
    """Flip edge (u,v) in the global adj, update violation counts incrementally.
    Returns (new_total, new_k5_pv, new_ik5_pv) and mutates adj."""
    global adj
    was_edge = adj[u, v]
    adj[u, v] = 1 - was_edge
    adj[v, u] = 1 - was_edge

    new_k5_pv = k5_pv.copy()
    new_ik5_pv = ik5_pv.copy()
    delta_total = 0

    others = [w for w in range(n) if w != u and w != v]
    for t in itertools.combinations(others, 3):
        c = (u, v) + t
        # Determine the 5-set status before and after the flip.
        # Only the (u,v) edge changed.
        # Check the other 9 edges.
        other_edges = []
        for a, b in itertools.combinations(c, 2):
            if (a, b) == (u, v) or (a, b) == (v, u):
                continue
            other_edges.append(adj[a, b])
        all_other_edge = all(other_edges)
        all_other_nonedge = all(e == 0 for e in other_edges)

        # Before flip:
        before_edge = was_edge
        after_edge = 1 - was_edge
        # A 5-set is a K5 iff all 10 edges present.
        before_k5 = all_other_edge and before_edge
        after_k5 = all_other_edge and after_edge
        before_ik5 = all_other_nonedge and (not before_edge)
        after_ik5 = all_other_nonedge and (not after_edge)

        for x in c:
            if before_k5:
                new_k5_pv[x] -= 1
            if after_k5:
                new_k5_pv[x] += 1
            if before_ik5:
                new_ik5_pv[x] -= 1
            if after_ik5:
                new_ik5_pv[x] += 1
        # total delta: (after_k5 - before_k5) + (after_ik5 - before_ik5)
        delta_total += (int(after_k5) - int(before_k5)) + (int(after_ik5) - int(before_ik5))

    new_total = total + delta_total
    return new_total, new_k5_pv, new_ik5_pv
