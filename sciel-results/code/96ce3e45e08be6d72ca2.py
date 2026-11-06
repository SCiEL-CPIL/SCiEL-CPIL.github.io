import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 1 << d
    deadline = time.time() + 140.0  # safety margin under 150s

    # Build all edges of Q7
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
    m = len(edges)  # 448

    # Build all 2D faces (4-cycles). A face is defined by two distinct dims (i, j) and a base vertex.
    # Face vertices: a, a^i, a^j, a^(i^j) where a has bits i,j = 0 (canonical).
    faces = []
    face_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            bi = 1 << i
            bj = 1 << j
            for a in range(n):
                if (a & bi) == 0 and (a & bj) == 0:
                    v1 = a
                    v2 = a ^ bi
                    v3 = a ^ bj
                    v4 = a ^ bi ^ bj
                    # four edges of the face
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    faces.append((e1, e2, e3, e4))
    f_count = len(faces)  # 280

    # For each edge, list of faces containing it
    edge_faces = [[] for _ in range(m)]
    for fi, face in enumerate(faces):
        for e in face:
            edge_faces[e].append(fi)

    # Conflict: two edges conflict if there exists a face containing both with the other two edges
    # already selected. We'll maintain this dynamically.

    # Initial LP / Lagrangian relaxation bound:
    # Maximize sum x_e s.t. for each face, sum x_e <= 3, 0<=x<=1.
    # The LP optimum gives an upper bound. We compute a simple bound:
    # Each face can contribute at most 3. Use LP via scipy if available, else greedy bound.

    def lp_bound(selected, forbidden):
        """Upper bound on total selected edges given current state.
        selected: set of edge indices already chosen.
        forbidden: set of edge indices that cannot be chosen.
        """
        remaining = m - len(selected) - len(forbidden)
        # For each remaining edge, check if it's still selectable (not in a face with 3 selected)
        # LP bound: sum over remaining edges of x_e where x_e in [0,1]
        # subject to face constraints. A simple valid upper bound:
        # count remaining edges that are "alive" (can still be in some feasible completion)
        # plus a fractional face-penalty.
        alive = 0
        for e in range(m):
            if e in selected or e in forbidden:
                continue
            # e is alive if no face containing e already has 3 selected
            alive_ok = True
            for fi in edge_faces[e]:
                face = faces[fi]
                sel_count = sum(1 for ee in face if ee in selected)
                if sel_count >= 3:
                    alive_ok = False
                    break
            if alive_ok:
                alive += 1
        # LP relaxation: each face with k selected (k<3) can add at most (3-k) more
        # This is a valid upper bound via the face constraints.
        # Simple: alive + 0 (conservative). Better: use LP.
        return len(selected) + alive

    # Stronger LP bound using scipy
    try:
        from scipy.optimize import linprog
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    def lp_bound_scipy(selected, forbidden):
        """Solve LP relaxation for a tighter bound."""
        if not HAVE_SCIPY:
            return lp_bound(selected, forbidden)
        # Variables: x_e for remaining edges
        rem = [e for e in range(m) if e not in selected and e not in forbidden]
        rem_idx = {e: i for i, e in enumerate(rem)}
        nr = len(rem)
        if nr == 0:
            return len(selected)
        # Objective: maximize sum x_e -> minimize -sum x_e
        c = -np.ones(nr)
        # Constraints: for each face, sum x_e (over rem in face) + sel_count_in_face <= 3
        A_ub = []
        b_ub = []
        for fi, face in enumerate(faces):
            sel_c = sum(1 for ee in face if ee in selected)
            rem_in_face = [ee for ee in face if ee in rem_idx]
            if not rem_in_face:
                if sel_c > 3:
                    return -1  # infeasible
                continue
            row = np.zeros(nr)
            for ee in rem_in_face:
                row[rem_idx[ee]] = 1.0
            A_ub.append(row)
            b_ub.append(3.0 - sel_c)
        if not A_ub:
            return len(selected) + nr
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
        if res.status == 0:
            return len(selected) + int(np.ceil(-res.fun - 1e-6))
        return len(selected) + nr  # fallback

    # Branch and bound
    best_edges = set()
    best_count = 0

    # Initial greedy solution to seed best
    def greedy_init(order=None):
        chosen = set()
        out = []
        if order is None:
            order = list(range(m))
        for e in order:
            x, y = edges[e]
            ok = True
            for fi in edge_faces[e]:
                face = faces[fi]
                sel = sum(1 for ee in face if ee in chosen)
                if sel >= 3:
                    ok = False
                    break
            if ok:
                chosen.add(e)
                out.append(e)
        return chosen, out

    # Try multiple greedy orderings, keep best
    import random
    rng = random.Random(42)
    greedy_orderings = [
        list(range(m)),
        sorted(range(m), key=lambda e: len(edge_faces[e])),
        sorted(range(m), key=lambda e: -len(edge_faces[e])),
    ]
    for _ in range(20):
        o = list(range(m))
        rng.shuffle(o)
        greedy_orderings.append(o)

    for order in greedy_orderings:
        ch, out = greedy_init(order)
        if len(ch) > best_count:
            best_count = len(ch)
            best_edges = set(ch)

    # Local improvement: try to add edges, then swap
    def try_improve(chosen):
        improved = True
        while improved:
            improved = False
            # Add any edge that fits
            for e in range(m):
                if e in chosen:
                    continue
                ok = True
                for fi in edge_faces[e]:
                    face = faces[fi]
                    sel = sum(1 for ee in face if ee in chosen)
                    if sel >= 3:
                        ok = False
                        break
                if ok:
                    chosen.add(e)
                    improved = True
        return chosen

    best_edges = try_improve(best_edges)
    best_count = len(best_edges)

    # Branch and bound search
    def bnb(selected, forbidden, depth, bound_fn):
        nonlocal best_count, best_edges
        if time.time() > deadline:
            return
        # Bound
        ub = bound_fn(selected, forbidden)
        if ub <= best_count:
            return
        # Find a remaining edge to branch on
        rem = [e for e in range(m) if e not in selected and e not in forbidden]
        if not rem:
            if len(selected) > best_count:
                best_count = len(selected)
                best_edges = set(selected)
            return
        # Pick most constrained edge: highest number of faces with 2 selected
        def constraint_score(e):
            score = 0
            for fi in edge_faces[e]:
                face = faces[fi]
                sel = sum(1 for ee in face if ee in selected)
                score += (3 - sel) ** 2
            return score
        rem_sorted = sorted(rem, key=constraint_score, reverse=True)
        e = rem_sorted[0]

        # Branch 1: select e
        new_sel = set(selected)
        new_sel.add(e)
        # Forbid edges that would complete a face
        new_forb = set(forbidden)
        for fi in edge_faces[e]:
            face = faces[fi]
            sel_in_face = [ee for ee in face if ee in new_sel]
            if len(sel_in_face) >= 3:
                # the remaining edge in this face is forbidden
                for ee in face:
                    if ee not in new_sel and ee not in new_forb:
                        new_forb.add(ee)
        bnb(new_sel, new_forb, depth + 1, bound_fn)

        # Branch 2: forbid e
        new_forb2 = set(forbidden)
        new_forb2.add(e)
        bnb(selected, new_forb2, depth + 1, bound_fn)

    # Use LP bound if available, else simple bound
    use_lp = HAVE_SCIPY
    bound_fn = lp_bound_scipy if use_lp else lp_bound

    # Limit depth to keep within time
    bnb(best_edges, set(), 0, bound_fn)

    # Final improvement
    best_edges = try_improve(best_edges)
    best_count = len(best_edges)

    # Return edges as [u, v]
    out = []
    for e in best_edges:
        x, y = edges[e]
        out.append([x, y])
    return out
# EVOLVE-BLOCK-END
