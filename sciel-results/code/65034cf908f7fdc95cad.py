import time
import random
import math

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 108  # Safety margin

    # Precompute all edges (u < v, differ in one bit)
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

    # Precompute all 2-faces (672) and their 4 edges
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dv = x ^ (1 << i) ^ (1 << j)
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, dv), max(b, dv))]
                    e3 = edge_idx[(min(dv, c), max(dv, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
    n_faces = len(face_edges)  # 672

    # Build conflict graph: two edges conflict if they share a face
    # conflict[e] = set of edges that conflict with e
    conflict = [set() for _ in range(m)]
    for f in range(n_faces):
        fe = face_edges[f]
        for i in range(4):
            for j in range(i + 1, 4):
                conflict[fe[i]].add(fe[j])
                conflict[fe[j]].add(fe[i])
    # Convert to lists for speed
    conflict_list = [list(s) for s in conflict]

    def greedy_hit_set():
        """Greedy: pick edge covering most uncovered faces."""
        R = [False] * m
        fc = [0] * n_faces
        unc = [True] * n_faces
        n_unc = n_faces
        while n_unc > 0:
            best_e = -1
            best_g = -1
            for e in range(m):
                if R[e]:
                    continue
                g = 0
                for f in edge_faces[e]:
                    if unc[f]:
                        g += 1
                if g > best_g:
                    best_g = g
                    best_e = e
            if best_e < 0:
                break
            R[best_e] = True
            for f in edge_faces[best_e]:
                if unc[f]:
                    unc[f] = False
                    n_unc -= 1
                fc[f] += 1
        return R, fc

    def greedy_mis():
        """Greedy Maximum Independent Set on conflict graph.
        Returns set of selected (non-conflicting) edge indices."""
        # Greedy coloring-based approach for MIS
        # Use greedy: repeatedly pick vertex with fewest conflicts among remaining
        remaining = set(range(m))
        selected = []
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = m + 1
            for v in remaining:
                deg = sum(1 for u in remaining if u in conflict[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            selected.append(best_v)
            remaining.remove(best_v)
            for u in conflict[best_v]:
                remaining.discard(u)
        return selected

    def local_search_mis(selected_set, budget_sec):
        """Local search on MIS: try to add edges, remove conflicting pairs."""
        sel = set(selected_set)
        cur_size = len(sel)
        best_size = cur_size
        best_sel = sel.copy()
        
        t_end = time.time() + budget_sec
        T = 2.0
        T_min = 0.05
        cool = 0.9999
        
        while time.time() < t_end:
            # Try to add a random edge not in sel
            e = random.randint(0, m - 1)
            if e not in sel:
                # Check if e conflicts with any edge in sel
                conflicts_in_sel = [u for u in conflict_list[e] if u in sel]
                if not conflicts_in_sel:
                    # Can add e freely
                    sel.add(e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_sel = sel.copy()
                else:
                    # Try to swap: remove one conflicting edge, add e
                    # Pick the conflicting edge that has the most conflicts (least "valuable")
                    victim = random.choice(conflicts_in_sel)
                    # Check if removing victim and adding e is beneficial
                    # e is always beneficial (we add 1), but we lose victim
                    # Only do this if it doesn't create new conflicts
                    new_sel_check = sel - {victim}
                    can_add = all(u not in new_sel_check for u in conflict_list[e])
                    if can_add:
                        sel.remove(victim)
                        sel.add(e)
                        # Size stays same, but we might enable future additions
                        # Accept with probability based on temperature
                        if random.random() < math.exp(-0.5 / T):
                            pass  # accept
                        else:
                            # Revert
                            sel.remove(e)
                            sel.add(victim)
            else:
                # e is in sel, try to remove it and see if we can add more
                # This is less useful; skip for now
                pass
            
            T *= cool
            if T < T_min:
                T = T_min
        
        return best_sel, best_size

    def sa_minimize(R_init, fc_init, budget_sec):
        """Simulated annealing to minimize |R| (hitting set size)."""
        R = R_init[:]
        fc = fc_init[:]
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]

        n_unc = sum(1 for c in fc if c == 0)

        T = 3.0
        T_min = 0.01
        cool = 0.9998
        t_end = time.time() + budget_sec

        while time.time() < t_end:
            e = random.randint(0, m - 1)
            if R[e]:
                newly_unc = []
                for f in edge_faces[e]:
                    if fc[f] == 1:
                        newly_unc.append(f)
                new_n_unc = n_unc + len(newly_unc)
                new_size = cur_size - 1
                if new_n_unc > 0:
                    delta = 1.0 + 5.0 * len(newly_unc)
                else:
                    delta = -1.0

                for f in edge_faces[e]:
                    fc[f] -= 1
                R[e] = False
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                if new_n_unc == 0:
                    pass
                else:
                    if random.random() < math.exp(-delta / T):
                        pass
                    else:
                        R[e] = True
                        for f in edge_faces[e]:
                            fc[f] += 1
                        cur_size += 1
                        n_unc -= len(newly_unc)
            else:
                dec_unc = 0
                for f in edge_faces[e]:
                    if fc[f] == 0:
                        dec_unc += 1
                new_n_unc = n_unc - dec_unc
                new_size = cur_size + 1
                if dec_unc > 0:
                    delta = 1.0 - 5.0 * dec_unc
                else:
                    delta = 1.0

                R[e] = True
                for f in edge_faces[e]:
                    fc[f] += 1
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                if dec_unc > 0:
                    pass
                else:
                    if random.random() < math.exp(-delta / T):
                        pass
                    else:
                        R[e] = False
                        for f in edge_faces[e]:
                            fc[f] -= 1
                        cur_size -= 1
                        n_unc += dec_unc

            T *= cool
            if T < T_min:
                T = T_min

        return best_R, best_size

    def polish(R, budget_sec):
        """Final polish: iteratively try to remove edges from hitting set."""
        R = R[:]
        fc = [0] * n_faces
        for e in range(m):
            if R[e]:
                for f in edge_faces[e]:
                    fc[f] += 1
        
        t_end = time.time() + budget_sec
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for e in range(m):
                if R[e]:
                    ok = True
                    for f in edge_faces[e]:
                        if fc[f] <= 1:
                            ok = False
                            break
                    if ok:
                        R[e] = False
                        for f in edge_faces[e]:
                            fc[f] -= 1
                        improved = True
        return R

    # Strategy 1: Greedy hitting set + SA
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)
    
    best_R = R0[:]
    best_size = initial_size

    # SA restarts
    num_restarts = 6
    per_restart_time = min(14.0, 90.0 / num_restarts)

    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        if r == 0:
            R_r = R0[:]
            fc_r = fc0[:]
        else:
            R_r = best_R[:]
            fc_r = [0] * n_faces
            for e in range(m):
                if R_r[e]:
                    for f in edge_faces[e]:
                        fc_r[f] += 1
            k = random.randint(4, 10)
            for _ in range(k):
                e = random.randint(0, m - 1)
                R_r[e] = not R_r[e]
            fc_r = [0] * n_faces
            for e in range(m):
                if R_r[e]:
                    for f in edge_faces[e]:
                        fc_r[f] += 1

        R_new, size_new = sa_minimize(R_r, fc_r, per_restart_time)
        if size_new < best_size:
            best_size = size_new
            best_R = R_new

    # Strategy 2: Greedy MIS + local search
    if time.time() < deadline - 15:
        mis_init = greedy_mis()
        mis_set, mis_size = local_search_mis(mis_init, min(20.0, deadline - time.time() - 10))
        # Convert MIS to hitting set: R[e] = True if e NOT in MIS
        R_mis = [False] * m
        for e in range(m):
            if e not in mis_set:
                R_mis[e] = True
        # Verify validity
        fc_check = [0] * n_faces
        for e in range(m):
            if not R_mis[e]:
                for f in edge_faces[e]:
                    fc_check[f] += 1
        if not any(c == 4 for c in fc_check):
            mis_hs_size = sum(R_mis)
            if mis_hs_size < best_size:
                best_size = mis_hs_size
                best_R = R_mis

    # Final polish
    best_R = polish(best_R, min(5.0, deadline - time.time() - 3))
    best_size = sum(best_R)

    # Build output
    fc_final = [0] * n_faces
    for e in range(m):
        if not best_R[e]:
            for f in edge_faces[e]:
                fc_final[f] += 1

    if any(c == 4 for c in fc_final):
        R_g, _ = greedy_hit_set()
        best_R = polish(R_g, 3.0)

    out = []
    for e in range(m):
        if not best_R[e]:
            x, y = all_edges[e]
            out.append([x, y])

    return out


if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")

    # Quick validation
    d = 7
    n = 1 << d
    sel = set()
    for u, v in edges:
        sel.add((min(u, v), max(u, v)))
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dv = x ^ (1 << i) ^ (1 << j)
                    cnt = 0
                    for p, q in [(a, b), (b, dv), (dv, c), (c, a)]:
                        if (min(p, q), max(p, q)) in sel:
                            cnt += 1
                    if cnt == 4:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    print("Valid!" if valid else "INVALID!")
