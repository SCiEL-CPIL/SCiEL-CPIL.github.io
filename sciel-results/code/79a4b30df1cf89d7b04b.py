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

    # Cache conflict matrix for reuse
    import numpy as np
    conf_mat = np.zeros((m, m), dtype=bool)
    for e in range(m):
        for u in conflict[e]:
            conf_mat[e, u] = True

    def greedy_mis():
        """Greedy Maximum Independent Set on conflict graph (vectorized).
        Returns list of selected (non-conflicting) edge indices."""
        remaining = np.ones(m, dtype=bool)
        selected = []
        while remaining.any():
            degs = conf_mat[:, remaining].sum(axis=1)
            degs[~remaining] = m + 1
            best_v = int(np.argmin(degs))
            selected.append(best_v)
            remaining[best_v] = False
            remaining[conf_mat[best_v, :]] = False
        return selected

    def local_search_mis(selected_set, budget_sec):
        """Local search on MIS: add edges, 2-opt swaps, simulated annealing."""
        sel = set(selected_set)
        cur_size = len(sel)
        best_size = cur_size
        best_sel = sel.copy()
        
        t_end = time.time() + budget_sec
        T = 2.0
        T_min = 0.05
        cool = 0.9999
        
        while time.time() < t_end:
            action = random.random()
            if action < 0.3:
                # Try to add a random edge not in sel
                e = random.randint(0, m - 1)
                if e not in sel:
                    conflicts_in_sel = [u for u in conflict_list[e] if u in sel]
                    if not conflicts_in_sel:
                        sel.add(e)
                        cur_size += 1
                        if cur_size > best_size:
                            best_size = cur_size
                            best_sel = sel.copy()
                    else:
                        victim = random.choice(conflicts_in_sel)
                        new_sel_check = sel - {victim}
                        can_add = all(u not in new_sel_check for u in conflict_list[e])
                        if can_add:
                            sel.remove(victim)
                            sel.add(e)
                            if random.random() >= math.exp(-0.5 / T):
                                sel.remove(e)
                                sel.add(victim)
            elif action < 0.6:
                # 2-opt swap: pick e in sel, f not in sel
                # Remove e, add f if f doesn't conflict with sel \ {e}
                # This can enable future additions even at same size
                e = random.choice(list(sel))
                f = random.randint(0, m - 1)
                if f not in sel:
                    new_sel = sel - {e}
                    can_add_f = all(u not in new_sel for u in conflict_list[f])
                    if can_add_f:
                        # Check: does removing e free up any edge we could add?
                        # For now just do the swap (size-neutral, explores landscape)
                        sel.discard(e)
                        sel.add(f)
                        if random.random() >= math.exp(-0.3 / T):
                            sel.discard(f)
                            sel.add(e)
            else:
                # Try to add a random edge (same as first branch)
                e = random.randint(0, m - 1)
                if e not in sel:
                    conflicts_in_sel = [u for u in conflict_list[e] if u in sel]
                    if not conflicts_in_sel:
                        sel.add(e)
                        cur_size += 1
                        if cur_size > best_size:
                            best_size = cur_size
                            best_sel = sel.copy()
                    else:
                        # Try removing the conflict with fewest alternative connections
                        best_victim = None
                        best_score = -1
                        for v in conflicts_in_sel:
                            # Score: how many other sel edges conflict with v (lower = less valuable)
                            alt_conf = sum(1 for u in conflict_list[v] if u in sel and u != e)
                            if alt_conf < best_score or best_victim is None:
                                best_score = alt_conf
                                best_victim = v
                        if best_victim is not None:
                            new_sel_check = sel - {best_victim}
                            can_add = all(u not in new_sel_check for u in conflict_list[e])
                            if can_add:
                                sel.remove(best_victim)
                                sel.add(e)
                                if random.random() >= math.exp(-0.5 / T):
                                    sel.remove(e)
                                    sel.add(best_victim)
            
            T *= cool
            if T < T_min:
                T = T_min
        
        return best_sel, best_size

    def sa_minimize(R_init, fc_init, budget_sec):
        """Simulated annealing to minimize |R| (hitting set size) with kicks."""
        R = R_init[:]
        fc = fc_init[:]
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]

        n_unc = sum(1 for c in fc if c == 0)

        T = 5.0
        T_min = 0.005
        cool = 0.9995
        t_end = time.time() + budget_sec
        last_improve = time.time()
        kick_interval = 2.0  # seconds without improvement before kicking

        while time.time() < t_end:
            # Kick mechanism: if stuck, randomly flip some edges
            if time.time() - last_improve > kick_interval:
                for _ in range(random.randint(3, 8)):
                    e = random.randint(0, m - 1)
                    if R[e]:
                        newly_unc = sum(1 for f in edge_faces[e] if fc[f] == 1)
                        if newly_unc <= 2:
                            for f in edge_faces[e]:
                                fc[f] -= 1
                            R[e] = False
                            cur_size -= 1
                            n_unc += newly_unc
                    else:
                        dec_unc = sum(1 for f in edge_faces[e] if fc[f] == 0)
                        if dec_unc >= 1:
                            for f in edge_faces[e]:
                                fc[f] += 1
                            R[e] = True
                            cur_size += 1
                            n_unc -= dec_unc
                last_improve = time.time()
                T = max(T, 2.0)  # Reheat

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
                    last_improve = time.time()

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
                    last_improve = time.time()

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
        """Final polish: iteratively try to remove edges from hitting set, with 2-edge swaps."""
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
            # Pass 1: Remove single edges that are redundant
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
            # Pass 2: Try 2-edge swaps (remove one, add another that covers more)
            if time.time() < t_end:
                for e1 in range(m):
                    if R[e1]:
                        # Check if removing e1 would leave any face uncovered
                        removable = True
                        for f in edge_faces[e1]:
                            if fc[f] <= 1:
                                removable = False
                                break
                        if not removable:
                            continue
                        # Try to find an edge e2 not in R that could replace e1
                        # e2 must cover at least one face that e1 covers, and not conflict
                        # Simple heuristic: try a few random candidates
                        for _ in range(5):
                            e2 = random.randint(0, m - 1)
                            if R[e2]:
                                continue
                            # Check if e2 conflicts with any selected edge (non-R edge)
                            conflicts = False
                            for u in conflict_list[e2]:
                                if not R[u]:
                                    conflicts = True
                                    break
                            if conflicts:
                                continue
                            # Check if e2 covers any face that e1 covers
                            overlap = 0
                            for f in edge_faces[e1]:
                                if fc[f] >= 2:  # Face is covered by at least 2 edges
                                    # Check if e2 also covers this face
                                    if f in edge_faces[e2]:
                                        overlap += 1
                            if overlap > 0:
                                # Do the swap
                                R[e1] = False
                                for f in edge_faces[e1]:
                                    fc[f] -= 1
                                R[e2] = True
                                for f in edge_faces[e2]:
                                    fc[f] += 1
                                improved = True
                                break
        return R

    # Strategy 1: Greedy hitting set + SA
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)
    
    best_R = R0[:]
    best_size = initial_size

    # SA restarts with adaptive budget
    remaining_time = deadline - time.time() - 20  # Reserve 20s for MIS + polish
    num_restarts = max(6, int(remaining_time / 12.0))
    per_restart_time = remaining_time / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 25:
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
            # More aggressive perturbation for later restarts
            k = random.randint(5 + r, 12 + r)
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

    # Strategy 2: Greedy MIS + local search (multiple restarts)
    if time.time() < deadline - 20:
        mis_budget_total = max(10.0, deadline - time.time() - 15)
        num_mis_restarts = max(2, int(mis_budget_total / 15.0))
        per_mis_time = mis_budget_total / num_mis_restarts
        
        best_sel_mis = None
        best_mis_size = 0
        
        for mr in range(num_mis_restarts):
            if time.time() > deadline - 15:
                break
            if mr == 0:
                mis_init = greedy_mis()
            else:
                # Perturb previous best
                if best_sel_mis is not None:
                    mis_init = list(best_sel_mis)
                else:
                    mis_init = greedy_mis()
                # Remove some random elements and re-greedy
                perturbed = set(mis_init)
                for _ in range(random.randint(5, 15)):
                    if perturbed:
                        perturbed.discard(random.choice(list(perturbed)))
                # Fill with greedy on remaining
                remaining = set(range(m)) - perturbed
                # Simple: just use perturbed as starting point
                mis_init = list(perturbed)
            
            mis_set, mis_size = local_search_mis(mis_init, min(per_mis_time, deadline - time.time() - 15))
            
            if best_sel_mis is None or len(mis_set) > best_mis_size:
                best_sel_mis = mis_set
                best_mis_size = len(mis_set)
        
        # Convert best MIS to hitting set: R[e] = True if e NOT in MIS
        if best_sel_mis is not None:
            R_mis = [False] * m
            for e in range(m):
                if e not in best_sel_mis:
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

    # Final polish with more time
    polish_time = min(10.0, max(3.0, deadline - time.time() - 5))
    best_R = polish(best_R, polish_time)
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
