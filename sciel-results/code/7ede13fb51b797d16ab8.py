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
    conflict = [set() for _ in range(m)]
    for f in range(n_faces):
        fe = face_edges[f]
        for i in range(4):
            for j in range(i + 1, 4):
                conflict[fe[i]].add(fe[j])
                conflict[fe[j]].add(fe[i])
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

    def two_opt_polish(R, budget_sec):
        """Two-opt polish: try swapping pairs of hitting set edges to reduce size."""
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
            # Try single removals first
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
            if not improved:
                # Try 2-opt: remove one edge from R, add one edge to R
                # This can reduce size if the added edge covers faces that
                # allowed removal of multiple edges
                for e_rem in range(m):
                    if not R[e_rem]:
                        continue
                    # Temporarily remove e_rem
                    for f in edge_faces[e_rem]:
                        fc[f] -= 1
                    # Check if any edge can now be removed
                    # Actually, for 2-opt we want to swap: remove e1, add e2
                    # where e2 is not in R and e1 is in R
                    # Net effect: size stays same, but might enable further reductions
                    # Better: try to find e_add not in R such that adding it
                    # allows removing 2+ edges from R
                    for e_add in range(m):
                        if R[e_add]:
                            continue
                        # Check if adding e_add helps
                        # Count how many faces e_add covers that are currently covered only by e_rem
                        helpful = 0
                        for f in edge_faces[e_add]:
                            if fc[f] == 0:
                                helpful += 1
                        if helpful > 0:
                            # Add e_add
                            R[e_add] = True
                            for f in edge_faces[e_add]:
                                fc[f] += 1
                            # Now try to remove e_rem again
                            ok = True
                            for f in edge_faces[e_rem]:
                                if fc[f] <= 1:
                                    ok = False
                                    break
                            if ok:
                                R[e_rem] = False
                                for f in edge_faces[e_rem]:
                                    fc[f] -= 1
                                improved = True
                                break
                            else:
                                # Revert e_add
                                R[e_add] = False
                                for f in edge_faces[e_add]:
                                    fc[f] -= 1
                    # Restore e_rem
                    for f in edge_faces[e_rem]:
                        fc[f] += 1
                    if improved:
                        break
        return R

    # Strategy 1: Greedy hitting set + SA
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)
    
    best_R = R0[:]
    best_size = initial_size

    # SA restarts with more iterations
    num_restarts = 8
    per_restart_time = min(12.0, 80.0 / num_restarts)

    for r in range(num_restarts):
        if time.time() > deadline - 15:
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
            # More aggressive perturbation
            k = random.randint(6, 14)
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

    # Final polish with two-opt
    best_R = two_opt_polish(best_R, min(10.0, deadline - time.time() - 5))
    best_size = sum(best_R)

    # Build output
    fc_final = [0] * n_faces
    for e in range(m):
        if not best_R[e]:
            for f in edge_faces[e]:
                fc_final[f] += 1

    if any(c == 4 for c in fc_final):
        R_g, _ = greedy_hit_set()
        best_R = two_opt_polish(R_g, 5.0)

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
