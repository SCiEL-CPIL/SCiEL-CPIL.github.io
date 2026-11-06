import time
import random
import math

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 105  # Safety margin (150s kill, 45s buffer)

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

    # Multi-start SA to minimize hitting set size (maximize selected edges)
    # We work in "removed" space: R[e]=True means edge is in hitting set (removed)
    # Goal: minimize sum(R) subject to every face having >=1 removed edge

    def sa_minimize(R_init, fc_init, budget_sec):
        """Simulated annealing to minimize |R| (hitting set size)."""
        R = R_init[:]
        fc = fc_init[:]
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]

        # Track uncovered faces (fc[f] == 0)
        n_unc = sum(1 for c in fc if c == 0)
        best_unc = n_unc

        T = 3.0
        T_min = 0.01
        cool = 0.9998
        t_end = time.time() + budget_sec

        while time.time() < t_end:
            e = random.randint(0, m - 1)
            if R[e]:
                # Try removing edge e from hitting set
                # Check: after removal, are all faces of e still covered?
                new_n_unc = 0
                newly_unc = []
                for f in edge_faces[e]:
                    if fc[f] == 1:
                        newly_unc.append(f)
                new_n_unc = n_unc + len(newly_unc)
                new_size = cur_size - 1
                delta = new_size + 5.0 * (new_n_unc - n_unc) if new_n_unc > 0 else new_size - cur_size

                # Apply
                for f in edge_faces[e]:
                    fc[f] -= 1
                R[e] = False
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Accept or reject
                if new_n_unc == 0:
                    # Always accept if still valid
                    pass
                else:
                    # Probabilistic acceptance
                    if random.random() < math.exp(-delta / T):
                        pass  # accept
                    else:
                        # Revert
                        R[e] = True
                        for f in edge_faces[e]:
                            fc[f] += 1
                        cur_size += 1
                        n_unc -= len(newly_unc)
            else:
                # Try adding edge e to hitting set
                dec_unc = 0
                for f in edge_faces[e]:
                    if fc[f] == 0:
                        dec_unc += 1
                new_n_unc = n_unc - dec_unc
                new_size = cur_size + 1
                delta = 1.0 - 5.0 * dec_unc

                # Apply
                R[e] = True
                for f in edge_faces[e]:
                    fc[f] += 1
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Accept or reject
                if dec_unc > 0:
                    pass  # always accept (reduces uncovered)
                else:
                    if random.random() < math.exp(-delta / T):
                        pass  # accept
                    else:
                        # Revert
                        R[e] = False
                        for f in edge_faces[e]:
                            fc[f] -= 1
                        cur_size -= 1
                        n_unc += dec_unc

            T *= cool
            if T < T_min:
                T = T_min

        return best_R, best_size

    # Greedy init
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)

    # Multiple SA restarts with different seeds/temperatures
    best_R = R0[:]
    best_size = initial_size

    num_restarts = 8
    per_restart_time = min(12.0, 100.0 / num_restarts)

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        # Perturb the initial solution
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
            # Random perturbation: flip k edges
            k = random.randint(3, 8)
            for _ in range(k):
                e = random.randint(0, m - 1)
                R_r[e] = not R_r[e]
            # Recompute fc
            fc_r = [0] * n_faces
            for e in range(m):
                if R_r[e]:
                    for f in edge_faces[e]:
                        fc_r[f] += 1

        R_new, size_new = sa_minimize(R_r, fc_r, per_restart_time)
        if size_new < best_size:
            best_size = size_new
            best_R = R_new

    # Greedy reduction: remove redundant edges from hitting set
    changed = True
    while changed:
        changed = False
        fc = [0] * n_faces
        for e in range(m):
            if best_R[e]:
                for f in edge_faces[e]:
                    fc[f] += 1
        for e in range(m):
            if best_R[e]:
            # Can we remove e from hitting set?
                ok = True
                for f in edge_faces[e]:
                    if fc[f] <= 1:
                        ok = False
                        break
                if ok:
                    best_R[e] = False
                    for f in edge_faces[e]:
                        fc[f] -= 1
                    best_size -= 1
                    changed = True
                    break

    # Build output: selected edges are those NOT in hitting set
    # Verify validity
    fc_final = [0] * n_faces
    for e in range(m):
        if not best_R[e]:
            for f in edge_faces[e]:
                fc_final[f] += 1

    # If invalid (some face has all 4 edges selected = 0 in hitting set), fix
    if any(c == 4 for c in fc_final):
        # Fallback: use greedy hitting set
        R_g, fc_g = greedy_hit_set()
        best_R = R_g
        # Reduce
        changed = True
        while changed:
            changed = False
            fc = [0] * n_faces
            for e in range(m):
                if best_R[e]:
                    for f in edge_faces[e]:
                        fc[f] += 1
            for e in range(m):
                if best_R[e]:
                    ok = True
                    for f in edge_faces[e]:
                        if fc[f] <= 1:
                            ok = False
                            break
                    if ok:
                        best_R[e] = False
                        for f in edge_faces[e]:
                            fc[f] -= 1
                        changed = True
                        break

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
