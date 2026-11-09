import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    d = 7
    n = 1 << d
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    m = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    faces = []
    edge_faces = [[] for _ in range(m)]
    for i in range(d):
        for j in range(i+1, d):
            for base in range(n):
                if (base >> i) & 1 or (base >> j) & 1:
                    continue
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                e0 = edge_index[(min(v0,v1), max(v0,v1))]
                e1 = edge_index[(min(v2,v3), max(v2,v3))]
                e2 = edge_index[(min(v0,v2), max(v0,v2))]
                e3 = edge_index[(min(v1,v3), max(v1,v3))]
                fi = len(faces)
                faces.append((e0, e1, e2, e3))
                for e in (e0, e1, e2, e3):
                    edge_faces[e].append(fi)
    num_faces = len(faces)
    
    def greedy_init(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = [False] * m
        face_sel = [0] * num_faces
        for idx in order:
            ok = True
            for f in edge_faces[idx]:
                if face_sel[f] >= 3:
                    ok = False
                    break
            if ok:
                selected[idx] = True
                for f in edge_faces[idx]:
                    face_sel[f] += 1
        return selected, face_sel
    
    def local_search(selected, face_sel, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(m):
                if not selected[idx]:
                    ok = True
                    for f in edge_faces[idx]:
                        if face_sel[f] >= 3:
                            ok = False
                            break
                    if ok:
                        selected[idx] = True
                        for f in edge_faces[idx]:
                            face_sel[f] += 1
                        improved = True
            for idx in range(m):
                if time.time() >= deadline:
                    break
                if not selected[idx]:
                    continue
                selected[idx] = False
                for f in edge_faces[idx]:
                    face_sel[f] -= 1
                added = 0
                for jdx in range(m):
                    if jdx == idx or selected[jdx]:
                        continue
                    ok = True
                    for f in edge_faces[jdx]:
                        if face_sel[f] >= 3:
                            ok = False
                            break
                    if ok:
                        selected[jdx] = True
                        for f in edge_faces[jdx]:
                            face_sel[f] += 1
                        added += 1
                if added >= 1:
                    improved = True
                else:
                    selected[idx] = True
                    for f in edge_faces[idx]:
                        face_sel[f] += 1
        return selected, face_sel
    
    def perturb(selected, face_sel, k, rng):
        """Randomly remove k selected edges."""
        sel_list = [i for i in range(m) if selected[i]]
        rng.shuffle(sel_list)
        for idx in sel_list[:k]:
            selected[idx] = False
            for f in edge_faces[idx]:
                face_sel[f] -= 1
        return selected, face_sel
    
    best_size = 0
    best_selected = None
    trial = 0
    rng = random.Random(42)
    
    while time.time() < DEADLINE - 10:
        trial += 1
        selected, face_sel = greedy_init(seed=trial)
        selected, face_sel = local_search(selected, face_sel, DEADLINE - 5)
        cur_size = sum(selected)
        if cur_size > best_size:
            best_size = cur_size
            best_selected = selected[:]
            # Perturb and search from perturbed state
            for _ in range(3):
                if time.time() >= DEADLINE - 5:
                    break
                pert_sel = best_selected[:]
                pert_face = [0] * num_faces
                for i in range(m):
                    if pert_sel[i]:
                        for f in edge_faces[i]:
                            pert_face[f] += 1
                perturb(pert_sel, pert_face, rng.randint(5, 15), rng)
                pert_sel, pert_face = local_search(pert_sel, pert_face, DEADLINE - 5)
                psize = sum(pert_sel)
                if psize > best_size:
                    best_size = psize
                    best_selected = pert_sel[:]
    
    if best_selected is None:
        selected, face_sel = greedy_init(seed=0)
        best_selected = selected
    
    result = [[edges[idx][0], edges[idx][1]] for idx in range(m) if best_selected[idx]]
    return result
