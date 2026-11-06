import time
import random
import numpy as np

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 115

    # Precompute edges
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

    # Precompute faces (672 total)
    face_edges = []
    edge_faces = [[] for _ in range(m)]

    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)

                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]

                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)

    n_faces = len(face_edges)  # 672

    # Convert to numpy arrays for speed
    face_edges_np = np.array(face_edges, dtype=np.int32)
    edge_faces_lists = [list(f) for f in edge_faces]

    best_removed = None
    best_size = m

    iterations = 80
    for it in range(iterations):
        if time.time() > deadline - 12:
            break

        # --- Multi-stage local search: greedy init, then 2-opt improvements ---
        R = set()
        uncovered = set(range(n_faces))

        # Greedy phase
        while uncovered:
            # Count how many uncovered faces each edge covers
            counts = np.zeros(m, dtype=np.int32)
            for f in uncovered:
                counts[face_edges_np[f]] += 1
            max_count = counts.max()
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_e = int(random.choice(candidates))
            R.add(best_e)
            for f in edge_faces_lists[best_e]:
                if f in uncovered:
                    uncovered.discard(f)

        # Build coverage array for current R
        def build_cover(R_set):
            cover = np.zeros(n_faces, dtype=np.int32)
            for e in R_set:
                for f in edge_faces_lists[e]:
                    cover[f] += 1
            return cover

        cover = build_cover(R)

        # Local search: try to remove redundant edges, then try 1-for-1 swaps
        improved = True
        max_iter = 2000
        li = 0
        while improved and li < max_iter and time.time() < deadline - 12:
            improved = False
            li += 1

            # Phase 1: remove redundant edges
            changed = False
            for e in list(R):
                redundant = True
                for f in edge_faces_lists[e]:
                    if cover[f] == 1:
                        redundant = False
                        break
                if redundant:
                    R.discard(e)
                    for f in edge_faces_lists[e]:
                        cover[f] -= 1
                    changed = True
                    improved = True
            if changed:
                continue

            # Phase 2: 1-for-1 swap — try removing one edge in R and adding one not in R
            # Find edges in R that are "most expensive" (cover few unique faces)
            # For each edge e in R, compute how many faces it uniquely covers
            # If it uniquely covers 0 faces, it was caught above, so all cover >= 1
            # Try: for edge e in R, remove it, then for each face f that becomes uncovered,
            # we need to pick a replacement edge.
            # Simpler: try all (remove e_in, add e_out) pairs where e_out covers
            # at least the faces that e_in uniquely covers.
            # This is O(|R| * (m-|R|) * 4) which is too slow if |R| ~ 140.
            # Instead, do it more cleverly.

            # Try removing each edge e_in in R, and see if we can re-cover the
            # newly uncovered faces using edges NOT in R.
            for e_in in list(R):
                if time.time() > deadline - 12:
                    break
                # Tentatively remove e_in
                newly_uncovered = set()
                for f in edge_faces_lists[e_in]:
                    if cover[f] == 1:
                        newly_uncovered.add(f)
                    cover[f] -= 1

                if not newly_uncovered:
                    # This shouldn't happen after Phase 1, but just in case
                    R.discard(e_in)
                    improved = True
                    changed = True
                    break

                # Find a single edge e_out not in R that covers ALL newly_uncovered faces
                found = False
                for e_out in range(m):
                    if e_out in R:
                        continue
                    covers_all = True
                    for f in newly_uncovered:
                        if e_out not in face_edges_np[f]:
                            covers_all = False
                            break
                    if covers_all:
                        # Execute swap
                        R.discard(e_in)
                        R.add(e_out)
                        # Update cover
                        for f in edge_faces_lists[e_out]:
                            cover[f] += 1
                        improved = True
                        found = True
                        break

                if not found:
                    # Roll back
                    for f in edge_faces_lists[e_in]:
                        cover[f] += 1

                if improved:
                    break
            if improved:
                continue

            # Phase 3: 1-for-2 swap — remove one edge, add two edges
            # Skip this phase for simplicity given time constraints

        # Record result
        if len(R) < best_size:
            best_size = len(R)
            best_removed = set(R)
            if best_size <= 140:
                break  # Theoretical lower bound region

    if best_removed is None:
        best_removed = set(range(m))
        best_size = m

    # Construct final answer
    selected_edges = []
    for e in range(m):
        if e not in best_removed:
            x, y = all_edges[e]
            selected_edges.append([x, y])

    return selected_edges

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")

    d = 7
    n = 1 << d
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))

    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)

                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))

                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1

                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break

    if valid:
        print("Valid!")
    else:
        print("Invalid!")
