import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0

    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)

    face_edges = []
    edge_faces = [[] for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                fid = len(face_edges)
                face_edges.append((e0, e1, e2, e3))
                for ei in (e0, e1, e2, e3):
                    edge_faces[ei].append(fid)

    num_faces = len(face_edges)
    edge_faces_arr = [np.array(f, dtype=np.int32) for f in edge_faces]

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = np.zeros(m, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        for ei in order:
            fids = edge_faces_arr[ei]
            if np.all(face_count[fids] < 3):
                selected[ei] = True
                face_count[fids] += 1
        return selected, face_count

    def complete_1flip(selected, face_count):
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                if selected[ei]:
                    continue
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    selected[ei] = True
                    face_count[fids] += 1
                    changed = True
        return selected, face_count

    def try_2swap(selected, face_count, seed, max_attempts=100):
        rng = random.Random(seed)
        sel_list = np.where(selected)[0].tolist()
        unsel_list = np.where(~selected)[0].tolist()
        if not sel_list or not unsel_list:
            return selected, face_count, 0

        # Score each selected edge by how many faces it blocks (faces at count 3)
        # Removing an edge that blocks many faces frees more capacity
        edge_block_score = np.zeros(m, dtype=np.int32)
        for ei in sel_list:
            fids = edge_faces_arr[ei]
            edge_block_score[ei] = np.sum(face_count[fids] >= 3)

        # Sort by block score descending, then shuffle within same score
        scored = sorted(sel_list, key=lambda x: -edge_block_score[x])
        # Add some randomness: shuffle within groups of same score
        rng.shuffle(scored)
        # Re-sort stably by score
        scored.sort(key=lambda x: -edge_block_score[x])

        rng.shuffle(unsel_list)

        best_gain = 0
        best_state = None

        for ri in scored[:max_attempts]:
            if time.time() >= deadline:
                break
            selected[ri] = False
            fids_ri = edge_faces_arr[ri]
            face_count[fids_ri] -= 1

            added = []
            for ai in unsel_list:
                if ai == ri:
                    continue
                fids_ai = edge_faces_arr[ai]
                if np.all(face_count[fids_ai] < 3):
                    selected[ai] = True
                    face_count[fids_ai] += 1
                    added.append(ai)

            gain = len(added) - 1
            if gain > best_gain:
                best_gain = gain
                best_state = (ri, added)

            selected[ri] = True
            face_count[fids_ri] += 1
            for ai in added:
                selected[ai] = False
                face_count[edge_faces_arr[ai]] -= 1

        if best_state is not None:
            ri, added = best_state
            selected[ri] = False
            face_count[edge_faces_arr[ri]] -= 1
            for ai in added:
                selected[ai] = True
                face_count[edge_faces_arr[ai]] += 1

        return selected, face_count, best_gain

    def try_2swap_remove2(selected, face_count, seed, max_attempts=40):
        """Remove 2 selected edges, add back all possible unselected edges."""
        rng = random.Random(seed)
        sel_list = np.where(selected)[0].tolist()
        unsel_list = np.where(~selected)[0].tolist()
        if len(sel_list) < 2 or not unsel_list:
            return selected, face_count, 0

        rng.shuffle(unsel_list)

        # Score edges by how many faces they block
        edge_block_score = np.zeros(m, dtype=np.int32)
        for ei in sel_list:
            fids = edge_faces_arr[ei]
            edge_block_score[ei] = np.sum(face_count[fids] >= 3)

        scored = sorted(sel_list, key=lambda x: -edge_block_score[x])
        rng.shuffle(scored)
        scored.sort(key=lambda x: -edge_block_score[x])

        best_gain = 0
        best_state = None

        # Try pairs from top-scored edges
        top = scored[:min(20, len(scored))]
        for i in range(len(top)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(top)):
                if time.time() >= deadline:
                    break
                ri, rj = top[i], top[j]
                fids_ri = edge_faces_arr[ri]
                fids_rj = edge_faces_arr[rj]
                selected[ri] = False
                selected[rj] = False
                face_count[fids_ri] -= 1
                face_count[fids_rj] -= 1

                added = []
                for ai in unsel_list:
                    if ai == ri or ai == rj:
                        continue
                    fids_ai = edge_faces_arr[ai]
                    if np.all(face_count[fids_ai] < 3):
                        selected[ai] = True
                        face_count[fids_ai] += 1
                        added.append(ai)

                gain = len(added) - 2
                if gain > best_gain:
                    best_gain = gain
                    best_state = (ri, rj, added)

                # Undo
                selected[ri] = True
                selected[rj] = True
                face_count[fids_ri] += 1
                face_count[fids_rj] += 1
                for ai in added:
                    selected[ai] = False
                    face_count[edge_faces_arr[ai]] -= 1

        if best_state is not None:
            ri, rj, added = best_state
            selected[ri] = False
            selected[rj] = False
            face_count[edge_faces_arr[ri]] -= 1
            face_count[edge_faces_arr[rj]] -= 1
            for ai in added:
                selected[ai] = True
                face_count[edge_faces_arr[ai]] += 1

        return selected, face_count, best_gain

    def count_sel(selected):
        return int(np.sum(selected))

    def local_search(selected, face_count, seed, max_iters=50):
        rng = random.Random(seed)
        current_count = count_sel(selected)
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()

        for it in range(max_iters):
            if time.time() >= deadline - 5.0:
                break

            # Try 1-flip (add)
            improved = False
            unsel_list = np.where(~selected)[0].tolist()
            rng.shuffle(unsel_list)
            for ai in unsel_list:
                fids_ai = edge_faces_arr[ai]
                if np.all(face_count[fids_ai] < 3):
                    selected[ai] = True
                    face_count[fids_ai] += 1
                    current_count += 1
                    improved = True
                    break

            if not improved:
                # Try 2-swap (remove 1, add k)
                sel_list = np.where(selected)[0].tolist()
                unsel_list = np.where(~selected)[0].tolist()

                # Score by blocking
                edge_block_score = np.zeros(m, dtype=np.int32)
                for ei in sel_list:
                    fids = edge_faces_arr[ei]
                    edge_block_score[ei] = np.sum(face_count[fids] >= 3)

                scored = sorted(sel_list, key=lambda x: -edge_block_score[x])
                rng.shuffle(scored)
                scored.sort(key=lambda x: -edge_block_score[x])

                found_swap = False
                for ri in scored[:50]:
                    if time.time() >= deadline - 5.0:
                        break
                    selected[ri] = False
                    fids_ri = edge_faces_arr[ri]
                    face_count[fids_ri] -= 1

                    added = []
                    for ai in unsel_list:
                        if ai == ri:
                            continue
                        fids_ai = edge_faces_arr[ai]
                        if np.all(face_count[fids_ai] < 3):
                            selected[ai] = True
                            face_count[fids_ai] += 1
                            added.append(ai)

                    gain = len(added) - 1
                    if gain > 0:
                        current_count += gain
                        found_swap = True
                        improved = True
                        break

                    selected[ri] = True
                    face_count[fids_ri] += 1
                    for ai in added:
                        selected[ai] = False
                        face_count[edge_faces_arr[ai]] -= 1

                if not found_swap:
                    # Try remove-2 swap
                    sel_list = np.where(selected)[0].tolist()
                    unsel_list = np.where(~selected)[0].tolist()
                    rng.shuffle(unsel_list)

                    edge_block_score = np.zeros(m, dtype=np.int32)
                    for ei in sel_list:
                        fids = edge_faces_arr[ei]
                        edge_block_score[ei] = np.sum(face_count[fids] >= 3)

                    scored = sorted(sel_list, key=lambda x: -edge_block_score[x])
                    top = scored[:min(15, len(scored))]
                    rng.shuffle(top)

                    found_swap2 = False
                    for i in range(len(top)):
                        if time.time() >= deadline - 5.0:
                            break
                        for j in range(i + 1, len(top)):
                            if time.time() >= deadline - 5.0:
                                break
                            ri, rj = top[i], top[j]
                            fids_ri = edge_faces_arr[ri]
                            fids_rj = edge_faces_arr[rj]
                            selected[ri] = False
                            selected[rj] = False
                            face_count[fids_ri] -= 1
                            face_count[fids_rj] -= 1

                            added = []
                            for ai in unsel_list:
                                if ai == ri or ai == rj:
                                    continue
                                fids_ai = edge_faces_arr[ai]
                                if np.all(face_count[fids_ai] < 3):
                                    selected[ai] = True
                                    face_count[fids_ai] += 1
                                    added.append(ai)

                            gain = len(added) - 2
                            if gain > 0:
                                current_count += gain
                                found_swap2 = True
                                improved = True
                                break
                            else:
                                selected[ri] = True
                                selected[rj] = True
                                face_count[fids_ri] += 1
                                face_count[fids_rj] += 1
                                for ai in added:
                                    selected[ai] = False
                                    face_count[edge_faces_arr[ai]] -= 1
                        if found_swap2:
                            break

                    if not found_swap2:
                        break

            if current_count > best_count:
                best_count = current_count
                best_selected = selected.copy()
                best_face_count = face_count.copy()

        return best_selected, best_face_count, best_count

    best_selected = None
    best_count = 0

    for seed in range(5000):
        if time.time() >= deadline - 5.0:
            break

        selected, face_count = greedy_random(seed)
        selected, face_count = complete_1flip(selected, face_count)
        cnt = count_sel(selected)

        if cnt > best_count:
            best_selected = selected.copy()
            best_count = cnt

        if time.time() < deadline - 10.0 and cnt < 304:
            selected, face_count, gain = try_2swap(selected, face_count, seed + 100000)
            if gain > 0:
                selected, face_count = complete_1flip(selected, face_count)
                cnt = count_sel(selected)
                if cnt > best_count:
                    best_selected = selected.copy()
                    best_count = cnt

        if time.time() < deadline - 15.0 and cnt < 304:
            # Also try remove-2 swap
            selected, face_count, gain2 = try_2swap_remove2(selected, face_count, seed + 300000)
            if gain2 > 0:
                selected, face_count = complete_1flip(selected, face_count)
                cnt = count_sel(selected)
                if cnt > best_count:
                    best_selected = selected.copy()
                    best_count = cnt

        if time.time() < deadline - 15.0 and cnt < 304:
            selected, face_count, cnt = local_search(selected, face_count, seed + 200000, max_iters=30)
            if cnt > best_count:
                best_selected = selected.copy()
                best_count = cnt

        if best_count >= 304:
            break

    # Final polish: run local search on best solution with more iterations
    if best_selected is not None and time.time() < deadline - 8.0:
        # Rebuild face_count from best_selected
        fc = np.zeros(num_faces, dtype=np.int32)
        for ei in np.where(best_selected)[0]:
            fc[edge_faces_arr[ei]] += 1
        best_selected, fc, final_cnt = local_search(best_selected, fc, 999999, max_iters=100)
        if final_cnt > best_count:
            best_count = final_cnt

    out = []
    for ei in range(m):
        if best_selected[ei]:
            x, y = edges[ei]
            out.append([x, y])

    return out
# EVOLVE-BLOCK-END
