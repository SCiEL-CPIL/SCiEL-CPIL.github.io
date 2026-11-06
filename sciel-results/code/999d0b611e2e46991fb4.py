def construct():
    import time
    start_time = time.time()
    DEADLINE = 145.0

    NEIGHBORS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def hex_dist(p):
        a, b = p
        return max(abs(a), abs(b), abs(a + b))

    # Candidate pool: hexagon of radius 5
    candidates = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if max(abs(a), abs(b), abs(a + b)) <= 5:
                candidates.append((a, b))

    point_set = set(candidates)
    cand_neighbors = {}
    for p in candidates:
        nbrs = []
        for da, db in NEIGHBORS:
            np = (p[0] + da, p[1] + db)
            if np in point_set:
                nbrs.append(np)
        cand_neighbors[p] = nbrs

    # Greedy center-out
    selected = [(0, 0)]
    selected_set = {(0, 0)}

    for _ in range(31):
        if time.time() > DEADLINE:
            break
        best_point = None
        best_score = -1
        best_tiebreak = float('inf')

        for p in candidates:
            if p in selected_set:
                continue
            edge_count = sum(1 for nbr in cand_neighbors[p] if nbr in selected_set)
            hd = hex_dist(p)
            if edge_count > best_score or (edge_count == best_score and hd < best_tiebreak):
                best_score = edge_count
                best_tiebreak = hd
                best_point = p

        if best_point is None:
            break
        selected.append(best_point)
        selected_set.add(best_point)

    # Local search: try swaps to improve edge count
    if time.time() < DEADLINE - 10:
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            for i in range(len(selected)):
                if time.time() > DEADLINE - 10:
                    break
                old_pt = selected[i]
                for new_pt in candidates:
                    if new_pt in selected_set:
                        continue
                    # Compute delta in edge count
                    lost = sum(1 for k in range(len(selected)) if k != i and selected[k] in cand_neighbors[old_pt])
                    gained = sum(1 for k in range(len(selected)) if k != i and selected[k] in cand_neighbors[new_pt])
                    if gained > lost:
                        selected[i] = new_pt
                        selected_set.discard(old_pt)
                        selected_set.add(new_pt)
                        improved = True
                        break
                if improved:
                    break

    return [[a, b] for a, b in selected[:32]]
