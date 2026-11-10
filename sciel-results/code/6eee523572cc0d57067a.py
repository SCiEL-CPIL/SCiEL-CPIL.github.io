import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))

    k4_edges_np = np.array(k4s, dtype=np.int32)
    n_k4 = len(k4s)

    def count_violations_fast(colours_np):
        """Fast violation count using numpy."""
        k4_colors = colours_np[k4_edges_np]
        # Count distinct colours per K4
        # Sort and count unique
        sorted_c = np.sort(k4_colors, axis=1)
        # unique count = 1 + number of positions where sorted[i] != sorted[i-1]
        diff = np.ones((n_k4, 6), dtype=np.int8)
        diff[:, 1:] = (sorted_c[:, 1:] != sorted_c[:, :-1]).astype(np.int8)
        counts = diff.sum(axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        c_np = np.asarray(colours, dtype=np.int32)
        return count_violations_fast(c_np) == 0

    # Generate multiple initial candidates
    candidates = []

    # Candidate 1: (i+j) % K for various K
    for K in range(13, 18):
        cand = [(i + j) % K for i in range(n) for j in range(i + 1, n)]
        candidates.append(cand)

    # Candidate 2: (i*j) % K
    for K in range(13, 18):
        cand = [(i * j) % K for i in range(n) for j in range(i + 1, n)]
        candidates.append(cand)

    # Candidate 3: (i + 2*j) % K
    for K in range(13, 18):
        cand = [(i + 2 * j) % K for i in range(n) for j in range(i + 1, n)]
        candidates.append(cand)

    # Candidate 4: (2*i + j) % K
    for K in range(13, 18):
        cand = [(2 * i + j) % K for i in range(n) for j in range(i + 1, n)]
        candidates.append(cand)

    # Candidate 5: Random with specific structure - group-based
    # Partition 16 vertices into 4 groups of 4
    # Colour edges within group with one set, between groups with another
    for K in range(13, 16):
        random.seed(K * 1000)
        cand = [random.randint(0, K - 1) for _ in range(120)]
        candidates.append(cand)

    # Candidate 6: More structured - use vertex labels
    # Color edge (i,j) based on (i mod 4, j mod 4) pattern
    for K in range(13, 16):
        random.seed(K * 2000 + 42)
        # Assign each vertex a "type"
        types = list(range(4)) * 4
        random.shuffle(types)
        # For each pair of types, assign a base colour
        type_colors = {}
        next_c = 0
        for t1 in range(4):
            for t2 in range(t1 + 1, 4):
                type_colors[(t1, t2)] = next_c % K
                next_c += 1
        cand = []
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = types[i], types[j]
                key = (min(t1, t2), max(t1, t2))
                cand.append(type_colors[key])
        candidates.append(cand)

    # Find best initial candidate
    best_cand = None
    best_violations = float('inf')
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        v = count_violations_fast(np.asarray(cand, dtype=np.int32))
        if v < best_violations:
            best_violations = v
            best_cand = cand

    current = best_cand if best_cand is not None else [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    current_v = best_violations

    # Local search function
    def local_search(colours, K, max_steps=3000):
        colours = list(colours)
        c_np = np.asarray(colours, dtype=np.int32)
        v = count_violations_fast(c_np)
        if v == 0:
            return colours, 0

        no_improve_count = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            if v == 0:
                break

            # Get violated K4s
            k4_colors = c_np[k4_edges_np]
            sorted_c = np.sort(k4_colors, axis=1)
            diff = np.ones((n_k4, 6), dtype=np.int8)
            diff[:, 1:] = (sorted_c[:, 1:] != sorted_c[:, :-1]).astype(np.int8)
            counts = diff.sum(axis=1)
            violated = np.where(counts < 5)[0]
            if len(violated) == 0:
                v = 0
                break

            # Pick a random violated K4
            k4_idx = violated[random.randint(0, len(violated) - 1)]
            k4_edge_idxs = k4_edges_np[k4_idx]

            # Try changing one edge in the K4
            best_new_v = v
            best_edge = -1
            best_color = -1

            # Try all 6 edges, for each try a few random colours + the K colours
            for eidx in k4_edge_idxs:
                old_c = colours[eidx]
                # Try a subset of colours: current + random + all K if K is small
                if K <= 15:
                    trial_colors = list(range(K))
                else:
                    trial_colors = [old_c] + [random.randint(0, K - 1) for _ in range(10)]

                for new_c in trial_colors:
                    if new_c == old_c:
                        continue
                    colours[eidx] = new_c
                    c_np[eidx] = new_c
                    new_v = count_violations_fast(c_np)
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_edge = eidx
                        best_color = new_c
                    colours[eidx] = old_c
                    c_np[eidx] = old_c

            if best_edge >= 0:
                colours[best_edge] = best_color
                c_np[best_edge] = best_color
                v = best_new_v
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Random perturbation
                    for _ in range(3):
                        eidx = random.randint(0, 119)
                        colours[eidx] = random.randint(0, K - 1)
                        c_np[eidx] = colours[eidx]
                    v = count_violations_fast(c_np)
                    no_improve_count = 0

        return colours, v

    # Try to reduce colour count
    for target_K in range(13, 17):
        if time.time() > deadline - 5:
            break

        # First, try to merge colours in current solution if it uses more than target_K
        current_set = set(current)
        if len(current_set) <= target_K:
            # Already using at most target_K colours, just verify and continue
            if is_valid(current):
                continue
        else:
            # Need to reduce to target_K colours
            # Try merging: pick two colours and merge them
            colors_list = sorted(current_set)
            if len(colors_list) <= target_K:
                continue

            merged = False
            # Try all pairs of colours to merge
            for i in range(len(colors_list)):
                for j in range(i + 1, len(colors_list)):
                    if time.time() > deadline - 5:
                        break
                    c1 = colors_list[i]
                    c2 = colors_list[j]
                    new_colours = [c2 if c == c1 else c for c in current]
                    if is_valid(new_colours):
                        current = new_colours
                        merged = True
                        break
                if merged:
                    break

            if not merged:
                # Map to target_K colours
                mapping = {}
                next_id = 0
                init_colours = []
                for c in current:
                    if c not in mapping:
                        if next_id >= target_K:
                            # Assign to existing
                            mapping[c] = random.randint(0, target_K - 1)
                        else:
                            mapping[c] = next_id
                            next_id += 1
                    init_colours.append(mapping[c])

                if not is_valid(init_colours):
                    # Local search
                    best_colours, best_v = local_search(init_colours, target_K)
                    if best_v == 0:
                        current = best_colours
                    else:
                        # Try random restarts
                        success = False
                        for restart in range(20):
                            if time.time() > deadline - 5:
                                break
                            rand_colours = [random.randint(0, target_K - 1) for _ in range(120)]
                            rand_colours, v = local_search(rand_colours, target_K)
                            if v == 0:
                                current = rand_colours
                                success = True
                                break
                        if not success:
                            break

    # Final validation
    if not is_valid(current):
        # Fallback to known valid solution
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed
        else:
            # Try a few random solutions
            for _ in range(50):
                if time.time() > deadline:
                    break
                rand = [random.randint(0, 15) for _ in range(120)]
                if is_valid(rand):
                    current = rand
                    break

    return current
