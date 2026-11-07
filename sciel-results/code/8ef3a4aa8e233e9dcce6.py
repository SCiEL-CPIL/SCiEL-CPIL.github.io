import time
import random
import math
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin for 90s limit

    n = P['n']
    edges = P['edges']
    m = len(edges)

    # Build adjacency bitmask and edge set
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))

    full_mask = (1 << n) - 1

    def mask_to_list(mask):
        result = []
        mm = mask
        while mm:
            lsb = mm & -mm
            result.append(lsb.bit_length() - 1)
            mm ^= lsb
        return result

    def expand_biclique(u, v, rem_adj):
        """Expand biclique starting with (u,v) to maximal size."""
        L_mask = 1 << u
        R_mask = 1 << v

        changed = True
        while changed:
            changed = False

            # Expand L: candidates connected to all in R
            common_L = full_mask
            mr = R_mask
            while mr:
                lsb = mr & -mr
                y = lsb.bit_length() - 1
                mr ^= lsb
                common_L &= rem_adj[y]
            new_L = common_L & ~L_mask & ~R_mask
            if new_L:
                L_mask |= new_L
                changed = True

            # Expand R: candidates connected to all in L
            common_R = full_mask
            ml = L_mask
            while ml:
                lsb = ml & -ml
                x = lsb.bit_length() - 1
                ml ^= lsb
                common_R &= rem_adj[x]
            new_R = common_R & ~R_mask & ~L_mask
            if new_R:
                R_mask |= new_R
                changed = True

        return L_mask, R_mask

    def greedy_partition(seed, max_candidates=8):
        """Greedy with multiple candidate evaluation per step."""
        random.seed(seed)
        rem_adj = adj[:]
        bicliques = []
        edges_left = m

        while edges_left > 0:
            if time.time() > deadline:
                return None

            # Collect candidate starting edges
            active_vertices = [u for u in range(n) if rem_adj[u] != 0]
            if not active_vertices:
                break

            # Pick several random starting edges
            candidates = []
            for _ in range(min(30, len(active_vertices) * 4)):
                u = random.choice(active_vertices)
                mask = rem_adj[u]
                if mask == 0:
                    continue
                # Get random neighbor
                mm = mask
                count = mm.bit_count()
                if count == 0:
                    continue
                idx = random.randint(0, count - 1)
                v = None
                temp = mm
                for i in range(count):
                    lsb = temp & -temp
                    if i == idx:
                        v = lsb.bit_length() - 1
                        break
                    temp ^= lsb
                if v is not None:
                    candidates.append((u, v))

            if not candidates:
                break

            # Evaluate each candidate: expand and measure "quality"
            best_L = None
            best_R = None
            best_score = -1
            best_edges_covered = 0

            for u, v in candidates[:max_candidates]:
                if not (rem_adj[u] & (1 << v)):
                    continue

                L_mask, R_mask = expand_biclique(u, v, rem_adj)
                edges_covered = L_mask.bit_count() * R_mask.bit_count()

                # Score: prefer larger bicliques
                score = edges_covered
                if score > best_score:
                    best_score = score
                    best_edges_covered = edges_covered
                    best_L = L_mask
                    best_R = R_mask

            if best_L is None:
                # Fallback: pick first valid candidate
                for u, v in candidates:
                    if rem_adj[u] & (1 << v):
                        best_L, best_R = expand_biclique(u, v, rem_adj)
                        best_edges_covered = best_L.bit_count() * best_R.bit_count()
                        break
                if best_L is None:
                    break

            # Remove edges between L and R
            ml = best_L
            while ml:
                lsb = ml & -ml
                u_cur = lsb.bit_length() - 1
                ml ^= lsb
                rem_adj[u_cur] &= ~best_R

            mr = best_R
            while mr:
                lsb = mr & -mr
                v_cur = lsb.bit_length() - 1
                mr ^= lsb
                rem_adj[v_cur] &= ~best_L

            edges_left -= best_edges_covered

            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])

        return bicliques

    def refine_partition(bicliques):
        """Try to remove redundant bicliques or merge adjacent ones."""
        if len(bicliques) <= 1:
            return bicliques

        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = bicliques[:]

            for i in range(len(current)):
                if time.time() > deadline:
                    return current

                # Check if biclique i is redundant
                L_i = set(current[i][0])
                R_i = set(current[i][1])

                # Get all cross-edges of biclique i
                cross_edges_i = set()
                for u in current[i][0]:
                    for v in current[i][1]:
                        if u < v:
                            cross_edges_i.add((u, v))
                        else:
                            cross_edges_i.add((v, u))

                # Check if all these edges are covered by other bicliques
                all_covered = True
                for (u, v) in cross_edges_i:
                    covered = False
                    for j in range(len(current)):
                        if j == i:
                            continue
                        L_j = set(current[j][0])
                        R_j = set(current[j][1])
                        # Check if (u,v) is a cross-edge in biclique j
                        if (u in L_j and v in R_j) or (u in R_j and v in L_j):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break

                if all_covered:
                    # Remove biclique i
                    current.pop(i)
                    improved = True
                    break  # Restart the loop since indices changed

        return current

    def try_local_search(bicliques):
        """Try to improve by moving vertices between bicliques."""
        if len(bicliques) <= 1:
            return bicliques

        current = bicliques[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                L_i = current[i][0][:]
                R_i = current[i][1][:]

                # Try moving a vertex from L_i to R_i or vice versa
                for u in L_i[:]:
                    # Check if moving u to R_i is valid
                    # u must be connected to all in R_i
                    valid = True
                    for v in R_i:
                        if not (adj[u] & (1 << v)):
                            valid = False
                            break
                    if valid:
                        # Check if this reduces total bicliques or improves structure
                        # For now, just try it and see if we can remove redundancy
                        new_bicliques = current[:]
                        new_bicliques[i] = [[x for x in L_i if x != u], R_i + [u]]
                        refined = refine_partition(new_bicliques)
                        if len(refined) < len(current):
                            current = refined
                            improved = True
                            break

                if improved:
                    break

        return current

    # Seed Solution: One biclique per vertex covering its outgoing edges
    seed_bicliques = []
    for u in range(n):
        right_side = [v for u2, v in edges if u2 == u]
        if right_side:
            seed_bicliques.append([[u], right_side])

    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Run multiple greedy restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        bicliques = greedy_partition(seed=r, max_candidates=8)
        if bicliques is None:
            continue

        # Refine
        bicliques = refine_partition(bicliques)

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            if best_count == 1:
                break

    # Try local search on the best solution
    if time.time() < deadline:
        improved = try_local_search(best_bicliques)
        if len(improved) < best_count:
            best_count = len(improved)
            best_bicliques = improved

    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])

    return result
