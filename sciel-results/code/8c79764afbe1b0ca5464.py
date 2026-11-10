import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 16
    total_edges = 120

    # Edge indexing
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1

    # Precompute all Q3 subgraphs as frozensets of edge indices
    # Q3: 8 vertices, 12 edges. Vertices are 3-bit binary numbers.
    # Edges connect vertices differing in exactly one bit.
    q3_edge_sets = []
    seen = set()
    for verts in itertools.combinations(range(n), 8):
        # Try to embed Q3: pick 4 vertices for "0" layer, 4 for "1" layer
        # Actually, Q3 has 8 vertices. We need to find if these 8 vertices
        # contain a Q3 subgraph.
        # A Q3 subgraph on 8 specific vertices: we need to check all ways
        # to assign them to the 8 cube positions.
        # But that's 8! = 40320 per 8-set, too slow for all 12870 sets.
        # Instead: for each 8-set, check if the induced subgraph on those
        # 8 vertices contains Q3.
        # Q3 is a 3-regular bipartite graph with 8 vertices and 12 edges.
        # In the induced subgraph on 8 vertices, Q3 appears iff the induced
        # subgraph has at least 12 edges AND contains a specific structure.
        # Actually, we just need to check: does the set of edges among these
        # 8 vertices contain a Q3?
        # A Q3 on 8 vertices: the 12 edges form a specific pattern.
        # Let's enumerate: for each 8-set, get all 28 possible edges.
        # A Q3 is a specific 12-edge subgraph. We need to check if any
        # 12-edge subset of the induced edges forms a Q3.
        # Better: precompute all Q3 edge patterns on 8 labeled vertices.
        pass

    # Precompute Q3 patterns on 8 labeled vertices (0..7)
    # Q3 vertices: 000,001,010,011,100,101,110,111
    # Edges: pairs differing in one bit
    cube_edges = set()
    for a in range(8):
        for b in range(a+1, 8):
            diff = a ^ b
            if diff in (1, 2, 4):  # powers of 2
                cube_edges.add(frozenset((a, b)))
    # cube_edges has 12 edges

    # All ways to map 8 labeled positions to the 8 cube vertices
    # We generate all distinct edge sets (as frozensets of frozensets)
    q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        # perm[i] = which cube vertex label goes to position i
        # Edge between positions i,j exists if perm[i] and perm[j] are adjacent in cube
        edges = frozenset()
        for i in range(8):
            for j in range(i+1, 8):
                if frozenset((perm[i], perm[j])) in cube_edges:
                    edges.add(frozenset((i, j)))
        q3_patterns.add(edges)
    # q3_patterns: set of frozensets of frozensets, each with 12 edges
    q3_pattern_list = list(q3_patterns)
    num_q3_patterns = len(q3_pattern_list)

    # For each 8-vertex subset, precompute which Q3 patterns map to which global edges
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)

    # For each subset, store list of (pattern_edge_set, global_edge_mask)
    # where global_edge_mask is a bitmask over the 28 local edges
    # Actually, let's store: for each subset, a list of 12-edge global masks
    # that correspond to Q3s within that subset.
    subset_q3_masks = []  # list of lists of int (28-bit masks)
    for s in subsets_8:
        # s is tuple of 8 global vertices
        # local vertex i corresponds to global vertex s[i]
        # local edge (i,j) -> global edge edge_index[s[i]][s[j]]
        q3_masks_for_subset = []
        seen_masks = set()
        for pattern in q3_pattern_list:
            mask = 0
            for (i, j) in pattern:
                u_g = s[i]
                v_g = s[j]
                e_idx = edge_index[u_g][v_g]
                mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_subset.append(mask)
        subset_q3_masks.append(q3_masks_for_subset)

    # Precompute for each global edge, which subsets it belongs to
    # and its local position within that subset
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_bit = [[-1]*total_edges for _ in range(num_subsets)]

    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_g = s[i]
                v_g = s[j]
                e_idx = edge_index[u_g][v_g]
                # local bit position: we need a consistent mapping
                # Let's use a local edge index
                edge_to_subsets[e_idx].append(s_idx)
                local_bit[s_idx][e_idx] = (i, j)  # store local vertex pair

    # Helper: check if a graph (as bitmask) is Q3-free
    def is_q3_free(G):
        for s_idx in range(num_subsets):
            s = subsets_8[s_idx]
            # Get the mask of edges among these 8 vertices
            local_mask = 0
            for i in range(8):
                for j in range(i+1, 8):
                    u_g = s[i]
                    v_g = s[j]
                    e_idx = edge_index[u_g][v_g]
                    if (G >> e_idx) & 1:
                        local_mask |= (1 << (i*7 - i*(i+1)//2))  # this is wrong
            # Actually, let's just check each Q3 mask directly
            for q3_mask in subset_q3_masks[s_idx]:
                if (G & q3_mask) == q3_mask:
                    return False
        return True

    # Faster check: only check subsets that could potentially contain a new Q3
    def check_q3_fast(G, changed_edge=None):
        if changed_edge is None:
            return is_q3_free(G)
        # Only check subsets containing the changed edge
        for s_idx in edge_to_subsets[changed_edge]:
            for q3_mask in subset_q3_masks[s_idx]:
                if (G & q3_mask) == q3_mask:
                    return False
        return True

    best_G = 0
    best_count = 0

    def try_improve(G, count, temp, max_steps=200):
        nonlocal best_G, best_count
        G = G
        count = count
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            # Try adding a random missing edge
            missing = [e for e in range(total_edges) if not (G >> e) & 1]
            if not missing:
                break
            e_add = random.choice(missing)
            new_G = G | (1 << e_add)
            if check_q3_fast(new_G, e_add):
                G = new_G
                count += 1
                if count > best_count:
                    best_count = count
                    best_G = G
            else:
                # Try removing a random existing edge and adding another
                existing = [e for e in range(total_edges) if (G >> e) & 1]
                if not existing:
                    continue
                e_rem = random.choice(existing)
                e_add2 = random.choice(missing)
                if e_add2 == e_rem:
                    continue
                new_G2 = G | (1 << e_add2) & ~(1 << e_rem)
                if check_q3_fast(new_G2, e_add2) and check_q3_fast(new_G2, e_rem):
                    # Accept with some probability (simulated annealing)
                    if random.random() < 0.5:
                        G = new_G2
                        # count stays the same
        return G, count

    # Spectral random construction: try different probabilities
    for p in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            G = 0
            for e in range(total_edges):
                if random.random() < p:
                    G |= (1 << e)

            # Remove Q3s
            changed = True
            while changed:
                changed = False
                for s_idx in range(num_subsets):
                    if time.time() > deadline - 5:
                        break
                    for q3_mask in subset_q3_masks[s_idx]:
                        if (G & q3_mask) == q3_mask:
                            # Remove one random edge from this Q3
                            q3_edges = [e for e in range(total_edges) if (q3_mask >> e) & 1]
                            e_rm = random.choice(q3_edges)
                            G &= ~(1 << e_rm)
                            changed = True
                            break
                    if changed:
                        break
                if time.time() > deadline - 5:
                    break

            count = G.bit_count()
            if count > best_count:
                best_count = count
                best_G = G

            # Simulated annealing refinement
            temp = 1.0
            for sa_iter in range(5):
                if time.time() > deadline - 5:
                    break
                G, count = try_improve(G, count, temp, max_steps=100)
                temp *= 0.5

    # Final greedy pass: try to add any missing edge that doesn't create Q3
    for e in range(total_edges):
        if time.time() > deadline - 3:
            break
        if not (best_G >> e) & 1:
            new_G = best_G | (1 << e)
            if check_q3_fast(new_G, e):
                best_G = new_G
                best_count += 1

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])

    return result
