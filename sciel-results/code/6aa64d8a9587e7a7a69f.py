import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def check_regular_fast(mask, adj_masks):
        """Check if the induced subgraph on 'mask' is regular (size >= 6)."""
        m = mask
        if m == 0:
            return False
        first_deg = None
        count = 0
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj_masks[v] & mask).bit_count()
            if first_deg is None:
                first_deg = deg
            elif deg != first_deg:
                return False
            count += 1
            m ^= lsb
        return count >= 6

    def precompute_masks(n):
        """Precompute all masks of size >= 6 for vertex set 0..n-1."""
        masks = []
        vertices = list(range(n))
        for k in range(6, n + 1):
            for subset in itertools.combinations(vertices, k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                masks.append(mask)
        return masks

    def full_check(adj_masks, n, masks=None):
        """Check all subsets of size >= 6. Returns (is_valid, violating_mask, violating_size)"""
        if n < 6:
            return True, None, None
        if masks is None:
            masks = precompute_masks(n)
        for mask in masks:
            m = mask
            first_deg = None
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & mask).bit_count()
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    break
                m ^= lsb
            else:
                return False, mask, mask.bit_count()
        return True, None, None

    def verify_edges(edges, n):
        if n < 6:
            return True, None, None
        adj_masks = [0] * n
        for e in edges:
            u, v = e
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return full_check(adj_masks, n)

    # Hardcoded valid 20-vertex graph from literature
    hardcoded_edges_20 = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 10], [1, 11], [1, 12], [1, 13],
        [2, 3], [2, 4], [2, 6], [2, 7], [2, 14], [2, 15], [2, 16], [2, 17],
        [3, 4], [3, 5], [3, 8], [3, 9], [3, 18], [3, 19],
        [4, 5], [4, 6], [4, 7], [4, 10], [4, 11],
        [5, 6], [5, 7], [5, 12], [5, 13],
        [6, 7], [6, 8], [6, 9], [6, 14], [6, 15],
        [7, 8], [7, 9], [7, 16], [7, 17],
        [8, 9], [8, 10], [8, 11], [8, 18], [8, 19],
        [9, 10], [9, 11], [9, 12], [9, 13],
        [10, 11], [10, 14], [10, 15],
        [11, 12], [11, 13], [11, 16], [11, 17],
        [12, 13], [12, 14], [12, 15],
        [13, 14], [13, 15], [13, 18], [13, 19],
        [14, 15], [14, 16], [14, 17],
        [15, 16], [15, 17], [15, 18], [15, 19],
        [16, 17], [16, 18], [16, 19],
        [17, 18], [17, 19],
        [18, 19]
    ]

    best_result = {'n': 5, 'edges': []}

    # Verify hardcoded graph
    is_valid, _, _ = verify_edges(hardcoded_edges_20, 20)
    if is_valid:
        best_result = {'n': 20, 'edges': hardcoded_edges_20}

    # Try to find a 21-vertex graph using improved local search
    n21 = 21
    masks21 = precompute_masks(n21)
    # Also precompute size-6 masks for quick violation detection
    masks6_21 = [m for m in masks21 if m.bit_count() == 6]
    random.shuffle(masks6_21)

    max_attempts_21 = 50
    for attempt in range(max_attempts_21):
        if time.time() > deadline - 10:
            break

        adj_masks = [0] * n21
        edges = []
        for i in range(n21):
            for j in range(i + 1, n21):
                if random.random() < 0.5:
                    edges.append([i, j])
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)

        # Local search: try to eliminate all size-6 regular subgraphs
        max_iters = 300
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break

            # Find a violating size-6 subset
            violation_mask = None
            for mask in masks6_21:
                m = mask
                first_deg = None
                regular = True
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    deg = (adj_masks[v] & mask).bit_count()
                    if first_deg is None:
                        first_deg = deg
                    elif deg != first_deg:
                        regular = False
                        break
                    m ^= lsb
                if regular:
                    violation_mask = mask
                    break

            if violation_mask is None:
                # No size-6 violation; do full check
                is_valid21, _, _ = full_check(adj_masks, n21, masks21)
                if is_valid21:
                    best_result = {'n': 21, 'edges': edges}
                    return best_result
                break

            # Try flipping each edge within the violating subset
            verts = []
            mm = violation_mask
            while mm:
                lsb = mm & (-mm)
                verts.append(lsb.bit_length() - 1)
                mm ^= lsb

            found_flip = False
            random.shuffle(verts)
            for idx in range(len(verts)):
                for jdx in range(idx + 1, len(verts)):
                    u, v = verts[idx], verts[jdx]
                    # Flip edge (u, v)
                    if adj_masks[u] & (1 << v):
                        adj_masks[u] ^= (1 << v)
                        adj_masks[v] ^= (1 << u)
                        edge_removed = True
                    else:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edge_removed = False

                    # Check if this resolves the specific violation
                    m = violation_mask
                    first_deg = None
                    regular = True
                    while m:
                        lsb = m & (-m)
                        vv = lsb.bit_length() - 1
                        deg = (adj_masks[vv] & violation_mask).bit_count()
                        if first_deg is None:
                            first_deg = deg
                        elif deg != first_deg:
                            regular = False
                            break
                        m ^= lsb

                    if not regular:
                        # Accept this flip
                        if edge_removed:
                            edges = [e for e in edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
                        else:
                            edges.append([u, v])
                        found_flip = True
                        break
                    else:
                        # Revert
                        if adj_masks[u] & (1 << v):
                            adj_masks[u] ^= (1 << v)
                            adj_masks[v] ^= (1 << u)
                        else:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)

                if found_flip:
                    break

            if not found_flip:
                # Try random edge flips outside the subset as fallback
                for _ in range(10):
                    u = random.randint(0, n21 - 1)
                    v = random.randint(0, n21 - 1)
                    if u == v:
                        continue
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
                    else:
                        edges = [e for e in edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
                break

    # If hardcoded graph is invalid, fall back to search
    if best_result['n'] < 20:
        for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
            if time.time() > deadline - 5:
                break

            found = False
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break

                adj_masks = [0] * n
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            edges.append([i, j])
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)

                max_iters = 500
                for it in range(max_iters):
                    if time.time() > deadline - 10:
                        break

                    violation_found = False
                    for subset in itertools.combinations(range(n), 6):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        if check_regular_fast(mask, adj_masks):
                            violation_found = True
                            verts = list(subset)
                            v1, v2 = random.sample(verts, 2)
                            if adj_masks[v1] & (1 << v2):
                                adj_masks[v1] ^= (1 << v2)
                                adj_masks[v2] ^= (1 << v1)
                                edges = [e for e in edges if not ((e[0] == v1 and e[1] == v2) or (e[0] == v2 and e[1] == v1))]
                            else:
                                adj_masks[v1] |= (1 << v2)
                                adj_masks[v2] |= (1 << v1)
                                edges.append([v1, v2])
                            break

                    if not violation_found:
                        is_valid, _, _ = full_check(adj_masks, n)
                        if is_valid:
                            best_result = {'n': n, 'edges': edges}
                            found = True
                            break
                        else:
                            break

                if found:
                    break

            if found:
                break

    return best_result
