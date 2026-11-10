"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute edge indices
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105

    def get_adj_deg(mask):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return adj, deg

    def is_valid_mask(mask):
        """Fast validity check using degree-class precomputation."""
        adj, deg = get_adj_deg(mask)
        # For each degree value, precompute the set of vertices with that degree
        deg_to_verts = {}
        for v in range(n):
            d = deg[v]
            if d not in deg_to_verts:
                deg_to_verts[d] = set()
            deg_to_verts[d].add(v)

        for i in range(E):
            if not (mask & (1 << i)):
                continue
            u, v = edge_list[i]
            # Neighbors of u (excluding v) and neighbors of v (excluding u)
            ns = adj[u] - {v}
            nt = adj[v] - {u}
            # For each s in ns, check if any t in nt has same degree as s and t != s
            for s in ns:
                ds = deg[s]
                # Check if any vertex in nt has degree ds and is not s
                same_deg = deg_to_verts.get(ds, set())
                # We need t in nt, t in same_deg, t != s
                # This is equivalent to: (nt & same_deg) - {s} is non-empty
                if (nt & same_deg) - {s}:
                    return False
        return True

    # Build initial graph: K_{7,8}
    A = set(range(7))
    B = set(range(7, 15))

    init_mask = 0
    for a in A:
        for b in B:
            init_mask |= (1 << edge_to_idx[(a, b)])

    # Check if K_{7,8} is valid
    if is_valid_mask(init_mask):
        best_mask = init_mask
        best_count = bin(init_mask).count('1')
    else:
        # Greedy construction from empty
        best_mask = 0
        best_count = 0
        for i in range(E):
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1

    # Phase 1: Greedy edge addition (multiple passes)
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break

    # Phase 2: Local search with 1-for-1, 1-for-2, 2-for-3 swaps
    rng = random.Random(42)
    for iteration in range(3000):
        if time.time() > deadline - 8:
            break

        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]

        if not present:
            break

        improved = False

        # Try single add
        if absent:
            for i in absent:
                new_mask = best_mask | (1 << i)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break

        if improved:
            continue

        # Try 1-for-2 swap (remove 1, add 2)
        if absent and len(absent) >= 2 and present:
            attempts = min(50, max(5, len(absent) * len(present) // 10))
            for _ in range(attempts):
                if time.time() > deadline - 8:
                    break
                ri = rng.choice(present)
                a1, a2 = rng.sample(absent, 2)
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                new_count = bin(new_mask).count('1')
                if new_count > best_count:
                    if is_valid_mask(new_mask):
                        best_mask = new_mask
                        best_count = new_count
                        improved = True
                        break

        if improved:
            continue

        # Try 2-for-3 swap (remove 2, add 3)
        if len(absent) >= 3 and len(present) >= 2:
            attempts = min(30, max(3, len(absent) * len(present) // 20))
            for _ in range(attempts):
                if time.time() > deadline - 8:
                    break
                r1, r2 = rng.sample(present, 2)
                a1, a2, a3 = rng.sample(absent, 3)
                new_mask = (best_mask & ~(1 << r1) & ~(1 << r2)) | (1 << a1) | (1 << a2) | (1 << a3)
                new_count = bin(new_mask).count('1')
                if new_count > best_count:
                    if is_valid_mask(new_mask):
                        best_mask = new_mask
                        best_count = new_count
                        improved = True
                        break

        if improved:
            continue

        # Periodic perturbation: remove few edges, re-add greedily
        if iteration % 100 == 50:
            new_mask = best_mask
            for _ in range(min(3, best_count)):
                ri = rng.choice(present)
                new_mask &= ~(1 << ri)
            # Re-add greedily
            for i in range(E):
                if new_mask & (1 << i):
                    continue
                try_mask = new_mask | (1 << i)
                if is_valid_mask(try_mask):
                    new_mask = try_mask
            new_count = bin(new_mask).count('1')
            if new_count > best_count:
                best_mask = new_mask
                best_count = new_count

    # Phase 3: Final polish - try all single additions and 1-for-2 swaps exhaustively
    if time.time() < deadline - 3:
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]

        # Exhaustive single adds
        for i in absent:
            if time.time() > deadline - 3:
                break
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1

        # Exhaustive 1-for-2 (if time permits)
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        if absent and len(absent) >= 2 and present:
            for ri in present:
                if time.time() > deadline - 1:
                    break
                for a1 in absent:
                    if time.time() > deadline - 1:
                        break
                    for a2 in absent:
                        if a2 <= a1:
                            continue
                        if time.time() > deadline - 1:
                            break
                        new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                        new_count = bin(new_mask).count('1')
                        if new_count > best_count:
                            if is_valid_mask(new_mask):
                                best_mask = new_mask
                                best_count = new_count

    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
