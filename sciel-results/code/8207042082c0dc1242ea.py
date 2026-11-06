"""Strong edge colouring via bitmask-accelerated local search.

The previous approach used set-based conflict checks which were slow. Here we
use bitmask operations (Python ints) for O(1) conflict checks, enabling far more
local search iterations within the time budget.

Strategy:
1. Generate instances and build conflict structure.
2. Multiple randomized first-fit starts.
3. Local search with:
   - Merge passes (try to fold colour classes together)
   - Single-edge moves (move one edge to a different colour)
   - Two-swap moves (swap colours of two edges)
   - All using bitmask conflict checks for speed
4. Reindex colours to be contiguous.
"""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        # Build conflict bitmask for each edge
        conf = [0] * m
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | (adj.get(u, set()) | adj.get(v, set()))
            for j in range(i + 1, m):
                a, b = edges[j]
                if a in near or b in near:
                    conf[i] |= (1 << j)
                    conf[j] |= (1 << i)
        return conf

    def greedy_first_fit(order, conf, m):
        """First-fit greedy colouring in given order. Returns colour array."""
        col = [-1] * m
        # For each colour, maintain a bitmask of edges using that colour
        colour_masks = {}
        for i in order:
            taken = 0
            # Check which colours are blocked
            for c, mask in colour_masks.items():
                if conf[i] & mask:
                    taken |= (1 << c)
            c = 0
            while taken & (1 << c):
                c += 1
            col[i] = c
            colour_masks[c] = colour_masks.get(c, 0) | (1 << i)
        return col

    def local_search(col, conf, m, deadline, rng):
        """Local search to reduce number of colours."""
        col = list(col)
        best_k = max(col) + 1 if col else 0
        best_col = list(col)
        
        def get_k():
            return max(col) + 1 if col else 0
        
        def build_colour_masks():
            masks = {}
            for i, c in enumerate(col):
                if c >= 0:
                    masks[c] = masks.get(c, 0) | (1 << i)
            return masks
        
        def can_move_edge(e, target_c, colour_masks):
            """Check if edge e can be assigned colour target_c."""
            if target_c not in colour_masks:
                return True
            return (conf[e] & colour_masks[target_c]) == 0
        
        def try_merge(colour_masks):
            """Try to merge two colour classes. Returns (success, new_masks)."""
            cols = sorted(colour_masks.keys())
            for i in range(len(cols)):
                c1 = cols[i]
                for j in range(i + 1, len(cols)):
                    c2 = cols[j]
                    if c2 not in colour_masks:
                        continue
                    # Check if all edges in c2 are compatible with c1
                    ok = True
                    mask1 = colour_masks[c1]
                    for e in range(m):
                        if (colour_masks[c2] >> e) & 1:
                            if conf[e] & mask1:
                                ok = False
                                break
                    if ok:
                        # Merge c2 into c1
                        for e in range(m):
                            if (colour_masks[c2] >> e) & 1:
                                col[e] = c1
                        colour_masks[c1] = mask1 | colour_masks[c2]
                        del colour_masks[c2]
                        return True, colour_masks
            return False, colour_masks
        
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 500:
            improved = False
            guard += 1
            colour_masks = build_colour_masks()
            
            # Merge pass
            merged, colour_masks = try_merge(colour_masks)
            if merged:
                improved = True
                k = get_k()
                if k < best_k:
                    best_k = k
                    best_col = list(col)
                continue
            
            # Single-edge moves
            moves_made = 0
            cols_list = sorted(colour_masks.keys())
            if len(cols_list) >= 2:
                for _ in range(200):
                    if time.time() >= deadline:
                        break
                    e = rng.randrange(m)
                    c = col[e]
                    # Try to move to another colour
                    other_cols = [c2 for c2 in cols_list if c2 != c and c2 in colour_masks]
                    if not other_cols:
                        continue
                    target = rng.choice(other_cols)
                    if can_move_edge(e, target, colour_masks):
                        # Remove from old colour
                        colour_masks[c] &= ~(1 << e)
                        if colour_masks[c] == 0:
                            del colour_masks[c]
                        # Add to new colour
                        colour_masks[target] = colour_masks.get(target, 0) | (1 << e)
                        col[e] = target
                        moves_made += 1
                        k = get_k()
                        if k < best_k:
                            best_k = k
                            best_col = list(col)
            
            if moves_made > 0:
                improved = True
                continue
            
            # Two-swap moves
            swaps_made = 0
            for _ in range(100):
                if time.time() >= deadline:
                    break
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2:
                    continue
                c1, c2 = col[e1], col[e2]
                if c1 == c2:
                    continue
                # Check if swapping is valid
                # e1 must be compatible with all edges in c2 (except e2)
                # e2 must be compatible with all edges in c1 (except e1)
                valid = True
                mask_c2 = colour_masks.get(c2, 0) & ~(1 << e2)
                if conf[e1] & mask_c2:
                    valid = False
                if valid:
                    mask_c1 = colour_masks.get(c1, 0) & ~(1 << e1)
                    if conf[e2] & mask_c1:
                        valid = False
                if not valid:
                    continue
                # Perform swap
                colour_masks[c1] = (colour_masks.get(c1, 0) & ~(1 << e1)) | (1 << e2)
                colour_masks[c2] = (colour_masks.get(c2, 0) & ~(1 << e2)) | (1 << e1)
                if colour_masks[c1] == 0:
                    del colour_masks[c1]
                if colour_masks[c2] == 0:
                    del colour_masks[c2]
                col[e1], col[e2] = c2, c1
                swaps_made += 1
                k = get_k()
                if k < best_k:
                    best_k = k
                    best_col = list(col)
            
            if swaps_made > 0:
                improved = True
        
        return best_col, best_k

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue
        
        conf = build_conflict(edges)
        cdeg = [(conf[i] & ((1 << m) - 1)).bit_count() for i in range(m)]
        order_base = list(range(m))
        deg_order = sorted(order_base, key=lambda i: -cdeg[i])
        asc_order = sorted(order_base, key=lambda i: cdeg[i])
        
        best_col = None
        best_k = m + 1
        rng = Random(999 + seed)
        
        # Deterministic starts
        for order in [order_base, deg_order, asc_order]:
            col = greedy_first_fit(order, conf, m)
            k = max(col) + 1 if col else 0
            if k < best_k:
                best_k = k
                best_col = list(col)
        
        # Randomized restarts
        remaining_instances = 3 - idx
        # Allocate time: give more time to later instances since they're harder
        time_per_instance = (DEADLINE - 25.0) / 3.0
        restart_deadline = t0 + time_per_instance * (idx + 1) - 5.0  # reserve 5s for local search
        
        num_restarts = 0
        while time.time() < restart_deadline:
            order = list(order_base)
            rng.shuffle(order)
            col = greedy_first_fit(order, conf, m)
            k = max(col) + 1 if col else 0
            if k < best_k:
                best_k = k
                best_col = list(col)
            num_restarts += 1
        
        # Local search on best solution
        if best_col is not None:
            ls_deadline = min(restart_deadline + 15.0, DEADLINE - 10.0)
            best_col, best_k = local_search(best_col, conf, m, ls_deadline, rng)
        
        # Reindex colours to be contiguous
        if best_col:
            present = sorted(set(c for c in best_col if c >= 0))
            remap = {c: i for i, c in enumerate(present)}
            best_col = [remap[c] for c in best_col]
        
        results.append(best_col)
    
    return results
