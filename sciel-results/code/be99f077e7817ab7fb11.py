import time
import random
import itertools

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj_mask[a] >> b) & 1):
                continue
            common = adj_mask[a] & adj_mask[b]
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True


def _is_k4_free_after_add(adj_mask, n, u, v):
    """Fast check: after adding edge (u,v), is graph still K4-free?"""
    common = adj_mask[u] & adj_mask[v]
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        if adj_mask[c] & common:
            return False
        c_bits ^= lb
    return True


def _is_triangle_free_mask(adj_mask, subset_mask):
    """Check if the set of vertices in subset_mask induces a triangle-free graph."""
    v_bits = subset_mask
    while v_bits:
        lb = v_bits & (-v_bits)
        u = lb.bit_length() - 1
        v_bits ^= lb
        N_u = adj_mask[u] & subset_mask
        w_bits = N_u
        while w_bits:
            lb2 = w_bits & (-w_bits)
            w = lb2.bit_length() - 1
            w_bits ^= lb2
            if adj_mask[w] & N_u:
                return False
    return True


def _max_tf_size_exact(adj_mask, n, timeout=0.5):
    """Find the largest triangle-free induced subgraph size using branch and bound."""
    start_time = time.time()

    # Try sizes from n down to 1
    for k in range(n, 0, -1):
        if time.time() - start_time > timeout:
            break
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_triangle_free_mask(adj_mask, mask):
                return k
    return 1


def _find_large_tf_set(adj_mask, n, target_size):
    """Find a triangle-free set of size target_size. Returns bitmask or None."""
    for subset in itertools.combinations(range(n), target_size):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_triangle_free_mask(adj_mask, mask):
            return mask
    return None


def _build_k666_mask(n=18):
    """Build K6,6,6 adjacency mask. Parts: [0-5], [6-11], [12-17]."""
    adj = [0] * n
    for a in range(6):
        for b in range(6, 12):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        for b in range(12, 18):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    for a in range(6, 12):
        for b in range(12, 18):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    return adj


def _intra_part_edges(n=18):
    """All edges within the three parts of K6,6,6."""
    edges = []
    for start in range(0, n, 6):
        for i in range(6):
            for j in range(i + 1, 6):
                edges.append((start + i, start + j))
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    n = 18

    def time_ok():
        return time.time() < deadline

    def evaluate(adj_mask):
        if not _is_k4_free_mask(adj_mask, n):
            return None
        return _max_tf_size_exact(adj_mask, n, timeout=1.5)

    # Start with K6,6,6 (provable score <= 12)
    best_adj = _build_k666_mask(n)
    best_score = evaluate(best_adj)
    if best_score is None:
        best_score = 12  # Fallback: K6,6,6 is provably K4-free with score <= 12

    # Phase 1: Greedy - add intra-part edges to reduce score
    intra_edges = _intra_part_edges(n)
    random.seed(42)
    random.shuffle(intra_edges)

    for u, v in intra_edges:
        if not time_ok():
            break
        if (best_adj[u] >> v) & 1:
            continue
        # Try adding
        best_adj[u] |= (1 << v)
        best_adj[v] |= (1 << u)
        if _is_k4_free_after_add(best_adj, n, u, v):
            new_score = evaluate(best_adj)
            if new_score is not None and new_score < best_score:
                best_score = new_score
            else:
                best_adj[u] &= ~(1 << v)
                best_adj[v] &= ~(1 << u)
        else:
            best_adj[u] &= ~(1 << v)
            best_adj[v] &= ~(1 << u)

    # Phase 2: Multi-start local search from K6,6,6 variants
    best_adj = _build_k666_mask(n)
    best_score = evaluate(best_adj) or 12

    for restart in range(30):
        if not time_ok():
            break

        # Create a perturbed K6,6,6: add some random intra-part edges
        adj = _build_k666_mask(n)
        shuffled = intra_edges[:]
        random.shuffle(shuffled)
        num_add = random.randint(3, 12)
        for u, v in shuffled[:num_add]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_after_add(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

        score = evaluate(adj)
        if score is None:
            continue

        # Local search: try adding more intra-part edges
        improved = True
        while improved and time_ok():
            improved = False
            shuffled2 = intra_edges[:]
            random.shuffle(shuffled2)
            for u, v in shuffled2:
                if not time_ok():
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    new_score = evaluate(adj)
                    if new_score is not None and new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        if score < best_score:
            best_score = score
            best_adj = adj[:]

    # Phase 3: Targeted attack on specific triangle-free sets
    if time_ok():
        for _ in range(50):
            if not time_ok():
                break
            # Find a large triangle-free set in best_adj
            tf_mask = _find_large_tf_set(best_adj, n, best_score)
            if tf_mask is None:
                # Recompute score
                new_score = evaluate(best_adj)
                if new_score is not None and new_score < best_score:
                    best_score = new_score
                continue

            # Get vertices in the triangle-free set
            verts = []
            m = tf_mask
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                verts.append(v)
                m ^= lb

            # Try to add edges within this set to break it
            # Only add edges that don't create K4
            found_improvement = False
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    if not time_ok():
                        break
                    u, v = verts[i], verts[j]
                    if (best_adj[u] >> v) & 1:
                        continue
                    # Check if u and v are in the same part
                    part_u = u // 6
                    part_v = v // 6
                    if part_u != part_v:
                        continue
                    # Try adding
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    if _is_k4_free_after_add(best_adj, n, u, v):
                        new_score = evaluate(best_adj)
                        if new_score is not None and new_score < best_score:
                            best_score = new_score
                            found_improvement = True
                        else:
                            best_adj[u] &= ~(1 << v)
                            best_adj[v] &= ~(1 << u)
                    else:
                        best_adj[u] &= ~(1 << v)
                        best_adj[v] &= ~(1 << u)
                    if found_improvement:
                        break
                if found_improvement:
                    break

    # Phase 4: Final polish - try all possible single edge additions
    if time_ok():
        adj = best_adj[:]
        improved = True
        while improved and time_ok():
            improved = False
            for u, v in intra_edges:
                if not time_ok():
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    score = evaluate(adj)
                    if score is not None and score < best_score:
                        best_score = score
                        best_adj = adj[:]
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])

    return best_edges
