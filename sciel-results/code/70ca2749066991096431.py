import time
import random
import numpy as np

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Initial state: 8 disjoint K5s (100 edges)
    adj = np.zeros((N, N), dtype=np.uint8)
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj[u, v] = 1
                adj[v, u] = 1

    current_edges = int(np.sum(adj)) // 2
    best_edges = current_edges
    best_adj = adj.copy()

    def can_add_fast(adj_mat, u, v):
        """Vectorized check: does adding (u,v) create a C6?
        Need a path u-w-x-y-z-v with all distinct.
        """
        # Exclude u and v from intermediate nodes
        exclude = np.zeros(N, dtype=np.uint8)
        exclude[u] = 1
        exclude[v] = 1

        # Step 1: w in N(u) \ {v}
        w_mask = adj_mat[u].copy()
        w_mask[u] = 0
        w_mask[v] = 0

        if not np.any(w_mask):
            return True

        # Step 2: x in N(w) \ {u, v, w}
        # For each w, compute valid x
        # We'll track which (w,x) pairs are valid
        # x must be in N(w), not u, not v, not w itself
        # adj_mat[w] gives N(w); exclude u, v, and w itself
        # Since adj is symmetric and no self-loops, adj[w,w]=0 already
        x_candidates = adj_mat[w_mask].copy()  # shape (N, N): for each w, its neighbors
        x_candidates[:, u] = 0
        x_candidates[:, v] = 0
        # Exclude w itself from its own neighbor list (already 0 in diagonal)
        # But we need to zero out the diagonal for the w rows
        for i in range(N):
            if w_mask[i]:
                x_candidates[i, i] = 0

        if not np.any(x_candidates):
            return True

        # Step 3: y in N(x) \ {u, v, w, x}
        # For each valid (w,x) pair, find y in N(x) excluding {u,v,w,x}
        y_candidates = np.zeros_like(adj_mat)
        for w_idx in range(N):
            if not w_mask[w_idx]:
                continue
            for x_idx in range(N):
                if not x_candidates[w_idx, x_idx]:
                    continue
                y_mask = adj_mat[x_idx].copy()
                y_mask[u] = 0
                y_mask[v] = 0
                y_mask[w_idx] = 0
                y_mask[x_idx] = 0
                y_candidates[w_idx, :] |= y_mask  # Mark valid y for this (w,x)

        if not np.any(y_candidates):
            return True

        # Step 4: z in N(y) \ {u, v, w, x, y} AND z in N(v)
        # For each valid (w,x,y), check if N(y) intersects (N(v) \ {u,v,w,x,y})
        # Precompute N(v)
        nv = adj_mat[v].copy()
        nv[u] = 0
        nv[v] = 0

        # For each (w,x) pair that has valid y's:
        for w_idx in range(N):
            if not w_mask[w_idx]:
                continue
            for x_idx in range(N):
                if not x_candidates[w_idx, x_idx]:
                    continue
                # y's valid for this (w,x)
                y_valid = y_candidates[w_idx, :]
                y_valid[x_idx] = 0  # y != x (should already be, but safe)
                y_valid[w_idx] = 0  # y != w
                if not np.any(y_valid):
                    continue
                # For each valid y, check if N(y) has a z in N(v) excluding {u,v,w,x,y}
                for y_idx in range(N):
                    if not y_valid[y_idx]:
                        continue
                    z_mask = adj_mat[y_idx].copy()
                    z_mask[u] = 0
                    z_mask[v] = 0
                    z_mask[w_idx] = 0
                    z_mask[x_idx] = 0
                    z_mask[y_idx] = 0
                    # Check intersection with N(v)
                    if np.any(z_mask & nv):
                        return False
        return True

    # Local search with kicks
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if adj[p[0], p[1]] == 0]

    while time.time() < deadline:
        random.shuffle(missing_edges)
        added_in_pass = 0
        new_missing = []

        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if can_add_fast(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()

        missing_edges = new_missing

        if added_in_pass == 0:
            # Kick: remove 2-5 random edges
            kick_size = random.randint(2, 5)
            existing = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j]]
            random.shuffle(existing)
            for _ in range(min(kick_size, len(existing))):
                u, v = existing.pop()
                adj[u, v] = 0
                adj[v, u] = 0
                current_edges -= 1
                missing_edges.append((u, v))

    # Final polish pass: try to add any remaining missing edges
    if time.time() < deadline - 5:
        final_missing = [(i, j) for i in range(N) for j in range(i + 1, N) if best_adj[i, j] == 0]
        random.shuffle(final_missing)
        polish_adj = best_adj.copy()
        polish_edges = best_edges
        for u, v in final_missing:
            if time.time() > deadline - 2:
                break
            if can_add_fast(polish_adj, u, v):
                polish_adj[u, v] = 1
                polish_adj[v, u] = 1
                polish_edges += 1
        if polish_edges > best_edges:
            best_edges = polish_edges
            best_adj = polish_adj

    # Return edges as list of [u, v]
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v]:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
