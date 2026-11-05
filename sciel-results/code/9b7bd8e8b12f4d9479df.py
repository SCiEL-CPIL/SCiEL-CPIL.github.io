import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110

    def has_k222(adj, n):
        """Check if adj (list of int bitmasks) contains K2,2,2.
        K2,2,2 exists iff there exist vertices u,v with >=4 common neighbors,
        and within those common neighbors, there exist x,y with >=2 common neighbors in that set."""
        for u in range(n):
            nu = adj[u]
            for v_idx in range(u + 1, n):
                nv = adj[v_idx]
                s_mask = nu & nv
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v_idx)
                if s_mask.bit_count() < 4:
                    continue
                # Need to find x, y in s_mask with |N(x) ∩ N(y) ∩ s_mask| >= 2
                x_mask = s_mask
                while x_mask:
                    x_bit = x_mask & -x_mask
                    x_idx = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    nx = adj[x_idx]
                    # y must be in s_mask, y > x_idx to avoid duplicates
                    y_mask = s_mask & ~((1 << (x_idx + 1)) - 1)
                    while y_mask:
                        y_bit = y_mask & -y_mask
                        y_idx = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        ny = adj[y_idx]
                        if (nx & ny & s_mask).bit_count() >= 2:
                            return True
        return False

    def build_adj(colors, n):
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        return adj0, adj1

    def count_violations_fast(colors, n):
        """Returns 0 if no K2,2,2 in either color class, else 1."""
        adj0, adj1 = build_adj(colors, n)
        if has_k222(adj0, n):
            return 1
        if has_k222(adj1, n):
            return 1
        return 0

    def count_violations_detailed(colors, n):
        """Count total K2,2,2 subgraphs in both color classes."""
        adj0, adj1 = build_adj(colors, n)

        def count_k222(adj):
            v = 0
            for u in range(n):
                nu = adj[u]
                for v_idx in range(u + 1, n):
                    nv = adj[v_idx]
                    s_mask = nu & nv
                    s_mask &= ~(1 << u)
                    s_mask &= ~(1 << v_idx)
                    if s_mask.bit_count() < 4:
                        continue
                    x_mask = s_mask
                    while x_mask:
                        x_bit = x_mask & -x_mask
                        x_idx = x_bit.bit_length() - 1
                        x_mask ^= x_bit
                        nx = adj[x_idx]
                        y_mask = s_mask & ~((1 << (x_idx + 1)) - 1)
                        while y_mask:
                            y_bit = y_mask & -y_mask
                            y_idx = y_bit.bit_length() - 1
                            y_mask ^= y_bit
                            ny = adj[y_idx]
                            common = nx & ny & s_mask
                            if common.bit_count() >= 2:
                                v += 1
            return v

        return count_k222(adj0) + count_k222(adj1)

    # 1. Generate Paley(29)
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)

    best_n = 29
    best_colors = colors_29
    if count_violations_fast(colors_29, 29) != 0:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30
    n = 30
    n_edges = n * (n - 1) // 2

    # Edge index formula: for edge (i,j) with i<j in K_n:
    # idx = sum_{k=0}^{i-1} (n-1-k) + (j-i-1) = i*(n-1) - i*(i-1)//2 + j-i-1
    def edge_idx(i, j, nn):
        return i * (nn - 1) - i * (i - 1) // 2 + (j - i - 1)

    new_edge_indices = []
    for i in range(29):
        new_edge_indices.append(edge_idx(i, 29, n))

    best_viol_30 = float('inf')
    best_colors_30 = None

    # Spectral initialization: use Paley(29) structure to guide vertex 29's connections
    # Build Paley(29) adjacency as numpy for spectral analysis
    paley_adj = np.zeros((29, 29), dtype=np.float64)
    for i in range(29):
        for j in range(i + 1, 29):
            if (j - i) % 29 in squares:
                paley_adj[i, j] = 1
                paley_adj[j, i] = 1

    # Get top eigenvectors to understand structure
    eigenvalues, eigenvectors = np.linalg.eigh(paley_adj)

    # For vertex 29, use spectral info to initialize its edges
    # The top eigenvector components give "importance" weights
    top_evec = eigenvectors[:, -1]  # largest eigenvalue
    # Normalize to get a reference
    ref_pattern = top_evec.copy()
    # Sign pattern: positive connections where ref is large positive
    # We'll use this to bias the initial coloring for vertex 29

    max_restarts = 300
    rng = random.Random(42)

    for r in range(max_restarts):
        if time.time() > deadline - 8:
            break

        # Initialize: Paley(29) base + random or spectral-biased vertex 29 edges
        current_colors = colors_29.copy()

        if r < 50:
            # Spectral-biased initialization
            # Use the top eigenvector to decide connections
            for i in range(29):
                if ref_pattern[i] > 0:
                    current_colors.append(0)  # connect to color 0
                else:
                    current_colors.append(1)
            # Shuffle the new edges to the correct positions
            # Actually, the new edges are at specific indices, let's rebuild properly
            current_colors = [0] * n_edges
            idx = 0
            for i in range(29):
                for j in range(i + 1, 29):
                    current_colors[idx] = colors_29[idx]
                    idx += 1
            # Now fill in vertex 29 edges
            for i in range(29):
                idx29 = edge_idx(i, 29, n)
                if ref_pattern[i] > 0:
                    current_colors[idx29] = 0
                else:
                    current_colors[idx29] = 1
        else:
            # Random initialization for new edges
            current_colors = colors_29.copy()
            # Need to insert new edges at correct positions
            # Rebuild properly
            current_colors = [0] * n_edges
            idx = 0
            for i in range(29):
                for j in range(i + 1, 29):
                    current_colors[idx] = colors_29[idx]
                    idx += 1
            for i in range(29):
                idx29 = edge_idx(i, 29, n)
                current_colors[idx29] = rng.randint(0, 1)

        current_viol = count_violations_fast(current_colors, n)
        if current_viol == 0:
            return {'n': n, 'colors': current_colors}

        current_detailed = count_violations_detailed(current_colors, n)

        if current_detailed < best_viol_30:
            best_viol_30 = current_detailed
            best_colors_30 = current_colors.copy()

        # Local search
        max_steps = 500
        stuck_count = 0
        no_improve = 0
        prev_detailed = current_detailed

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            # Choose which edge to flip
            # Bias towards new edges (vertex 29 connections)
            if rng.random() < 0.6:
                edge_idx_to_flip = rng.choice(new_edge_indices)
            else:
                edge_idx_to_flip = rng.randint(0, n_edges - 1)

            old_val = current_colors[edge_idx_to_flip]
            current_colors[edge_idx_to_flip] = 1 - old_val

            new_viol = count_violations_fast(current_colors, n)
            if new_viol == 0:
                return {'n': n, 'colors': current_colors}

            new_detailed = count_violations_detailed(current_colors, n)

            if new_detailed < current_detailed:
                current_detailed = new_detailed
                stuck_count = 0
                no_improve = 0
                if new_detailed < best_viol_30:
                    best_viol_30 = new_detailed
                    best_colors_30 = current_colors.copy()
            else:
                # Revert
                current_colors[edge_idx_to_flip] = old_val
                stuck_count += 1
                no_improve += 1

                # Accept with small probability for diversity (simulated annealing-like)
                if rng.random() < 0.05 and new_detailed - current_detailed <= 2:
                    current_colors[edge_idx_to_flip] = 1 - old_val
                    current_detailed = new_detailed

                if stuck_count > 40:
                    # Jump: flip 2-3 edges
                    n_jumps = rng.randint(2, 4)
                    for _ in range(n_jumps):
                        if rng.random() < 0.5:
                            j = rng.choice(new_edge_indices)
                        else:
                            j = rng.randint(0, n_edges - 1)
                        current_colors[j] = 1 - current_colors[j]
                    current_detailed = count_violations_detailed(current_colors, n)
                    if current_detailed < best_viol_30:
                        best_viol_30 = current_detailed
                        best_colors_30 = current_colors.copy()
                    stuck_count = 0

            if no_improve > 200:
                break

    # If we found a valid 30-vertex coloring, return it
    if best_colors_30 is not None and count_violations_fast(best_colors_30, 30) == 0:
        return {'n': 30, 'colors': best_colors_30}

    return {'n': best_n, 'colors': best_colors}
