"""Combined approach: direct backtracking CSP with forward checking + local-search refinement."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges

    # Precompute edge index mapping
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1

    # Precompute all K4s and their edge indices
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    for v in itertools.combinations(range(n), 4):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i, j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(len(k4_edges))
        k4_edges.append(e_indices)

    num_k4s = len(k4_edges)

    # Convert to numpy arrays for speed
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)
    edge_to_k4_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4]

    # Precompute pair indices for K4 edge positions
    pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                 (1,2),(1,3),(1,4),(1,5),
                 (2,3),(2,4),(2,5),
                 (3,4),(3,5),
                 (4,5)]
    pair_a = np.array([p[0] for p in pairs_idx], dtype=np.int32)
    pair_b = np.array([p[1] for p in pairs_idx], dtype=np.int32)

    def count_violations_fast(colors_np):
        """Fast violation count using numpy."""
        k4_colours = colors_np[k4_edges_arr]  # shape (1820, 6)
        num_equal_pairs = np.zeros(num_k4s, dtype=np.int32)
        for pi in range(15):
            num_equal_pairs += (k4_colours[:, pair_a[pi]] == k4_colours[:, pair_b[pi]]).astype(np.int32)
        violations = np.sum(num_equal_pairs >= 2)
        return violations

    def compute_slack(colors_np):
        """Compute slack for each K4: number of equal pairs (0=valid with 6 distinct, 1=valid with 5 distinct, >=2=violation)."""
        k4_colours = colors_np[k4_edges_arr]
        slack = np.zeros(num_k4s, dtype=np.int32)
        for pi in range(15):
            slack += (k4_colours[:, pair_a[pi]] == k4_colours[:, pair_b[pi]]).astype(np.int32)
        return slack

    def is_valid_slack(slack_arr):
        return np.all(slack_arr < 2)

    # Base solution: (i+j) mod 17, uses 17 colours
    base_colors_list = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors_list[:]
    best_num = len(set(best_colors))  # 17

    best_colors_np = np.array(best_colors, dtype=np.int32)
    assert count_violations_fast(best_colors_np) == 0

    # Direct backtracking CSP with forward checking
    def direct_csp(k, max_nodes=500000):
        """
        Direct backtracking: assign colours from palette {0..k-1} to edges in lex order.
        Forward checking: after each assignment, check all K4s that contain the new edge.
        Symmetry breaking: fix first edge to colour 0.
        """
        colors = np.full(num_edges, -1, dtype=np.int32)
        slack = np.zeros(num_k4s, dtype=np.int32)
        # Track which K4s are "open" (not all 6 edges assigned yet)
        # For each K4, count how many of its 6 edges are assigned
        k4_assigned_count = np.zeros(num_k4s, dtype=np.int32)
        nodes_visited = 0

        # For each edge, list of K4s it belongs to
        # Precompute for speed
        edge_k4s = [lst for lst in edge_to_k4]

        # Order edges: lex order, but we can try most-constrained first
        # Actually, for CSP with forward checking, we want to assign edges in an order
        # that creates K4s as early as possible to prune.
        # Lex order works because K4s are formed when the 4th vertex is added.
        # Let's use lex order for simplicity and symmetry breaking.

        def check_k4_validity(k4_idx):
            """Check if a K4 is valid given current assignments. Returns True if valid or incomplete."""
            e_indices = k4_edges_arr[k4_idx]
            # Count assigned edges
            assigned = 0
            col_counts = {}
            for ei in e_indices:
                c = colors[ei]
                if c == -1:
                    continue
                assigned += 1
                col_counts[c] = col_counts.get(c, 0) + 1
            # If not all assigned, we can't fully validate, but we can check for immediate violations
            # A violation occurs if any colour appears >= 2 times AND there are not enough unassigned
            # edges to bring distinct count up to 5.
            # Actually, simpler: if any colour count >= 2, and the number of distinct colours
            # among assigned + number of unassigned < 5, it's a violation.
            # But even simpler: if any colour count >= 2, and even if all unassigned edges
            # get new distinct colours, can we reach 5 distinct?
            num_unassigned = 6 - assigned
            num_distinct_assigned = len(col_counts)
            # Max possible distinct = num_distinct_assigned + num_unassigned
            if num_distinct_assigned + num_unassigned < 5:
                return False
            # Also, if any colour count >= 2, it's not immediately a violation unless
            # the above condition fails. Actually, the condition above is sufficient.
            # Wait: if we have 2 edges of same colour, and 4 unassigned, max distinct = 1 + 4 = 5, OK.
            # If we have 2 edges of same colour, and 3 unassigned, max distinct = 1 + 3 = 4 < 5, FAIL.
            # So the condition num_distinct_assigned + num_unassigned < 5 is correct.
            return True

        def backtrack(edge_idx):
            nonlocal nodes_visited
            if edge_idx == num_edges:
                return True
            if nodes_visited >= max_nodes:
                return False
            if time.time() > deadline:
                return False

            nodes_visited += 1
            ei = edge_idx
            # Try colours in order
            for c in range(k):
                if ei == 0 and c != 0:
                    continue  # Symmetry breaking
                colors[ei] = c
                # Update affected K4s
                affected = edge_k4s[ei]
                valid = True
                for k4_idx in affected:
                    k4_assigned_count[k4_idx] += 1
                    if not check_k4_validity(k4_idx):
                        valid = False
                        break
                if valid:
                    if backtrack(edge_idx + 1):
                        return True
                # Undo
                for k4_idx in affected:
                    k4_assigned_count[k4_idx] -= 1
            colors[ei] = -1
            return False

        if backtrack(0):
            return colors.copy()
        return None

    # Local search with slack tracking
    def local_search(current_np, max_iters=50000):
        """Local search that tries to reduce palette size."""
        cur = current_np.copy()
        slack = compute_slack(cur)
        cur_palette = sorted(set(cur.tolist()))
        best_local = cur.copy()
        best_local_num = len(cur_palette)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            ei = random.randint(0, num_edges - 1)
            old_c = cur[ei]

            if len(cur_palette) <= 1:
                break

            # Try a random colour from palette (excluding current)
            other_palette = [c for c in cur_palette if c != old_c]
            if not other_palette:
                continue
            new_c = random.choice(other_palette)

            cur[ei] = new_c
            affected = edge_to_k4_arr[ei]
            if len(affected) > 0:
                k4_sub = cur[k4_edges_arr[affected]]
                new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                for pi in range(15):
                    new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                slack[affected] = new_slack_sub

            if np.all(slack < 2):
                # Valid move - check if palette reduced
                new_palette = set(cur.tolist())
                if len(new_palette) < best_local_num:
                    best_local = cur.copy()
                    best_local_num = len(new_palette)
                    cur_palette = sorted(new_palette)
            else:
                # Revert
                cur[ei] = old_c
                if len(affected) > 0:
                    k4_sub = cur[k4_edges_arr[affected]]
                    new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                    for pi in range(15):
                        new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                    slack[affected] = new_slack_sub

        return best_local, best_local_num

    # Try direct CSP for small k values
    for k in range(13, 16):
        if time.time() > deadline - 10:
            break
        result = direct_csp(k, max_nodes=200000)
        if result is not None:
            if count_violations_fast(result) == 0:
                best_colors_np = result
                best_num = k
                break

    # If direct CSP failed, use local search from base
    if best_num == 17:
        current_np = best_colors_np.copy()
        for iteration in range(100):
            if time.time() > deadline:
                break
            local_best, local_num = local_search(current_np, max_iters=20000)
            if local_num < best_num:
                best_colors_np = local_best.copy()
                best_num = local_num
                current_np = local_best.copy()
            else:
                break

    # Final check
    if count_violations_fast(best_colors_np) != 0:
        best_colors_np = np.array(base_colors_list, dtype=np.int32)
        best_num = 17

    best_colors = best_colors_np.tolist()

    # Renormalize colours to 0..best_num-1
    col_map = {}
    for i, c in enumerate(sorted(set(best_colors))):
        col_map[c] = i
    best_colors = [col_map[c] for c in best_colors]

    return best_colors
# EVOLVE-BLOCK-END
