import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    best_edges = None
    best_count = float('inf')
    best_n = None

    # For each n, we have 2^n colorings. A hypergraph is non-2-colorable iff
    # for every coloring c, at least one edge is monochromatic under c.
    # This is a hitting set problem: choose minimum edges such that every
    # coloring is "hit" (has a monochromatic edge).
    #
    # ILP formulation:
    #   Variables: x_e in {0,1} for each edge e
    #   Minimize: sum x_e
    #   Subject to: for each coloring c, sum_{e monochromatic under c} x_e >= 1
    #
    # For n=9: 512 colorings, 126 edges. Very manageable.
    # For n=10: 1024 colorings, 252 edges.
    # For n=11: 2048 colorings, 462 edges.

    for n in [9, 10, 11]:
        if time.time() > deadline - 5:
            break

        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        num_colorings = 1 << n

        # Precompute edge vertex masks
        edge_masks = []
        for e in all_edges:
            mask_e = 0
            for v in e:
                mask_e |= (1 << v)
            edge_masks.append(mask_e)

        # Build constraint matrix A: A[c, e] = 1 if edge e is monochromatic under coloring c
        # A coloring c is a bitmask. Edge e is monochromatic if (c & mask_e) == 0 or (c & mask_e) == mask_e
        # We need A as a sparse-like structure. For n=9, 512x126 = 64512 entries, dense is fine.
        # For n=10, 1024x252 = 258048, still fine.
        # For n=11, 2048x462 = 946176, fine.

        # Actually, let's build it as a numpy array for scipy
        A = np.zeros((num_colorings, num_edges), dtype=np.int8)

        for c in range(num_colorings):
            for i in range(num_edges):
                mask_e = edge_masks[i]
                inter = c & mask_e
                if inter == 0 or inter == mask_e:
                    A[c, i] = 1

        # Check if the full graph is non-colorable (i.e., every coloring has at least one mono edge)
        # If any row of A is all zeros, that coloring has no mono edges, so full graph is colorable.
        if np.any(A.sum(axis=1) == 0):
            continue  # Full graph is colorable, skip this n

        # ILP: minimize sum(x_e) subject to A @ x >= 1, x in {0,1}
        c_obj = np.ones(num_edges)

        # Constraints: A @ x >= 1
        # scipy milp uses A_ub @ x <= b_ub and A_eq @ x == b_eq
        # A @ x >= 1  =>  -A @ x <= -1
        A_ub = -A.astype(np.float64)
        b_ub = np.full(num_colorings, -1.0)

        constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
        integrality = np.ones(num_edges)  # all integer
        bounds = Bounds(0, 1)

        try:
            result = milp(
                c=c_obj,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
        except Exception:
            # Fallback to greedy if MILP fails
            result = None

        if result is not None and result.status == 0:
            # Extract solution
            x = np.round(result.x)
            selected_edges = [all_edges[i] for i in range(num_edges) if x[i] > 0.5]
            count = len(selected_edges)

            # Verify: check that this set is indeed non-2-colorable
            # Build active mask and check
            active_set = set(selected_edges)
            # Quick verification using bad_masks approach
            is_valid = True
            for c in range(num_colorings):
                has_mono = False
                for e in selected_edges:
                    mask_e = 0
                    for v in e:
                        mask_e |= (1 << v)
                    inter = c & mask_e
                    if inter == 0 or inter == mask_e:
                        has_mono = True
                        break
                if not has_mono:
                    is_valid = False
                    break

            if is_valid and count < best_count:
                best_count = count
                best_edges = selected_edges
                best_n = n
        else:
            # Fallback: greedy reduction
            if time.time() > deadline - 5:
                continue

            # Precompute bad_masks for greedy
            bad_masks = [0] * num_colorings
            for c in range(num_colorings):
                bm = 0
                for i in range(num_edges):
                    mask_e = edge_masks[i]
                    inter = c & mask_e
                    if inter == 0 or inter == mask_e:
                        bm |= (1 << i)
                bad_masks[c] = bm

            def is_non_colorable(active_mask):
                for bm in bad_masks:
                    if (bm & active_mask) == 0:
                        return False
                return True

            current_mask = (1 << num_edges) - 1
            if not is_non_colorable(current_mask):
                continue

            best_local_mask = current_mask
            best_local_count = num_edges

            restarts = 0
            while time.time() < deadline - 3:
                restarts += 1
                curr = current_mask
                changed = True
                while changed:
                    changed = False
                    indices = list(range(num_edges))
                    random.shuffle(indices)
                    for i in indices:
                        if curr & (1 << i):
                            new_mask = curr & ~(1 << i)
                            if is_non_colorable(new_mask):
                                curr = new_mask
                                changed = True

                count = bin(curr).count('1')
                if count < best_local_count:
                    best_local_count = count
                    best_local_mask = curr

                if best_local_count <= 51:
                    break

            if best_local_count < best_count:
                best_count = best_local_count
                best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]
                best_n = n

    if best_edges is None:
        # Fallback: K_9^5
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
