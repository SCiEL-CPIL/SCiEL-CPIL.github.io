# EVOLVE-BLOCK-START
import numpy as np


def _entropy_from_counts(counts, total):
    """
    Compute Shannon entropy in bits from counts and total probability mass.
    counts: array of integers (counts of items in each bin)
    total: total number of items (sum of counts)
    """
    if total <= 0:
        return 0.0
    probs = counts / total
    # Ignore zero probabilities
    mask = probs > 0
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    return float(-np.sum(p * np.log2(p)))


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Uses greedy + local search to find a sparse support set S
    that maximizes D/M, where D is entropy of x-y and M is max entropy of x, x+y, x+2y.
    """
    n = 10
    total_cells = n * n

    # Precompute bin indices for each cell (y, x)
    idx_D = np.zeros((n, n), dtype=np.int32)
    idx_H0 = np.zeros((n, n), dtype=np.int32)
    idx_H1 = np.zeros((n, n), dtype=np.int32)
    idx_H2 = np.zeros((n, n), dtype=np.int32)

    for y in range(n):
        for x in range(n):
            idx_D[y, x] = (x - y) + (n - 1)
            idx_H0[y, x] = x
            idx_H1[y, x] = x + y
            idx_H2[y, x] = x + 2 * y

    size_D = 2 * n - 1
    size_H0 = n
    size_H1 = 2 * n - 1
    size_H2 = 3 * n - 1

    # Flatten for easier iteration
    cells = np.array([(y, x) for y in range(n) for x in range(n)])
    idx_D_flat = idx_D.flatten()
    idx_H0_flat = idx_H0.flatten()
    idx_H1_flat = idx_H1.flatten()
    idx_H2_flat = idx_H2.flatten()

    def compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, k):
        if k <= 0:
            return 0.0
        D = _entropy_from_counts(counts_D, k)
        H0 = _entropy_from_counts(counts_H0, k)
        H1 = _entropy_from_counts(counts_H1, k)
        H2 = _entropy_from_counts(counts_H2, k)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M

    # Greedy construction
    selected = np.zeros(total_cells, dtype=bool)
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    current_k = 0

    best_ratio = 0.0
    best_selected = np.zeros(total_cells, dtype=bool)

    # Greedy: add cells one by one
    for step in range(total_cells):
        best_gain = -np.inf
        best_cell_idx = -1

        for i in range(total_cells):
            if selected[i]:
                continue

            iD = idx_D_flat[i]
            iH0 = idx_H0_flat[i]
            iH1 = idx_H1_flat[i]
            iH2 = idx_H2_flat[i]

            # Tentatively add
            counts_D[iD] += 1
            counts_H0[iH0] += 1
            counts_H1[iH1] += 1
            counts_H2[iH2] += 1
            new_k = current_k + 1

            ratio_new = compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)

            # Revert
            counts_D[iD] -= 1
            counts_H0[iH0] -= 1
            counts_H1[iH1] -= 1
            counts_H2[iH2] -= 1

            if ratio_new > best_gain:
                best_gain = ratio_new
                best_cell_idx = i

        if best_cell_idx < 0:
            break

        # Add the best cell
        selected[best_cell_idx] = True
        iD = idx_D_flat[best_cell_idx]
        iH0 = idx_H0_flat[best_cell_idx]
        iH1 = idx_H1_flat[best_cell_idx]
        iH2 = idx_H2_flat[best_cell_idx]
        counts_D[iD] += 1
        counts_H0[iH0] += 1
        counts_H1[iH1] += 1
        counts_H2[iH2] += 1
        current_k += 1

        if best_gain > best_ratio:
            best_ratio = best_gain
            best_selected = selected.copy()

        # Early stopping heuristic: if ratio drops significantly and we've added enough
        if current_k > 5 and best_gain < best_ratio * 0.7:
            # Try a few more steps, but stop if it keeps getting worse
            if current_k > 10 and best_gain < best_ratio * 0.5:
                break

    # Local search: try swaps (add one, remove another)
    # Start from best_selected
    selected = best_selected.copy()
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    for i in range(total_cells):
        if selected[i]:
            counts_D[idx_D_flat[i]] += 1
            counts_H0[idx_H0_flat[i]] += 1
            counts_H1[idx_H1_flat[i]] += 1
            counts_H2[idx_H2_flat[i]] += 1
    current_k = np.sum(selected)
    current_ratio = compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)

    improved = True
    max_iter = 50
    for _ in range(max_iter):
        if not improved:
            break
        improved = False

        # Try all possible swaps: remove i (selected), add j (not selected)
        selected_list = np.where(selected)[0]
        unselected_list = np.where(~selected)[0]

        best_swap_ratio = current_ratio
        best_swap_remove = -1
        best_swap_add = -1

        for i in selected_list:
            for j in unselected_list:
                # Remove i
                iD_i = idx_D_flat[i]
                iH0_i = idx_H0_flat[i]
                iH1_i = idx_H1_flat[i]
                iH2_i = idx_H2_flat[i]
                counts_D[iD_i] -= 1
                counts_H0[iH0_i] -= 1
                counts_H1[iH1_i] -= 1
                counts_H2[iH2_i] -= 1

                # Add j
                iD_j = idx_D_flat[j]
                iH0_j = idx_H0_flat[j]
                iH1_j = idx_H1_flat[j]
                iH2_j = idx_H2_flat[j]
                counts_D[iD_j] += 1
                counts_H0[iH0_j] += 1
                counts_H1[iH1_j] += 1
                counts_H2[iH2_j] += 1

                new_k = current_k  # same size
                ratio_new = compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)

                # Revert
                counts_D[iD_j] -= 1
                counts_H0[iH0_j] -= 1
                counts_H1[iH1_j] -= 1
                counts_H2[iH2_j] -= 1
                counts_D[iD_i] += 1
                counts_H0[iH0_i] += 1
                counts_H1[iH1_i] += 1
                counts_H2[iH2_i] += 1

                if ratio_new > best_swap_ratio + 1e-12:
                    best_swap_ratio = ratio_new
                    best_swap_remove = i
                    best_swap_add = j

        if best_swap_remove >= 0:
            # Apply the best swap
            i = best_swap_remove
            j = best_swap_add
            iD_i = idx_D_flat[i]
            iH0_i = idx_H0_flat[i]
            iH1_i = idx_H1_flat[i]
            iH2_i = idx_H2_flat[i]
            counts_D[iD_i] -= 1
            counts_H0[iH0_i] -= 1
            counts_H1[iH1_i] -= 1
            counts_H2[iH2_i] -= 1

            iD_j = idx_D_flat[j]
            iH0_j = idx_H0_flat[j]
            iH1_j = idx_H1_flat[j]
            iH2_j = idx_H2_flat[j]
            counts_D[iD_j] += 1
            counts_H0[iH0_j] += 1
            counts_H1[iH1_j] += 1
            counts_H2[iH2_j] += 1

            selected[i] = False
            selected[j] = True
            current_ratio = best_swap_ratio
            best_ratio = max(best_ratio, current_ratio)
            best_selected = selected.copy()
            improved = True

    # Also try adding one more cell to the best found
    for i in range(total_cells):
        if best_selected[i]:
            continue
        iD = idx_D_flat[i]
        iH0 = idx_H0_flat[i]
        iH1 = idx_H1_flat[i]
        iH2 = idx_H2_flat[i]

        # Rebuild counts from best_selected
        counts_D = np.zeros(size_D, dtype=np.int32)
        counts_H0 = np.zeros(size_H0, dtype=np.int32)
        counts_H1 = np.zeros(size_H1, dtype=np.int32)
        counts_H2 = np.zeros(size_H2, dtype=np.int32)
        for idx in range(total_cells):
            if best_selected[idx]:
                counts_D[idx_D_flat[idx]] += 1
                counts_H0[idx_H0_flat[idx]] += 1
                counts_H1[idx_H1_flat[idx]] += 1
                counts_H2[idx_H2_flat[idx]] += 1
        k = np.sum(best_selected)

        counts_D[iD] += 1
        counts_H0[iH0] += 1
        counts_H1[iH1] += 1
        counts_H2[iH2] += 1
        new_k = k + 1
        ratio_new = compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)

        if ratio_new > best_ratio:
            best_ratio = ratio_new
            best_selected[i] = True

    # Build final distribution
    P = np.zeros((n, n), dtype=np.float64)
    k = np.sum(best_selected)
    if k > 0:
        P[best_selected.reshape(n, n)] = 1.0 / k
    else:
        P = np.ones((n, n), dtype=np.float64) / total_cells

    # Ensure validity
    P = np.clip(P, 0, None)
    total_sum = np.sum(P)
    if total_sum > 1e-9:
        P = P / total_sum
    else:
        P = np.ones((n, n), dtype=np.float64) / total_cells

    return P


# EVOLVE-BLOCK-END
