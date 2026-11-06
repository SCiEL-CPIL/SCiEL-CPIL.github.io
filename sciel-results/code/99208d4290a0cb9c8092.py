# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE_MARGIN = 5.0

def _precompute_aps():
    """Precompute all arithmetic progressions and their index lists (0-based)."""
    aps = []
    # step 1: contiguous blocks
    for start in range(N):
        for length in range(2, N - start + 1):  # length 1 trivial, skip
            aps.append((start, 1, length))
    # step >= 2
    for step in range(2, N // 2 + 1):
        for start in range(N):
            # max length such that start + (length-1)*step < N
            max_len = (N - 1 - start) // step + 1
            if max_len < 2:
                continue
            for length in range(2, max_len + 1):
                aps.append((start, step, length))
    return aps

def _build_ap_index_lists(aps):
    """Convert (start, step, length) triples to lists of indices."""
    index_lists = []
    for start, step, length in aps:
        idx = list(range(start, start + length * step, step))
        index_lists.append(idx)
    return index_lists

def _compute_max_abs_sum(signs, index_lists):
    """Compute the maximum absolute signed sum over all APs."""
    max_abs = 0
    for idx in index_lists:
        s = 0
        for i in idx:
            s += signs[i]
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
    return max_abs

def _incremental_max_abs(signs, index_lists, idx):
    """
    Return the maximum absolute sum over all APs, exploiting that only APs
    containing index idx change when signs[idx] flips.
    """
    # We need to track which APs contain each index for efficiency.
    # For now, just recompute - this is a placeholder.
    return _compute_max_abs_sum(signs, index_lists)

def _flip(signs, i):
    signs[i] = -signs[i]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    # Precompute APs
    aps = _precompute_aps()
    index_lists = _build_ap_index_lists(aps)
    num_aps = len(index_lists)

    # Build index -> AP mapping for fast lookup
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, idx_list in enumerate(index_lists):
        for i in idx_list:
            index_to_aps[i].append(ap_idx)

    # Precompute AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)

    def compute_ap_sums(signs):
        for ap_idx, idx_list in enumerate(index_lists):
            s = 0
            for i in idx_list:
                s += signs[i]
            ap_sums[ap_idx] = s

    # Initialize with Thue-Morse sequence (known low discrepancy)
    # Thue-Morse: sign(i) = -1 if popcount(i) is odd, +1 if even
    signs = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # popcount
        pc = 0
        x = i
        while x:
            pc += x & 1
            x >>= 1
        signs[i] = 1 if pc % 2 == 0 else -1

    compute_ap_sums(signs)
    current_max = int(np.max(np.abs(ap_sums)))

    best_signs = signs.copy()
    best_max = current_max

    # Local search: single flips
    rng = np.random.default_rng(12345)

    def try_flip(i):
        """Try flipping sign at index i, return improvement in max abs sum."""
        old_sign = signs[i]
        new_sign = -old_sign
        # Update AP sums for APs containing i
        affected_aps = index_to_aps[i]
        for ap_idx in affected_aps:
            ap_sums[ap_idx] += 2 * new_sign  # flip changes sum by 2*new_sign
        # Compute new max
        new_max = int(np.max(np.abs(ap_sums)))
        # Revert
        signs[i] = old_sign
        for ap_idx in affected_aps:
            ap_sums[ap_idx] -= 2 * new_sign
        return new_max

    # Iterated local search
    while time.time() < deadline:
        improved = False
        # Greedy single flips: keep flipping if it reduces max
        for iteration in range(100):
            if time.time() > deadline:
                break
            best_i = -1
            best_new_max = current_max
            # Random subset of indices to check for speed
            candidates = rng.choice(N, size=min(32, N), replace=False)
            for i in candidates:
                new_max = try_flip(i)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_i = i
            if best_i >= 0 and best_new_max < current_max:
                # Commit the flip
                _flip(signs, best_i)
                for ap_idx in index_to_aps[best_i]:
                    ap_sums[ap_idx] += 2 * signs[best_i]
                current_max = best_new_max
                improved = True
            else:
                break

        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()

        # Perturbation for next iteration
        num_flips = rng.integers(2, 7)
        flip_indices = rng.choice(N, size=num_flips, replace=False)
        for i in flip_indices:
            _flip(signs, i)
            for ap_idx in index_to_aps[i]:
                ap_sums[ap_idx] += 2 * signs[i]
        # Recompute max after perturbation
        current_max = int(np.max(np.abs(ap_sums)))

    # Return best found
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
