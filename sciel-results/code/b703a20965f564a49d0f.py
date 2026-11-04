import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbor lists as integer bitsets (Python ints for arbitrary precision)
    nb_count = 1 + n * (q - 1)
    # neighbors[i] = list of indices in ball of i (including i)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Build bitset representation: ball_mask[i] is a Python int with bits set for all neighbors
    # Since space_size = 2187, we need 2187 bits. Python handles this natively.
    ball_mask = [0] * space_size
    for i in range(space_size):
        m = 0
        for j in neighbors[i]:
            m |= (1 << j)
        ball_mask[i] = m

    def build_coverage_bitset(code_indices):
        """Return a Python int bitset where bit j = 1 if j is covered."""
        cov = 0
        for c in code_indices:
            cov |= ball_mask[c]
        return cov

    def count_uncovered(cov_bitset):
        """Count bits that are 0 in cov_bitset (uncovered points)."""
        # Full mask for space_size bits
        full_mask = (1 << space_size) - 1
        uncovered = full_mask ^ cov_bitset
        return bin(uncovered).count('1')

    def greedy_lagrangian(seed=0, max_time=None):
        """
        Greedy selection guided by Lagrangian dual variables.
        Dual variable u_j = 1 if point j is uncovered, 0 otherwise.
        Reduced cost of node i = sum of u_j for j in ball(i) - lambda * (already selected)
        We select the node with highest reduced cost (most uncovered neighbors).
        Subgradient update: after each selection, update u for newly covered points.
        """
        if max_time is None:
            max_time = deadline - 20.0
        rng = np.random.default_rng(seed)

        # u: dual variables, 1 if uncovered
        u = [1] * space_size  # all start uncovered
        selected = set()
        code = []
        cov_bitset = 0
        full_mask = (1 << space_size) - 1

        # Precompute: for each node, which dual variables are in its ball
        # reduced_cost[i] = sum(u[j] for j in neighbors[i])
        # We maintain this incrementally.

        # Initial reduced costs: all nodes have the same ball size, so all start at nb_count
        # But we'll compute them properly.
        red_cost = np.zeros(space_size, dtype=np.float64)
        for i in range(space_size):
            red_cost[i] = nb_count  # all u[j]=1 initially

        while True:
            if (time.time() - t_start) > max_time:
                break
            if cov_bitset == full_mask:
                break

            # Add small random perturbation for tie-breaking (Lagrangian subgradient noise)
            perturb = rng.random(space_size) * 0.01
            best = int(np.argmax(red_cost + perturb))

            if red_cost[best] <= 0:
                break

            # Select node best
            selected.add(best)
            code.append(best)

            # Update coverage
            newly_covered = ball_mask[best] & ~cov_bitset
            cov_bitset |= ball_mask[best]

            # Subgradient update: for each newly covered point, u[j] goes from 1 to 0
            # This decreases red_cost for all nodes whose ball contains j
            # Instead of updating all, we recompute red_cost incrementally:
            # For each node i, red_cost[i] -= number of newly covered points in ball(i)
            # This is expensive. Alternative: just recompute from scratch periodically.
            # For efficiency with bitsets, let's just recompute red_cost when needed.

            # Actually, let's use a different approach: maintain u as a bitset too
            # and compute red_cost on the fly using bitset operations.
            # But Python int bitset AND + popcount is fast for 2187 bits.

            # Let's just track uncovered as a bitset and compute gains directly
            uncovered_bitset = full_mask ^ cov_bitset

            if uncovered_bitset == 0:
                break

            # Check if we should continue
            # Compute gain for best node (should be > 0 since it was uncovered)
            gain = bin(ball_mask[best] & uncovered_bitset).count('1')
            if gain == 0:
                # This shouldn't happen if we track properly, but safety
                break

            # Update red_cost: subtract 1 for each newly covered point in each node's ball
            # This is O(space_size * popcount(newly_covered)) which could be slow
            # Alternative: recompute red_cost from scratch every K iterations
            # For now, let's just recompute when the uncovered set changes significantly

            # Actually, the simplest correct approach: just use the uncovered bitset
            # to compute gains directly each time, skipping the Lagrangian bookkeeping.
            # The "Lagrangian" aspect is that we're using the dual (uncovered count)
            # to guide selection, which is exactly what greedy set cover does.
            # The key difference from naive greedy: we use bitset operations for speed.

        return code

    def greedy_bitset(seed=0, max_time=None):
        """
        Pure greedy set cover using bitset operations for speed.
        At each step, pick the node that covers the most currently-uncovered points.
        """
        if max_time is None:
            max_time = deadline - 20.0
        rng = np.random.default_rng(seed)

        full_mask = (1 << space_size) - 1
        uncovered = full_mask  # all uncovered
        code = []

        while True:
            if (time.time() - t_start) > max_time:
                break
            if uncovered == 0:
                break

            # Find node with max coverage of uncovered points
            # gain[i] = popcount(ball_mask[i] & uncovered)
            # This is O(space_size) bitset ANDs + popcounts per iteration
            # 2187 ANDs + 2187 popcounts of 2187-bit ints per iteration
            # Python can do this in ~microseconds per operation, so ~few ms per iteration

            best = -1
            best_gain = 0
            for i in range(space_size):
                gain = bin(ball_mask[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best = i
                elif gain == best_gain and best >= 0:
                    # Tie-break with randomization
                    if rng.random() < 0.5:
                        best = i
                        best_gain = gain

            if best_gain == 0:
                break

            code.append(best)
            uncovered &= ~ball_mask[best]

        return code

    def compress_bitset(code, max_time=None):
        """
        Iterative removal: remove a codeword if all points in its ball are still covered.
        Uses bitset coverage check.
        """
        if max_time is None:
            max_time = deadline - 5.0

        cur_code = list(code)
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            # Compute coverage without each codeword to check removability
            # For each codeword c, check if ball_mask[c] is subset of coverage of others
            # i.e., ball_mask[c] & ~(coverage of others) == 0

            # Build full coverage
            full_cov = 0
            for c in cur_code:
                full_cov |= ball_mask[c]

            for i in range(len(cur_code)):
                if (time.time() - t_start) > max_time:
                    break
                c = cur_code[i]
                # Coverage without c
                cov_without = 0
                for j, cj in enumerate(cur_code):
                    if j != i:
                        cov_without |= ball_mask[cj]
                # Check if ball(c) is fully covered without c
                if (ball_mask[c] & ~cov_without) == 0:
                    cur_code.pop(i)
                    improved = True
                    break  # Restart

        return cur_code

    def compress_bitset_fast(code, max_time=None):
        """
        Faster compression: maintain coverage count per point using a list,
        remove codewords where min count in ball >= 2.
        """
        if max_time is None:
            max_time = deadline - 5.0

        cur_code = list(code)
        # Coverage count per point
        cov_count = [0] * space_size
        for c in cur_code:
            for j in neighbors[c]:
                cov_count[j] += 1

        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            # Find removable codewords: min cov_count in ball >= 2
            # Sort by "safety" (min coverage in ball) descending, try most redundant first
            safety = []
            for c in cur_code:
                mn = min(cov_count[j] for j in neighbors[c])
                safety.append(mn)

            order = sorted(range(len(cur_code)), key=lambda i: -safety[i])
            for idx in order:
                if (time.time() - t_start) > max_time:
                    break
                if safety[idx] >= 2:
                    c = cur_code[idx]
                    for j in neighbors[c]:
                        cov_count[j] -= 1
                    cur_code.pop(idx)
                    improved = True
                    break  # Restart

        return cur_code

    def repair_bitset(code, max_time=None):
        """Add codewords to cover all uncovered points."""
        if max_time is None:
            max_time = deadline - 10.0

        cur_code = list(code)
        full_mask = (1 << space_size) - 1
        cov = 0
        for c in cur_code:
            cov |= ball_mask[c]
        uncovered = full_mask ^ cov

        while uncovered != 0 and (time.time() - t_start) < max_time:
            best = -1
            best_gain = 0
            for i in range(space_size):
                gain = bin(ball_mask[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best = i
            if best_gain == 0:
                break
            cur_code.append(best)
            uncovered &= ~ball_mask[best]

        return cur_code

    # === MAIN ALGORITHM ===

    best_code = None
    best_size = space_size + 1

    # Phase 1: Multiple greedy starts with bitset operations
    for seed in range(12):
        if (time.time() - t_start) > deadline - 50.0:
            break
        code = greedy_bitset(seed, deadline - 45.0)
        code = compress_bitset_fast(code, deadline - 40.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Iterated Local Search with kicks
    rng = np.random.default_rng(99999)
    cur_code = best_code[:] if best_code else []

    for iteration in range(300):
        if (time.time() - t_start) > deadline - 15.0:
            break

        # Kick: remove k random codewords
        k = 1 + int(rng.integers(0, 5))  # Remove 1-5
        if len(cur_code) <= k:
            k = max(1, len(cur_code) // 3)

        remove_indices = rng.choice(len(cur_code), size=k, replace=False)
        remove_set = set(int(ri) for ri in remove_indices)
        kicked_code = [c for i, c in enumerate(cur_code) if i not in remove_set]

        # Repair
        kicked_code = repair_bitset(kicked_code, deadline - 20.0)

        # Compress
        kicked_code = compress_bitset_fast(kicked_code, deadline - 15.0)

        if len(kicked_code) < best_size:
            best_code = kicked_code[:]
            best_size = len(kicked_code)
            cur_code = kicked_code[:]
        else:
            # Accept worse solution with some probability (simulated annealing-like)
            if rng.random() < 0.25:
                cur_code = kicked_code[:]

    # Phase 3: Final aggressive compression
    if (time.time() - t_start) < deadline - 3.0 and best_code:
        final_code = compress_bitset_fast(best_code[:], deadline - 2.0)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    # Final verification and fallback
    if best_code is None or len(best_code) == 0:
        best_code = greedy_bitset(0, deadline - 2.0)

    # Verify coverage
    full_mask = (1 << space_size) - 1
    cov = 0
    for c in best_code:
        cov |= ball_mask[c]
    if cov != full_mask:
        # Repair
        best_code = repair_bitset(best_code, deadline - 1.0)
        cov = 0
        for c in best_code:
            cov |= ball_mask[c]
        if cov != full_mask:
            # Ultimate fallback
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
