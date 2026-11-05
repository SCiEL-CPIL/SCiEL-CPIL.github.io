"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)  # 129

    # Map sum value -> index in target_sums (sum range [0, 256])
    sum_to_idx = np.full(257, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def compute_counts(A_array):
        """
        Compute representation counts for target sums given set A as a boolean array.
        A_array: boolean array of length 129 (index = value, True if in set)
        Returns: counts array of length 129 (for target sums)
        """
        A_vals = np.where(A_array)[0]  # values in A
        m = len(A_vals)
        if m == 0:
            return np.zeros(num_targets, dtype=np.int32)

        # All unordered pairs with replacement
        # Use broadcasting: A_vals[:, None] + A_vals[None, :]
        # But we only want i <= j (unordered with equal summands allowed)
        # Count each unordered pair once: for i<j count once, for i==j count once

        # Efficient: compute all pair sums, then count
        # For m elements, there are m*(m+1)/2 unordered pairs
        # Use upper triangle of the sum matrix
        pair_sums = A_vals[:, None] + A_vals[None, :]  # m x m matrix
        # We want upper triangle (i <= j)
        triu_idx = np.triu_indices(m)
        relevant_sums = pair_sums[triu_idx]

        # Count how many fall in each target sum
        counts = np.zeros(num_targets, dtype=np.int32)
        # Map sums to indices
        valid = (relevant_sums >= min_sum) & (relevant_sums <= max_sum)
        if np.any(valid):
            valid_sums = relevant_sums[valid]
            idxs = sum_to_idx[valid_sums]
            # Count occurrences
            unique, cnt = np.unique(idxs, return_counts=True)
            counts[unique] = cnt

        return counts

    def compute_counts_from_list(A_list):
        """Compute counts from a list of values."""
        A_array = np.zeros(129, dtype=bool)
        for v in A_list:
            A_array[v] = True
        return compute_counts(A_array)

    def get_cost(counts):
        """Cost = peak multiplicity, with penalty for uncovered sums."""
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            return 10000 + uncovered * 100 + np.max(counts)
        return int(np.max(counts))

    def greedy_construct():
        """
        Greedy construction: start with empty set, add elements to cover
        uncovered sums, prioritizing the sum with fewest representations.
        """
        A = set()
        A_array = np.zeros(129, dtype=bool)
        counts = np.zeros(num_targets, dtype=np.int32)

        # We need to cover all 129 target sums
        # Greedily add elements that cover the most uncovered sums

        max_iter = 200
        for _ in range(max_iter):
            if np.all(counts > 0):
                break

            # Find the sum with the fewest representations (should be 0)
            uncovered_mask = counts == 0
            if not np.any(uncovered_mask):
                break

            # Pick the uncovered sum that is "hardest" to cover
            # A sum s can be formed by pairs (a, s-a) where a in A or s-a in A
            # For now, just pick the first uncovered sum
            uncovered_sums = target_sums[uncovered_mask]
            # Prioritize sums near the middle (192) as they have more pair options?
            # Or prioritize sums that are far from current coverage
            target_sum = int(uncovered_sums[len(uncovered_sums) // 2])

            # Try all possible elements x not in A
            # Adding x creates pairs (x, a) for a in A, and (x, x)
            best_x = None
            best_gain = 0

            for x in range(129):
                if x in A:
                    continue

                # Compute how many new sums would be covered
                # Pairs: (x, x) -> 2x
                # Pairs: (x, a) for a in A -> x + a
                new_sums_set = set()
                s_xx = 2 * x
                if min_sum <= s_xx <= max_sum:
                    new_sums_set.add(s_xx)

                for a in A:
                    s = x + a
                    if min_sum <= s <= max_sum:
                        new_sums_set.add(s)

                # Count how many of these are currently uncovered
                gain = 0
                for s in new_sums_set:
                    idx = sum_to_idx[s]
                    if idx >= 0 and counts[idx] == 0:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is not None and best_gain > 0:
                A.add(best_x)
                A_array[best_x] = True
                # Recompute counts (could do delta, but recompute is simpler for correctness)
                counts = compute_counts(A_array)
            else:
                # No element helps; break to avoid infinite loop
                break

        # If still uncovered, add random elements
        while np.any(counts == 0) and len(A) < 129:
            uncovered_mask = counts == 0
            uncovered_sums = target_sums[uncovered_mask]
            target_sum = int(random.choice(uncovered_sums))

            # Find x such that there exists a in A with x + a = target_sum
            # i.e., x = target_sum - a
            found = False
            for a in A:
                x = target_sum - a
                if 0 <= x <= 128 and x not in A:
                    A.add(x)
                    A_array[x] = True
                    found = True
                    break

            if not found:
                # Try (x, x) where 2x = target_sum
                if target_sum % 2 == 0:
                    x = target_sum // 2
                    if 0 <= x <= 128 and x not in A:
                        A.add(x)
                        A_array[x] = True
                        found = True

            if not found:
                # Add a random element
                candidates = [x for x in range(129) if x not in A]
                if candidates:
                    x = random.choice(candidates)
                    A.add(x)
                    A_array[x] = True

            counts = compute_counts(A_array)

        return A, counts, get_cost(counts)

    def local_search(A, counts, cost, time_limit):
        """
        Local search: try adding, removing, or swapping elements.
        Use delta updates for speed.
        """
        best_A = set(A)
        best_counts = counts.copy()
        best_cost = cost

        A_array = np.zeros(129, dtype=bool)
        for v in A:
            A_array[v] = True

        A_list = list(A)
        current_cost = cost
        current_counts = counts.copy()

        all_elements = list(range(129))
        no_improve_count = 0

        while time.time() < time_limit:
            # Choose a move
            move_type = random.random()
            new_A = set(current_A := A_list)

            if move_type < 0.3:  # Add
                candidates = [x for x in all_elements if x not in new_A]
                if candidates:
                    # Prioritize elements that cover uncovered sums
                    uncovered_mask = current_counts == 0
                    if np.any(uncovered_mask):
                        uncovered_sums = target_sums[uncovered_mask]
                        # Score each candidate by how many uncovered sums it covers
                        best_x = None
                        best_score = -1
                        for x in candidates[:30]:  # Limit to 30 for speed
                            score = 0
                            s_xx = 2 * x
                            if min_sum <= s_xx <= max_sum:
                                idx = sum_to_idx[s_xx]
                                if idx >= 0 and current_counts[idx] == 0:
                                    score += 2
                            for a in A_list:
                                s = x + a
                                if min_sum <= s <= max_sum:
                                    idx = sum_to_idx[s]
                                    if idx >= 0 and current_counts[idx] == 0:
                                        score += 1
                            if score > best_score:
                                best_score = score
                                best_x = x
                        if best_x is not None and best_score > 0:
                            new_A.add(best_x)
                        else:
                            new_A.add(random.choice(candidates))
                    else:
                        new_A.add(random.choice(candidates))
            elif move_type < 0.5:  # Remove
                if len(new_A) > 25:
                    # Try removing the element that contributes least to coverage
                    # Approximate: remove a random element
                    x = random.choice(list(new_A))
                    new_A.remove(x)
            else:  # Swap
                if len(new_A) > 1:
                    x = random.choice(list(new_A))
                    new_A.remove(x)
                    candidates = [y for y in all_elements if y not in new_A]
                    if candidates:
                        y = random.choice(candidates)
                        new_A.add(y)

            if new_A == set(A_list):
                continue

            # Compute new counts
            new_A_array = np.zeros(129, dtype=bool)
            for v in new_A:
                new_A_array[v] = True
            new_counts = compute_counts(new_A_array)
            new_cost = get_cost(new_counts)

            if new_cost < current_cost:
                A_list = list(new_A)
                current_counts = new_counts
                current_cost = new_cost
                no_improve_count = 0

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A_list)
                    best_counts = new_counts.copy()
            else:
                # Accept worse moves with some probability (to escape local minima)
                delta = new_cost - current_cost
                if delta < 50 and random.random() < 0.1:
                    A_list = list(new_A)
                    current_counts = new_counts
                    current_cost = new_cost
                no_improve_count += 1

            # If stuck for too long, do a random restart
            if no_improve_count > 500:
                # Perturb: remove some random elements and re-greedy
                for _ in range(min(5, len(A_list) // 4)):
                    if A_list:
                        A_list.remove(random.choice(A_list))
                A_array = np.zeros(129, dtype=bool)
                for v in A_list:
                    A_array[v] = True
                current_counts = compute_counts(A_array)
                current_cost = get_cost(current_counts)
                no_improve_count = 0

        return best_A, best_counts, best_cost

    # Multi-start
    best_A = None
    best_cost = float('inf')
    best_counts = None

    # Start 1: Greedy construction
    A1, counts1, cost1 = greedy_construct()
    if cost1 < best_cost:
        best_cost = cost1
        best_A = A1
        best_counts = counts1

    # Start 2: Dense interval subset
    A2 = set(range(0, 129, 2))  # Even numbers
    counts2 = compute_counts_from_list(A2)
    cost2 = get_cost(counts2)
    if cost2 < best_cost:
        best_cost = cost2
        best_A = A2
        best_counts = counts2

    # Start 3: Random moderate size
    A3 = set(random.sample(range(129), 60))
    counts3 = compute_counts_from_list(A3)
    cost3 = get_cost(counts3)
    if cost3 < best_cost:
        best_cost = cost3
        best_A = A3
        best_counts = counts3

    # If best is still invalid, use full set as fallback
    if best_cost > 10000:
        best_A = set(range(129))
        best_counts = compute_counts_from_list(best_A)
        best_cost = get_cost(best_counts)

    # Local search from best start
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_A, best_counts, best_cost = local_search(
            best_A, best_counts, best_cost, deadline - 2.0
        )

    # Final verification
    final_counts = compute_counts_from_list(best_A)
    if np.any(final_counts == 0):
        # Fallback to full set
        return list(range(129))

    return list(best_A)
# EVOLVE-BLOCK-END
