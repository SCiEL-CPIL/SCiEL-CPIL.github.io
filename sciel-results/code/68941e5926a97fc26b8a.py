import time
import random
import math
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a vector-packing formulation with Lloyd's algorithm (k-means) on difference
    space to partition the difference set into disjoint clusters, then refines each
    row via greedy construction within its cluster.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope
    lb = total_diffs

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    # ---- Vector Packing / Lloyd's Algorithm Approach ----
    # Represent each row by the set of differences it occupies.
    # Use k-means-like clustering on the difference space to partition
    # differences into n_rows disjoint groups, then construct rows within each group.

    def _diffs_to_vector(diffs, scope):
        """Convert a set of differences to a binary vector of length scope+1."""
        vec = np.zeros(scope + 1, dtype=np.uint8)
        for d in diffs:
            if d <= scope:
                vec[d] = 1
        return vec

    def _vector_to_diffs(vec):
        """Convert a binary vector back to a set of differences."""
        return set(np.nonzero(vec)[0].tolist())

    def _lloyd_partition(scope, n_rows, k, max_iter=50):
        """
        Use Lloyd's algorithm to partition the difference space [1, scope] into
        n_rows clusters. Each cluster should contain exactly diffs_per_row values.
        Returns a list of n_rows sets, each being a cluster of difference values.
        """
        diffs_per_row = k * (k + 1) // 2
        total = n_rows * diffs_per_row

        # Initialize: assign differences randomly to clusters
        all_diffs = list(range(1, scope + 1))
        random.shuffle(all_diffs)

        # We only need total_diffs differences, so take the first total_diffs
        selected_diffs = all_diffs[:total]

        # Initialize clusters by random assignment
        clusters = [set() for _ in range(n_rows)]
        for i, d in enumerate(selected_diffs):
            clusters[i % n_rows].add(d)

        # Lloyd's iterations
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Step 1: Compute centroids (mean of each cluster)
            centroids = []
            for c in clusters:
                if c:
                    centroids.append(np.mean(list(c)))
                else:
                    centroids.append(0)
            centroids = np.array(centroids)

            # Step 2: Reassign each difference to nearest centroid
            new_clusters = [set() for _ in range(n_rows)]
            counts = [0] * n_rows
            for d in selected_diffs:
                # Find nearest centroid
                dists = np.abs(centroids - d)
                min_idx = np.argmin(dists)
                new_clusters[min_idx].add(d)
                counts[min_idx] += 1

            # Step 3: Balance clusters - move excess to deficient
            for i in range(n_rows):
                while counts[i] > diffs_per_row:
                    # Move largest element from cluster i to a deficient cluster
                    target = min(range(n_rows), key=lambda j: counts[j])
                    if counts[target] >= diffs_per_row:
                        break
                    # Move the largest element
                    max_elem = max(new_clusters[i])
                    new_clusters[i].remove(max_elem)
                    new_clusters[target].add(max_elem)
                    counts[i] -= 1
                    counts[target] += 1
                while counts[i] < diffs_per_row:
                    # Take smallest element from a surplus cluster
                    source = max(range(n_rows), key=lambda j: counts[j])
                    if counts[source] <= diffs_per_row:
                        break
                    min_elem = min(new_clusters[source])
                    new_clusters[source].remove(min_elem)
                    new_clusters[i].add(min_elem)
                    counts[source] -= 1
                    counts[i] += 1

            # Check convergence
            if all(new_clusters[i] == clusters[i] for i in range(n_rows)):
                break
            clusters = new_clusters

        return clusters

    def _construct_row_from_cluster(cluster, k, scope):
        """
        Given a set of allowed difference values (the cluster), try to construct
        a row of k+1 elements starting at 0 such that all pairwise differences
        are in the cluster.
        """
        allowed = set(cluster)
        if not allowed:
            return None

        # Greedy construction with backtracking
        def _backtrack(row, pos, used_local):
            if pos == k + 1:
                return row[:]

            start = row[pos - 1] + 1
            end = min(scope, 200) + 1

            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                # Check all new diffs are in allowed and not in used_local
                valid = True
                for d in new_diffs:
                    if d not in allowed or d in used_local:
                        valid = False
                        break
                if not valid:
                    continue

                row.append(v)
                result = _backtrack(row, pos + 1, used_local | new_diffs)
                if result is not None:
                    return result
                row.pop()

            return None

        row = [0]
        result = _backtrack(row, 1, set())
        return result

    def _improve_with_lloyd(n_rows, k, scope, deadline):
        """
        Try to improve the solution using Lloyd's algorithm partitioning.
        Returns improved rows and scope, or None if no improvement.
        """
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break

            # Create a partition of the difference space
            clusters = _lloyd_partition(scope, n_rows, k, max_iter=30)

            if time.time() > deadline - 5:
                break

            # Try to construct rows from each cluster
            rows = []
            valid = True
            for c in clusters:
                row = _construct_row_from_cluster(c, k, scope)
                if row is None:
                    valid = False
                    break
                rows.append(row)

            if not valid:
                continue

            # Validate the complete solution
            if _validate(rows, n_rows, k):
                new_scope = _global_max(rows)
                return rows, new_scope

        return None, None

    # ---- Hybrid Local Search (kept as fallback/refinement) ----
    def _local_search_polish(rows, n_rows, k, deadline):
        """
        Fast local search to polish the solution by trying small decreases.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                if time.time() > deadline:
                    break
                for pi in range(1, k + 1):
                    old_val = current_rows[ri][pi]
                    for delta in range(1, 6):
                        new_val = old_val - delta
                        if new_val <= current_rows[ri][pi - 1]:
                            continue
                        if pi < k and new_val >= current_rows[ri][pi + 1]:
                            continue

                        cand_row = current_rows[ri][:]
                        cand_row[pi] = new_val

                        new_diffs = set()
                        valid = True
                        n = len(cand_row)
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = cand_row[j] - cand_row[i]
                                if d in new_diffs:
                                    valid = False
                                    break
                                new_diffs.add(d)

                        if not valid:
                            continue

                        for d in new_diffs:
                            if d in all_used and d not in row_diffs[ri]:
                                valid = False
                                break

                        if not valid:
                            continue

                        cand_row_max = max(cand_row)
                        if cand_row_max < row_maxes[ri]:
                            all_used -= row_diffs[ri]
                            all_used |= new_diffs
                            current_rows[ri] = cand_row
                            row_diffs[ri] = new_diffs
                            row_maxes[ri] = cand_row_max
                            current_val = max(row_maxes)
                            improved = True
                            break

        return current_rows, current_val

    # ---- Main optimization loop ----
    # First, try Lloyd's algorithm with various scope values
    current_scope = best_scope
    for scope_attempt in range(current_scope, max(lb, current_scope - 20), -1):
        if time.time() > deadline - 30:
            break

        # Try to find a valid solution with this scope
        result_rows, result_scope = _improve_with_lloyd(n_rows, k, scope_attempt, deadline)

        if result_rows is not None and result_scope < best_scope:
            best_scope = result_scope
            best_solution = result_rows

    # Polish the best solution found
    if time.time() < deadline - 10:
        polish_deadline = min(deadline - 5, time.time() + 20.0)
        polished_rows, polished_scope = _local_search_polish(
            [r[:] for r in best_solution], n_rows, k, polish_deadline
        )
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_solution = polished_rows

    # Final validation
    if not _validate(best_solution, n_rows, k):
        # Fallback to greedy
        best_solution = greedy_construct(n_rows, k)
        best_scope = max(max(row) for row in best_solution)

    return best_solution

# EVOLVE-BLOCK-END
