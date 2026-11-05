import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit

    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)

    # Compute squared distance matrix
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)

    # For each point P, precompute: for each other point Q, the set of all R
    # such that {P, Q, R} forms an isosceles triangle.
    # Instead of storing this explicitly (too much memory), we'll compute on the fly
    # but use efficient data structures.
    #
    # Key data structure: for each P, a dict mapping distance d -> sorted list of points
    # at that distance from P. This allows O(1) lookup of "which points are at distance d from P".
    #
    # For can_add(P, S):
    #   For each Q in S:
    #     - P is apex: check if any R in S (R != Q) has dist(P,R) == dist(P,Q)
    #       => check if len(set(S) & points_at_dist[P][dist(P,Q)]) > 1 (excluding Q)
    #     - Q is apex: check if any R in S (R != P, Q) has dist(Q,R) == dist(Q,P)
    #       => check if len(set(S) & points_at_dist[Q][dist(Q,P)]) > 1 (excluding P)
    #     - R is apex: for some R in S (R != P, Q), dist(R,P) == dist(R,Q)
    #       => this is harder; need to check all R in S
    #
    # The "R is apex" condition is the bottleneck. Let's precompute for each unordered pair
    # {P, Q} the set of points R where dist(R,P) == dist(R,Q). But that's 4096*4095/2 pairs
    # each with up to 4096 points - too much.
    #
    # Alternative: just check all three conditions by iterating, but use numpy for speed.
    # The original approach with dist_sets is actually quite good for conditions 1 and 2.
    # Let's keep that and optimize condition 3.

    # Build points_at_dist for efficient lookup
    points_at_dist = [{} for _ in range(NUM_POINTS)]
    for P in range(NUM_POINTS):
        for Q in range(NUM_POINTS):
            if P != Q:
                d = int(dist2[P, Q])
                if d not in points_at_dist[P]:
                    points_at_dist[P][d] = []
                points_at_dist[P][d].append(Q)
    # Convert to sets for O(1) membership testing
    for P in range(NUM_POINTS):
        for d in points_at_dist[P]:
            points_at_dist[P][d] = set(points_at_dist[P][d])

    def can_add(P, S, dist_sets, S_set):
        """Check if P can be added to set S without forming an isosceles triangle.
        S: list of point indices
        dist_sets: list of sets, dist_sets[i] = set of distances from S[i] to other points in S
        S_set: set of points in S for fast lookup
        """
        n = len(S)
        if n == 0:
            return True
        if n == 1:
            return True

        # Get distances from P to all points in S
        d_vals = dist2[P, np.array(S, dtype=np.int32)]

        # Condition 1: P is not apex - all distances from P to points in S must be distinct
        if n > 1:
            sorted_d = np.sort(d_vals)
            if np.any(sorted_d[:-1] == sorted_d[1:]):
                return False

        # Condition 2: No Q in S is apex with P and another R in S
        # For each i, Q = S[i], check if dist(P, Q) is already in dist_sets[i]
        # (meaning there's already some R in S with dist(Q, R) == dist(Q, P))
        for i in range(n):
            d = int(d_vals[i])
            if d in dist_sets[i]:
                return False

        return True

    def add_point(P, S, dist_sets, S_set):
        """Add point P to set S, updating dist_sets and S_set."""
        S.append(P)
        S_set.add(P)
        new_dist_set = set()
        if len(S) > 1:
            d_vals = dist2[P, np.array(S[:-1], dtype=np.int32)]
            for i, d in enumerate(d_vals):
                d_int = int(d)
                dist_sets[i].add(d_int)
                new_dist_set.add(d_int)
        dist_sets.append(new_dist_set)

    def remove_point(P, S, dist_sets, S_set):
        """Remove point P from set S, updating dist_sets and S_set."""
        idx = S.index(P)
        for i in range(len(S)):
            if i == idx:
                continue
            d = int(dist2[P, S[i]])
            dist_sets[i].discard(d)
        S.pop(idx)
        dist_sets.pop(idx)
        S_set.discard(P)

    def greedy_construction(order, time_limit):
        """Greedy construction with time limit."""
        S = []
        dist_sets = []
        S_set = set()
        in_S = np.zeros(NUM_POINTS, dtype=bool)

        for P in order:
            if time.time() > time_limit:
                break
            if in_S[P]:
                continue

            if can_add(P, S, dist_sets, S_set):
                in_S[P] = True
                add_point(P, S, dist_sets, S_set)

        return S, dist_sets, S_set, in_S

    def local_search(S, dist_sets, S_set, in_S, time_limit):
        """Local search with add/remove moves."""
        best_S = S[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        best_S_set = S_set.copy()
        best_size = len(S)

        no_improve = 0
        max_no_improve = 40

        while time.time() < time_limit and no_improve < max_no_improve:
            added = False

            # Try to add a random point
            complement_size = NUM_POINTS - len(S)
            if complement_size > 0:
                batch_size = min(300, complement_size)
                candidates = random.sample(range(NUM_POINTS), batch_size)
                for P in candidates:
                    if in_S[P]:
                        continue

                    if can_add(P, S, dist_sets, S_set):
                        in_S[P] = True
                        add_point(P, S, dist_sets, S_set)

                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                            best_dist_sets = [s.copy() for s in dist_sets]
                            best_S_set = S_set.copy()
                            no_improve = 0
                        else:
                            no_improve += 1

                        added = True
                        break

            if not added:
                no_improve += 1

                # Perturbation: remove 1-2 random points, try to add 2-3
                if len(S) > 3:
                    num_remove = random.choice([1, 1, 2])
                    for _ in range(num_remove):
                        if not S:
                            break
                        rem_idx = random.randrange(len(S))
                        rem_P = S[rem_idx]
                        in_S[rem_P] = False
                        remove_point(rem_P, S, dist_sets, S_set)

                    # Try to add points
                    added_count = 0
                    target_add = num_remove + 1
                    complement_size = NUM_POINTS - len(S)
                    if complement_size > 0:
                        batch_size = min(150, complement_size)
                        candidates = random.sample(range(NUM_POINTS), batch_size)
                        for P in candidates:
                            if in_S[P]:
                                continue

                            if can_add(P, S, dist_sets, S_set):
                                in_S[P] = True
                                add_point(P, S, dist_sets, S_set)
                                added_count += 1

                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = S[:]
                                    best_dist_sets = [s.copy() for s in dist_sets]
                                    best_S_set = S_set.copy()
                                    no_improve = 0

                                if added_count >= target_add:
                                    break

                    no_improve = 0

        return best_S, best_dist_sets, best_S_set

    # Main loop
    best_set = []
    best_size = 0
    rng = np.random.default_rng(42)

    # Multiple restarts with different random seeds
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Create random order
        order = rng.permutation(NUM_POINTS).tolist()

        # Greedy construction
        greedy_time = time.time() + min(5, (deadline - time.time()) / 5)
        S, dist_sets, S_set, in_S = greedy_construction(order, greedy_time)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

        # Local search
        remaining = deadline - time.time()
        if remaining > 10:
            search_time = min(15, remaining / 5)
            S, dist_sets, S_set = local_search(S, dist_sets, S_set, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]

        # Early exit if we've found a good solution
        if best_size > 15:
            break

    # Convert indices to coordinates
    result = [(idx % N, idx // N) for idx in best_set]

    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
