import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import maximum_flow


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    N = 1000

    # Maximum flow formulation:
    # Left nodes: candidate integers 1..N (nodes 1..N)
    # Right nodes: sums 2..2N (nodes N+1..3N)
    # Source: node 0
    # Sink: node 3N+1
    #
    # For each candidate x, add edge source->x with capacity 1
    # For each sum s, add edge s->sink with capacity 2
    # For each x and each y in {x, x+1, ..., N} where x+y <= 2N:
    #   add edge x -> (N + x + y) with capacity 1
    #
    # Wait, this doesn't work directly because the constraint is on the NUMBER
    # of representations, not on individual edges. Let me reconsider.
    #
    # Actually, the constraint is: for each sum s, the number of pairs (a,b) with
    # a <= b, a+b=s, and both in S is at most 2.
    #
    # This is NOT a simple bipartite matching/flow problem because the constraint
    # is on the count of edges incident to each sum node, and each left node can
    # connect to multiple sum nodes.
    #
    # Let me think about this differently. The problem is actually an integer
    # programming problem. But we can use a greedy approach with the flow insight.
    #
    # Actually, let me reconsider the structure. If we select a set S, then for
    # each s, the number of representations is the number of pairs (a,b) in S
    # with a <= b and a+b = s. This is equivalent to: for each s, count how many
    # a in S such that s-a is also in S and a <= s-a.
    #
    # This is NOT a standard max flow problem. Let me use a different approach:
    # iterative improvement with better heuristics.
    #
    # Better approach: Use a priority-based greedy where we always add the element
    # that creates the "least constrained" sums, then do local search.

    best_S = set()
    best_size = 0

    # Strategy 1: Greedy with lookahead
    def _greedy_with_lookahead(N, deadline, seed=42):
        import random
        random.seed(seed)
        S = set()
        counts = np.zeros(2 * N + 1, dtype=np.int32)

        # Precompute for each x, which sums it would affect
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            # Check if x can be added
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            else:
                # Check all y in S where x + y <= 2*N
                if S:
                    S_arr = np.array(list(S), dtype=np.int32)
                    # Only consider y >= x to avoid double counting, but we need
                    # to check all y since x+y might already have 2 reps
                    sums = x + S_arr
                    mask = sums <= 2 * N
                    if np.any(counts[sums[mask]] >= 2):
                        valid = False

            if valid:
                S.add(x)
                counts[2 * x] += 1
                if S:
                    S_arr = np.array(list(S), dtype=np.int32)
                    mask = S_arr != x
                    if np.any(mask):
                        sums = x + S_arr[mask]
                        valid_sums = sums <= 2 * N
                        counts[sums[valid_sums]] += 1

        return S, counts

    # Strategy 2: Multiple random orderings
    def _greedy_random_order(N, deadline, seed=42):
        import random
        random.seed(seed)
        S = set()
        counts = np.zeros(2 * N + 1, dtype=np.int32)

        order = list(range(1, N + 1))
        random.shuffle(order)

        for x in order:
            if time.time() > deadline:
                break
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            else:
                if S:
                    S_arr = np.array(list(S), dtype=np.int32)
                    sums = x + S_arr
                    if np.any(counts[sums] >= 2):
                        valid = False

            if valid:
                S.add(x)
                counts[2 * x] += 1
                if S:
                    S_arr = np.array(list(S), dtype=np.int32)
                    mask = S_arr != x
                    if np.any(mask):
                        sums = x + S_arr[mask]
                        counts[sums] += 1

        return S, counts

    # Strategy 3: Local search
    def _local_search(S, counts, N, deadline, seed=42):
        import random
        random.seed(seed)
        current_S = set(S)
        current_counts = counts.copy()
        best_S = frozenset(current_S)
        best_size = len(current_S)

        in_S = np.zeros(N + 1, dtype=bool)
        for x in current_S:
            in_S[x] = True

        it = 0
        while time.time() < deadline:
            it += 1

            # Try to add a random element
            available_mask = ~in_S
            available_mask[0] = False
            available_indices = np.where(available_mask)[0]
            if len(available_indices) == 0:
                break

            y = int(random.choice(available_indices))

            valid = True
            if current_counts[2 * y] >= 2:
                valid = False
            elif len(current_S) > 0:
                S_arr = np.array(sorted(current_S), dtype=np.int32)
                sums = y + S_arr
                if np.any(current_counts[sums] >= 2):
                    valid = False

            if valid:
                current_counts[2 * y] += 1
                if len(current_S) > 0:
                    S_arr = np.array(sorted(current_S), dtype=np.int32)
                    sums = y + S_arr
                    current_counts[sums] += 1
                current_S.add(y)
                in_S[y] = True

                if len(current_S) > best_size:
                    best_S = frozenset(current_S)
                    best_size = len(current_S)
            else:
                # Try swap
                if len(current_S) > 0:
                    S_arr = np.array(sorted(current_S), dtype=np.int32)
                    x = int(random.choice(S_arr))

                    # Remove x
                    current_counts[2 * x] -= 1
                    mask = S_arr != x
                    if np.any(mask):
                        sums = x + S_arr[mask]
                        current_counts[sums] -= 1
                    current_S.remove(x)
                    in_S[x] = False
                    S_arr = S_arr[S_arr != x]

                    # Try to add y
                    valid2 = True
                    if current_counts[2 * y] >= 2:
                        valid2 = False
                    elif len(current_S) > 0:
                        sums = y + S_arr
                        if np.any(current_counts[sums] >= 2):
                            valid2 = False

                    if valid2:
                        current_counts[2 * y] += 1
                        if len(current_S) > 0:
                            sums = y + S_arr
                            current_counts[sums] += 1
                        current_S.add(y)
                        in_S[y] = True

                        if len(current_S) > best_size:
                            best_S = frozenset(current_S)
                            best_size = len(current_S)
                    else:
                        # Rollback
                        current_counts[2 * x] += 1
                        if len(current_S) > 0:
                            sums = x + S_arr
                            current_counts[sums] += 1
                        current_S.add(x)
                        in_S[x] = True

        return set(best_S)

    # Run multiple strategies
    strategies = [
        ("greedy_asc", lambda: _greedy_with_lookahead(N, deadline, 42)),
        ("greedy_desc", lambda: _greedy_random_order(N, deadline, 123)),
        ("greedy_rand1", lambda: _greedy_random_order(N, deadline, 456)),
        ("greedy_rand2", lambda: _greedy_random_order(N, deadline, 789)),
    ]

    for name, func in strategies:
        if time.time() > deadline - 20:
            break
        S, counts = func()
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

        # Local search
        if time.time() < deadline - 10 and best_S:
            improved = _local_search(best_S, counts, N, deadline - 5, seed=42 + len(strategies))
            if len(improved) > best_size:
                best_S = improved
                best_size = len(improved)

    # Final local search
    if time.time() < deadline - 5 and best_S:
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        S_list = sorted(best_S)
        for x in best_S:
            counts[2 * x] += 1
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                counts[S_list[i] + S_list[j]] += 1

        final_S = _local_search(best_S, counts, N, deadline, seed=999)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(final_S)

    return list(best_S)
