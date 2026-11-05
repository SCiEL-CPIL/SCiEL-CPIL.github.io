import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s margin before 150s kill

    def triple_index(a, b, c):
        """Compute index of triple (a,b,c) in lexicographic order of all triples of range(n)."""
        # We'll precompute a lookup instead
        pass

    def build_cliques(n):
        """Precompute K4 and K5 edge-index arrays as numpy arrays."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_edges = len(triples)

        # K4s: each is 4 edges
        k4_list = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idxs = (
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)],
            )
            k4_list.append(idxs)
        k4s = np.array(k4_list, dtype=np.int32) if k4_list else np.empty((0, 4), dtype=np.int32)

        # K5s: each is 10 edges
        k5_list = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5_list.append(tuple(idxs))
        k5s = np.array(k5_list, dtype=np.int32) if k5_list else np.empty((0, 10), dtype=np.int32)

        return k4s, k5s, num_edges

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_k4 = 0
        if len(k4s) > 0:
            # colors[k4s] gives shape (num_k4s, 4)
            k4_colors = colors[k4s]
            red_k4 = int(np.all(k4_colors == 0, axis=1).sum())
        blue_k5 = 0
        if len(k5s) > 0:
            k5_colors = colors[k5s]
            blue_k5 = int(np.all(k5_colors == 1, axis=1).sum())
        return red_k4, blue_k5, red_k4 + blue_k5

    def local_search(n, k4s, k5s, num_edges, max_time, seed=0):
        """
        Run local search / simulated annealing to find a valid coloring.
        Returns (colors, cost) or (None, inf) if no valid found.
        """
        rng = np.random.RandomState(seed)
        colors = rng.randint(0, 2, num_edges, dtype=np.int8)

        # Precompute for each edge, which K4s and K5s it belongs to
        # This is expensive but necessary for fast delta computation
        # For n=34: num_edges=5984, k4s=46376, k5s=278256
        # Each edge is in C(n-3,1) K4s = n-3 K4s and C(n-3,2) K5s = (n-3)(n-4)/2 K5s

        # Build edge-to-clique mapping using numpy
        # For K4s: k4s has shape (num_k4s, 4)
        # For each edge e, find all k4 indices where e appears
        edge_to_k4 = [[] for _ in range(num_edges)]
        if len(k4s) > 0:
            for i in range(len(k4s)):
                for e in k4s[i]:
                    edge_to_k4[e].append(i)
        edge_to_k4 = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in edge_to_k4]

        edge_to_k5 = [[] for _ in range(num_edges)]
        if len(k5s) > 0:
            for i in range(len(k5s)):
                for e in k5s[i]:
                    edge_to_k5[e].append(i)
        edge_to_k5 = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in edge_to_k5]

        # Precompute: for each K4, the 3 "other" edges for each position
        # k4s[:, j] gives the j-th edge of each K4
        # For edge at position j in K4 i, the other 3 are k4s[i, [0,1,2,3] \ {j}]

        # Initial cost
        red_k4, blue_k5, current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()

        if current_cost == 0:
            return colors, 0

        # Simulated annealing parameters
        temp = 1.0
        cool_rate = 0.998
        min_temp = 0.05

        steps = 0
        while time.time() < max_time:
            steps += 1
            # Pick a random edge to flip
            e = rng.randint(num_edges)
            old_c = colors[e]
            new_c = 1 - old_c

            delta = 0

            # Check K4s containing e
            k4_idxs = edge_to_k4[e]
            if len(k4_idxs) > 0:
                # For each K4 containing e, check if flipping changes red-K4 status
                # Get all edges in these K4s
                k4_edges = k4s[k4_idxs]  # shape (num_k4s_containing_e, 4)
                # Find which position e is in each K4
                # Actually, let's just check: for each K4, count how many of its 4 edges are red
                # If old_c == 0 (red): flipping to blue
                #   If all 4 were red -> was a red K4, now not. delta -= 1
                #   If exactly 3 others red + this was red -> same as above
                #   If this was the only red or not all red -> no change to red K4 count
                # If old_c == 1 (blue): flipping to red
                #   If all 4 others... wait, this edge is blue so K4 wasn't all red
                #   If other 3 are red and this becomes red -> new red K4. delta += 1

                # Vectorized: for each K4 in k4_edges, check the other 3 edges
                # Create a mask for position of e in each K4
                # k4_edges contains e at some position in each row
                # Let's find positions
                pos = np.zeros((len(k4_idxs), 4), dtype=bool)
                for j in range(4):
                    pos[:, j] = (k4_edges[:, j] == e)
                # pos should have exactly one True per row

                # Get the other 3 edges
                # For each row, the other edges are those where pos is False
                other_mask = ~pos  # shape (num_k4s, 4), True for non-e edges
                # Count red among other edges
                other_red = (k4_edges == 0) & other_mask  # shape (num_k4s, 4)
                other_red_count = other_red.sum(axis=1)  # shape (num_k4s,)

                if old_c == 0:
                    # e was red, becomes blue
                    # Red K4 broken if all 4 were red, i.e., other 3 were red
                    broken = (other_red_count == 3)
                    delta -= int(broken.sum())
                else:
                    # e was blue, becomes red
                    # Red K4 created if other 3 were red
                    created = (other_red_count == 3)
                    delta += int(created.sum())

            # Check K5s containing e
            k5_idxs = edge_to_k5[e]
            if len(k5_idxs) > 0:
                k5_edges = k5s[k5_idxs]  # shape (num_k5s_containing_e, 10)
                # Find position of e in each K5
                pos5 = np.zeros((len(k5_idxs), 10), dtype=bool)
                for j in range(10):
                    pos5[:, j] = (k5_edges[:, j] == e)
                other_mask5 = ~pos5
                other_blue = (k5_edges == 1) & other_mask5
                other_blue_count = other_blue.sum(axis=1)

                if old_c == 1:
                    # e was blue, becomes red
                    # Blue K5 broken if all 10 were blue, i.e., other 9 were blue
                    broken5 = (other_blue_count == 9)
                    delta -= int(broken5.sum())
                else:
                    # e was red, becomes blue
                    # Blue K5 created if other 9 were blue
                    created5 = (other_blue_count == 9)
                    delta += int(created5.sum())

            new_cost = current_cost + delta

            # Acceptance
            accept = False
            if new_cost < current_cost:
                accept = True
            elif new_cost == current_cost:
                accept = rng.rand() < 0.5
            else:
                # Simulated annealing: accept with probability exp(-delta/temp)
                if delta > 0:
                    accept = rng.rand() < np.exp(-delta / max(temp, 1e-10))

            if accept:
                colors[e] = new_c
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return colors, 0

            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp

        return best_colors, best_cost

    # Try to find valid colorings for decreasing n
    for n in [34, 33, 32, 31, 30, 28, 25, 20, 15]:
        if time.time() > deadline:
            break

        try:
            k4s, k5s, num_edges = build_cliques(n)
        except Exception:
            continue

        # Multiple restarts
        best_result = None
        best_cost = float('inf')

        # Allocate time for this n
        time_remaining = deadline - time.time()
        num_restarts = min(20, max(3, int(time_remaining / 5.0)))
        time_per_restart = time_remaining / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
            t_max = min(time_per_restart, deadline - time.time() - 1.0)
            if t_max < 1.0:
                break
            try:
                colors, cost = local_search(n, k4s, k5s, num_edges, t_max, seed=r * 12345 + 678)
                if cost < best_cost:
                    best_cost = cost
                    best_result = colors
                    if cost == 0:
                        return {'n': n, 'colors': colors.tolist()}
            except Exception:
                continue

        if best_result is not None and best_cost == 0:
            return {'n': n, 'colors': best_result.tolist()}

    # Fallback: return a small valid coloring
    # For n=4, we need to avoid all-red K4. Use one blue edge.
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}
