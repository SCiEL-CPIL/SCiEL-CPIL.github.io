"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_N = P['n']
_EDGE_ORDER = P['edge_order']
_NEDGES = len(_EDGE_ORDER)

# Precompute edge -> position (1-based)
_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _EDGE_POS[(_a, _b)] = _i
    _EDGE_POS[(_b, _a)] = _i

# Build adjacency list per vertex sorted by edge position
_ADJ = [[] for _ in range(_N)]
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _ADJ[_a].append((_i, _b))
    _ADJ[_b].append((_i, _a))
for _v in range(_N):
    _ADJ[_v].sort()  # sort by position


def _find_cycle_for_k(K, time_limit=None, start_time=None, rng=None):
    """Find a Hamilton cycle using only edges with position <= K.
    Returns a list of vertices (the cycle order) or None."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0
    if rng is None:
        rng = random.Random(614)

    # Filter adjacency: only edges with pos <= K
    adj = [[] for _ in range(_N)]
    for v in range(_N):
        for (pos, u) in _ADJ[v]:
            if pos <= K:
                adj[v].append(u)
        # sort by degree (ascending) for heuristic - fewer options first
        adj[v].sort(key=lambda u: len(adj[u]))

    # Check: every vertex must have degree >= 2 in the filtered graph
    for v in range(_N):
        if len(adj[v]) < 2:
            return None

    # Try to find Hamilton cycle via DFS from vertex 0
    # We fix vertex 0 as the start to break symmetry.
    # State: current vertex, visited bitmask, path
    # Use iterative DFS with backtracking and degree-based pruning.

    full_mask = (1 << _N) - 1
    start = 0
    start_mask = 1 << start

    # Precompute neighbor bitmasks for fast iteration
    adj_masks = [0] * _N
    for v in range(_N):
        m = 0
        for u in adj[v]:
            m |= (1 << u)
        adj_masks[v] = m

    # DFS with recursion limit handling - use iterative approach
    # For n=24, we need a good heuristic.
    # Key insight: in a Hamilton cycle, each vertex has exactly 2 neighbors in the cycle.
    # We build the cycle by extending a path from vertex 0, and at the end connect back to 0.

    # Use a stack-based DFS
    # Each stack entry: (vertex, visited_mask, path_list, next_neighbor_idx)
    # But path_list is expensive to copy. Instead, use a single path array and backtrack.

    path = [0] * _N
    path[0] = 0
    visited = start_mask
    neighbor_idx = [0] * _N  # index into adj[v] for next neighbor to try

    # We'll use a recursive-style approach with explicit stack
    # Actually for n=24, let's use a smarter approach:
    # Pick a starting edge from vertex 0 (two choices for the two neighbors),
    # then extend the path.

    # Better: try all pairs of neighbors of vertex 0 as the "endpoints" of the path
    # that will connect back to 0. Actually, the cycle goes 0 -> ... -> 0.
    # So from 0, we pick one neighbor to start the path, and the last vertex must be
    # another neighbor of 0.

    # Let's do: fix the first step from 0 to some neighbor n1.
    # Then we need a Hamilton path from n1 to n2 where n2 is also a neighbor of 0,
    # visiting all other vertices.

    # For efficiency, try different starting neighbors.

    best_cycle = None

    def _dfs_iterative(first_neighbor):
        """Find Hamilton path from first_neighbor to some neighbor of 0 (excluding 0 itself
        except at the end), visiting all vertices."""
        # Start: path = [0, first_neighbor], visited includes 0 and first_neighbor
        p = [0] * _N
        p[0] = 0
        p[1] = first_neighbor
        v_mask = (1 << 0) | (1 << first_neighbor)
        depth = 1  # we've placed 2 vertices, next position is index 2

        # Stack of (vertex_at_depth, depth, next_neighbor_index)
        # We'll use a list-based approach
        stack = []
        # Current state: we're at first_neighbor (depth=1), need to go to depth 2..23
        # Then check if the last vertex connects to 0.

        # Use explicit stack with backtracking
        # state: (current_vertex, depth, next_idx)
        # path is shared, we modify and backtrack

        cur = first_neighbor
        d = 1
        nidx = 0
        stack.append((cur, d, nidx))

        while stack:
            cur, d, nidx = stack[-1]

            if d == _N - 1:
                # All vertices placed. Check if cur connects to 0.
                if 0 in adj[cur]:
                    return list(p[:d+1])
                # Need to backtrack
                stack.pop()
                if not stack:
                    return None
                # Backtrack: increment neighbor index of parent
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                # Mark cur as unvisited
                v_mask &= ~(1 << cur)
                # Remove cur from path
                p[d] = 0
                continue

            # Try neighbors of cur starting from nidx
            found_next = False
            while nidx < len(adj[cur]):
                nxt = adj[cur][nidx]
                if not (v_mask & (1 << nxt)):
                    # Valid next vertex
                    nidx += 1
                    stack[-1] = (cur, d, nidx)
                    # Move to nxt
                    v_mask |= (1 << nxt)
                    p[d+1] = nxt
                    stack.append((nxt, d+1, 0))
                    found_next = True
                    break
                else:
                    nidx += 1
                    stack[-1] = (cur, d, nidx)

            if not found_next:
                # No more neighbors, backtrack
                stack.pop()
                if not stack:
                    return None
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                p[d] = 0

        return None

    # Try each neighbor of 0 as the first step
    # Order neighbors by their degree (ascending) for better pruning
    neighbors_of_0 = sorted(adj[0], key=lambda u: len(adj[u]))

    for n1 in neighbors_of_0:
        if time.time() - start_time > time_limit - 2:
            break
        # Try to find path from n1 back to 0
        result = _dfs_iterative(n1)
        if result is not None:
            return result  # [0, n1, ..., last] where last connects to 0

    return None


def _score_cycle(cycle):
    """Compute the max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i+1) % n]
        pos = _EDGE_POS[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    # First, establish an initial upper bound with a greedy/heuristic cycle
    # Try the identity permutation as a baseline
    baseline_cycle = list(range(_N))
    baseline_score = _score_cycle(baseline_cycle)

    best_cycle = baseline_cycle
    best_score = baseline_score

    # Binary search for the minimum K where a Hamilton cycle exists
    # Lower bound: at least 24 (need 24 edges, positions are 1-based)
    # Upper bound: best_score from baseline
    lo = 24
    hi = best_score

    # First, try to quickly check if very small K works
    # Do a few direct checks at small values
    # Then binary search

    # Quick check: try K values from a small range first
    # The theoretical minimum is when we can find 24 edges forming a cycle
    # among the earliest positions.

    # Binary search
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if time.time() > deadline - 5:
            break
        cycle = _find_cycle_for_k(mid, time_limit=deadline - time.time(), start_time=start_time)
        if cycle is not None:
            # Verify the cycle
            if len(set(cycle)) == _N and len(cycle) == _N:
                s = _score_cycle(cycle)
                if s <= mid:
                    best_cycle = cycle
                    best_score = s
                    hi = mid
                else:
                    # Cycle found but score > mid? Shouldn't happen if we filtered correctly
                    # But just in case, use it if better
                    if s < best_score:
                        best_cycle = cycle
                        best_score = s
                    hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1

    # After binary search, try to improve by checking lo (the minimum feasible K)
    if time.time() < deadline - 3:
        cycle = _find_cycle_for_k(lo, time_limit=deadline - time.time(), start_time=start_time)
        if cycle is not None and len(set(cycle)) == _N:
            s = _score_cycle(cycle)
            if s < best_score:
                best_cycle = cycle
                best_score = s

    # Final: try a few more random restarts with the best known K to see if we find a better cycle
    # Actually the score is determined by the max edge position, so any valid cycle for K=lo has score <= lo.
    # The score is exactly the max position among the cycle's edges.
    # So if we found a cycle with all edges <= lo, its score is at most lo.
    # To get an even better score, we'd need a cycle with all edges < lo, which we already ruled out.

    # Return the best cycle found
    return best_cycle
# EVOLVE-BLOCK-END
