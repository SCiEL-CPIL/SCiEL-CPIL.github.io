"""Maximum independent set in a 3-uniform hypergraph of equal-lcm triples,
solved with branch-and-bound and bitset-based conflict tracking."""
# EVOLVE-BLOCK-START
from math import lcm as _lcm

def construct():
    N = 128
    # vertices 1..128 -> index 0..127
    vals = list(range(1, N + 1))

    # Precompute lcm matrix
    lcm_mat = [[_lcm(vals[i], vals[j]) for j in range(N)] for i in range(N)]

    # bad_with[i][j] = bitset of vertices k (k distinct from i,j) such that
    # (i, j, k) is a bad triple: lcm(i,j)==lcm(i,k)==lcm(j,k)
    bad_with = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            L = lcm_mat[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if lcm_mat[i][k] == L and lcm_mat[j][k] == L:
                    mask |= (1 << k)
            bad_with[i][j] = mask
            bad_with[j][i] = mask

    # For each vertex v, precompute the list of neighbors (other vertices) and
    # the "pair activation" structure. When v is added to S, a candidate u is
    # blocked if there exists s in S with (s, v, u) a bad triple, i.e.
    # u in bad_with[s][v]. So we need, for fixed v and a set S, the union of
    # bad_with[s][v] over s in S. We precompute bad_with[s][v] as bitsets.

    import sys
    sys.setrecursionlimit(100000)

    best = [0]
    best_set = [0]  # bitset of best solution

    # Order vertices to help pruning: process in a fixed order.
    # We'll do a recursive search. To keep it fast, use a simple greedy
    # upper bound: the number of still-available candidates.

    # We maintain:
    #   committed: bitset of vertices already chosen (in S)
    #   avail: bitset of candidates that can still be added (not yet decided,
    #           and not blocked)
    #   blocked: bitset of candidates that are blocked (cannot be added)
    # The remaining pool = avail (candidates not blocked and not committed).
    # When we add v to committed, new blocked = union over s in committed
    #   of bad_with[s][v], intersected with the remaining pool.

    # To make the search tractable, we use a greedy ordering and a strong
    # upper bound: |committed| + popcount(avail). Prune when that <= best.

    def popcount(x):
        return bin(x).count('1')

    # Precompute, for each v, an array bad_with_col[v][s] = bad_with[s][v]
    # (same object, symmetric). We'll iterate over set bits of committed.

    def setbits(mask):
        # yield indices of set bits
        while mask:
            lsb = mask & (-mask)
            yield lsb.bit_length() - 1
            mask ^= lsb

    # Greedy initial solution to seed best
    greedy = []
    greedy_mask = 0
    for v in range(N):
        # check v against all pairs in greedy
        ok = True
        for a in range(len(greedy)):
            for b in range(a + 1, len(greedy)):
                ga, gb = greedy[a], greedy[b]
                if v in (ga, gb):
                    continue
                if lcm_mat[ga][gb] == lcm_mat[ga][v] == lcm_mat[gb][v]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            greedy.append(v)
            greedy_mask |= (1 << v)
    best[0] = len(greedy)
    best_set[0] = greedy_mask

    # Branch and bound
    # We process vertices in descending order of "constraint" to prune early.
    # A simpler robust approach: recursive on index, with bitsets.

    # We'll use a different, cleaner recursion:
    #   state = (idx, committed_mask, blocked_mask)
    #   remaining candidates = vertices with index >= idx that are not in
    #   committed and not in blocked.
    #   When we decide to include vertex v (at its turn), we must ensure it's
    #   not blocked, then add it to committed and update blocked for future.
    #   When we skip v, just move on.
    # Upper bound: current committed count + number of remaining candidates.

    # To update blocked when adding v: for each s in committed, blocked |=
    # bad_with[s][v]. We need the set of committed vertices. We can iterate
    # over set bits of committed_mask.

    # For speed, precompute for each v a list of (s, bad_with[s][v]) for all s.
    # Actually just use bad_with[s][v] directly.

    import time
    start = time.time()
    TIMEOUT = 55  # seconds

    # Order: we want a good ordering. Use the greedy order reversed, or a
    # degree-based order. Simpler: process in natural order but with a strong
    # bound. Given N=128 and the structure, the optimum is large so the search
    # space is constrained.

    # We'll do a recursive DFS with memo-free pruning.

    def dfs(committed_mask, committed_count, blocked_mask, remaining_mask):
        # remaining_mask: candidates still available (index >= current, not committed, not blocked)
        # Upper bound
        if committed_count + popcount(remaining_mask) <= best[0]:
            return
        # If time is up, return
        if time.time() - start > TIMEOUT:
            return
        if remaining_mask == 0:
            if committed_count > best[0]:
                best[0] = committed_count
                best_set[0] = committed_mask
            return
        # Pick the next vertex to branch on: the lowest set bit in remaining_mask
        # (or a heuristic choice). We'll pick the lowest for determinism.
        lsb = remaining_mask & (-remaining_mask)
        v = lsb.bit_length() - 1
        rest = remaining_mask ^ lsb

        # Option 1: include v
        # Update blocked: for each s in committed, blocked |= bad_with[s][v]
        new_blocked = blocked_mask
        cm = committed_mask
        while cm:
            sbit = cm & (-cm)
            s = sbit.bit_length() - 1
            new_blocked |= bad_with[s][v]
            cm ^= sbit
        # The newly blocked vertices among rest
        new_rest = rest & ~new_blocked
        # Recurse with v included
        dfs(committed_mask | lsb, committed_count + 1, new_blocked, new_rest)

        # Option 2: exclude v
        dfs(committed_mask, committed_count, blocked_mask, rest)

    # Initial: no committed, no blocked, all vertices remaining
    all_mask = (1 << N) - 1
    dfs(0, 0, 0, all_mask)

    # Build result from best_set
    result = [vals[i] for i in range(N) if (best_set[0] >> i) & 1]
    if not result:
        # fallback to greedy
        result = [vals[i] for i in greedy]
    return result
# EVOLVE-BLOCK-END
