"""Program for R_5(C4) maximization: greedy randomized construction with
dynamic C4-risk tracking, plus local-search polish.  Targets n=26 (Lazebnik-Woldar)
and attempts n=27/28 if time permits."""
import time
import random
from collections import defaultdict


def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # ---------- Helpers ----------

    def edge_idx(i, j, n):
        """Lexicographic index of edge (i,j) with i<j in a list of length n*(n-1)//2."""
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Within row i, edge (i,j) is at position (j-i-1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Reverse: given linear index idx (0-based), return (i,j) with i<j."""
        # Solve for i: idx = i*(n-1) - i*(i-1)//2 + (j-i-1)
        # Row i has (n-1-i) edges, starting at offset sum_{k=0}^{i-1}(n-1-k)
        # = i*(n-1) - i*(i-1)//2
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def build_neighbor_sets(n, colors):
        """For each colour, build dict vertex -> set of neighbours in that colour."""
        cn = [defaultdict(set) for _ in range(5)]
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            cn[c][i].add(j)
            cn[c][j].add(i)
        return cn

    def count_c4(n, colors):
        """Count total 'excess' common-neighbor pairs across all colours. 0 means valid."""
        cn = build_neighbor_sets(n, colors)
        total = 0
        for c in range(5):
            for u in range(n):
                su = cn[c].get(u, set())
                if not su:
                    continue
                for v in range(u + 1, n):
                    sv = cn[c].get(v, set())
                    if not sv:
                        continue
                    common = su & sv
                    if len(common) >= 2:
                        total += len(common) - 1
        return total

    def is_valid(n, colors):
        return count_c4(n, colors) == 0

    # ---------- Greedy construction with C4-risk tracking ----------

    def greedy_construct(target_n, seed=0, time_limit=30.0):
        """
        Greedily assign colours to edges in a smart order, tracking for each colour
        the number of common neighbours between every pair of vertices.  When assigning
        a colour to edge (u,v), reject colours that would create a C4 (i.e., would make
        some pair (u,v) have >=2 common neighbours in that colour).  Among safe colours,
        prefer the one that minimises the increase in "at-risk" pairs (pairs going from
        1 to 2 common neighbours... actually we just avoid creating C4s).

        Returns (colors_list_or_None, valid_flag).
        """
        random.seed(seed)
        t_start = time.time()
        n = target_n
        m = n * (n - 1) // 2
        colors = [-1] * m

        # common[c][u][v] = number of common neighbours of u and v in colour c
        # Use a 2D array per colour for speed
        # common[c] is a list of lists: common[c][u][v]
        common = [[[0] * n for _ in range(n)] for _ in range(5)]

        # adj[c][u] = set of neighbours of u in colour c
        adj = [defaultdict(set) for _ in range(5)]

        # Edge order: random permutation, but we can bias towards edges that are
        # "harder" (higher degree endpoints) first.  For simplicity, random order.
        edge_order = list(range(m))
        random.shuffle(edge_order)

        # For each edge in order, assign a colour
        for eidx in edge_order:
            if time.time() > t_start + time_limit:
                return None, False

            i, j = idx_to_edge(eidx, n)

            # Try each colour; find safe ones
            safe_colors = []
            for c in range(5):
                # Check: does adding edge (i,j) in colour c create a C4?
                # A C4 is created if there exists a vertex w such that
                # w is adjacent to both i and j in colour c, AND there's another
                # vertex w' also adjacent to both i and j in colour c.
                # Actually, adding edge (i,j) to colour c creates a C4 if
                # the number of common neighbours of i and j in colour c is already >= 1.
                # Because then (i, j, w, j, i) ... no.
                #
                # Let me think again. A C4 in colour c is a set of 4 vertices {a,b,c,d}
                # with edges ab, bc, cd, da all in colour c.  Equivalently, two vertices
                # have >=2 common neighbours.
                #
                # When we add edge (i,j) in colour c, we need to check if this creates
                # a situation where some pair has >=2 common neighbours.
                #
                # Before adding (i,j), suppose i and j have k common neighbours in colour c.
                # After adding (i,j), i and j still have k common neighbours (the edge (i,j)
                # doesn't change common neighbours of i and j with respect to other vertices).
                #
                # But adding (i,j) changes the common neighbours of pairs involving i or j
                # with other vertices.  Specifically, for any vertex w:
                #   - i and w now have one more common neighbour (j) if w is adjacent to j in colour c
                #   - j and w now have one more common neighbour (i) if w is adjacent to i in colour c
                #
                # So we need to check: for each w != i,j, does adding (i,j) make the pair
                # (i,w) or (j,w) have >=2 common neighbours?
                #
                # Before: common[c][i][w] = |N_c(i) ∩ N_c(w)|
                # After adding (i,j): if j ∈ N_c(w), then common[c][i][w] increases by 1
                #   (because j is now in N_c(i) and j was already in N_c(w))
                # Similarly for (j,w).
                #
                # So colour c is safe iff:
                #   For all w != i,j:
                #     if j ∈ N_c(w) and common[c][i][w] >= 1, then unsafe (would become 2)
                #     if i ∈ N_c(w) and common[c][j][w] >= 1, then unsafe
                #
                # Wait, that's not quite right either.  Let me re-derive.
                #
                # N_c(i) after adding (i,j) = N_c(i) ∪ {j}
                # N_c(w) unchanged.
                # common_new(c, i, w) = |N_c(i) ∪ {j} ∩ N_c(w)| = |N_c(i) ∩ N_c(w)| + [j ∈ N_c(w)]
                # So common_new = common_old + 1 if j ∈ N_c(w), else common_old.
                #
                # We need common_new < 2 for all pairs, i.e., common_new <= 1.
                # So if j ∈ N_c(w) and common_old(c, i, w) >= 1, then common_new >= 2 -> C4.
                #
                # Similarly for pair (j, w): common_new(c, j, w) = common_old(c, j, w) + [i ∈ N_c(w)]
                #
                # Also need to check pair (i, j) itself:
                # common_new(c, i, j) = |N_c(i) ∪ {j} ∩ N_c(j) ∪ {i}|
                # But N_c(i) doesn't contain j before, and N_c(j) doesn't contain i before.
                # common_new(c, i, j) = |N_c(i) ∩ N_c(j)| + [j ∈ N_c(j)] + [i ∈ N_c(i)] + [j ∈ N_c(j) and i ∈ N_c(i)]
                # j is not in N_c(j) (no self-loops), i not in N_c(i).
                # So common_new(c, i, j) = common_old(c, i, j).
                # No change for the pair (i,j) itself.
                #
                # So the check is:
                #   safe = True
                #   for w in range(n):
                #       if w == i or w == j: continue
                #       if j in adj[c][w] and common[c][i][w] >= 1:
                #           safe = False; break
                #       if i in adj[c][w] and common[c][j][w] >= 1:
                #           safe = False; break

                safe = True
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if j in adj[c][w] and common[c][i][w] >= 1:
                        safe = False
                        break
                    if i in adj[c][w] and common[c][j][w] >= 1:
                        safe = False
                        break
                if safe:
                    safe_colors.append(c)

            if not safe_colors:
                # Dead end
                return None, False

            # Choose a safe colour; prefer one that minimises the increase in "risk"
            # Risk metric: for each safe colour c, compute how many pairs (i,w) or (j,w)
            # would go from 1 to 2 common neighbours (i.e., become at-risk but not yet C4).
            # Actually if it would become 2, that's a C4, which we already excluded.
            # So among safe colours, all keep all pairs at <=1.  Just pick randomly.
            c = random.choice(safe_colors)
            colors[eidx] = c

            # Update data structures
            adj[c][i].add(j)
            adj[c][j].add(i)

            # Update common counts for pairs involving i or j
            for w in range(n):
                if w == i or w == j:
                    continue
                if j in adj[c][w]:
                    common[c][i][w] += 1
                    common[c][w][i] += 1
                if i in adj[c][w]:
                    common[c][j][w] += 1
                    common[c][w][j] += 1
            # Pair (i,j): common count unchanged as discussed
            # But we should set common[c][i][j] = common[c][j][i] = |N_c(i) ∩ N_c(j)|
            # which is already tracked.  Actually we never updated it.  Let's compute:
            # common[c][i][j] should be the number of vertices w adjacent to both i and j in colour c.
            # We can track this incrementally: when we add edge (i,j), for any w adjacent to
            # both i and j in colour c, common[c][i][j] increases.  But we don't easily know
            # which w are adjacent to both.  However, we only need common values for the
            # safety check, which uses common[c][i][w] for w != i,j.  The value common[c][i][j]
            # is only needed if we later check a pair involving i and j, which we do via
            # the same logic.  Actually in the safety check above, we check common[c][i][w]
            # for w != i,j, so we never need common[c][i][j] directly in the check.
            # But for correctness of future checks, we need common[c][i][j] to be accurate
            # if some future edge (i, x) is being considered and we check pair (j, x) or similar.
            #
            # Let me reconsider: the safety check for adding edge (u,v) in colour c checks:
            #   for w != u,v: if v in adj[c][w] and common[c][u][w] >= 1 -> unsafe
            # This uses common[c][u][w] where w is a third vertex.  So we need common[c][u][w]
            # to be accurate for all pairs (u,w) where w is not the edge being added.
            #
            # When we add edge (i,j) in colour c, we update common[c][i][w] for all w != i,j
            # if j ∈ adj[c][w], and common[c][j][w] for all w != i,j if i ∈ adj[c][w].
            # We do NOT update common[c][i][j].  But common[c][i][j] = |N_c(i) ∩ N_c(j)|.
            # This value changes when we add edges incident to i or j.  Specifically, when we
            # add edge (i, x) in colour c, common[c][i][j] increases by 1 if x ∈ N_c(j).
            # So we need to update common[c][i][j] when adding edges incident to i or j.
            #
            # In the current update, when adding (i,j), we should also update common[c][i][j]:
            # common[c][i][j] = |N_c(i) ∩ N_c(j)|.  Before adding (i,j), this is some value.
            # After adding (i,j), N_c(i) gains j and N_c(j) gains i, but j ∉ N_c(j) and i ∉ N_c(i),
            # so the intersection doesn't change.  So common[c][i][j] is unchanged.  Good.
            #
            # But when a FUTURE edge (i, x) is added in colour c, we need to update common[c][i][j]
            # if x ∈ N_c(j).  In the future update, when adding (i,x), we will iterate over w
            # and if x ∈ adj[c][w], we update common[c][i][w].  If w = j, and x ∈ adj[c][j],
            # then we update common[c][i][j] += 1.  Yes!  Because in the update for adding (i,x):
            #   for w in range(n):
            #       if w == i or w == x: continue
            #       if x in adj[c][w]: common[c][i][w] += 1
            #       if i in adj[c][w]: common[c][x][w] += 1
            # So if w = j and x ∈ adj[c][j], common[c][i][j] is incremented.  Correct.

        # All edges assigned
        return colors, True

    # ---------- Local search ----------

    def local_search(n, colors, max_iters=50000, time_limit=30.0):
        """Local search: minimise C4 count. Returns (colors, valid_flag)."""
        m = n * (n - 1) // 2
        current = count_c4(n, colors)
        if current == 0:
            return colors, True

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        t_start = time.time()
        for it in range(max_iters):
            if time.time() > t_start + time_limit:
                break
            if current == 0:
                return colors, True

            eidx = random.randint(0, m - 1)
            old_c = colors[eidx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue

            colors[eidx] = new_c
            new_val = count_c4(n, colors)
            if new_val > current:
                colors[eidx] = old_c
            else:
                current = new_val

        return colors, (current == 0)

    # ---------- Main strategy ----------

    # Try greedy construction for n=28, 27, 26 with multiple seeds
    for target_n in [28, 27, 26]:
        if time.time() >= deadline - 5:
            break

        remaining = deadline - time.time() - 3
        if remaining < 3:
            break

        # Budget per seed
        num_seeds = min(5, max(1, int(remaining / 8)))
        time_per_seed = min(10.0, remaining / (num_seeds + 1))

        for seed in range(num_seeds):
            if time.time() > deadline - 3:
                break
            colors_g, valid_g = greedy_construct(target_n, seed=seed * 1000 + 42, time_limit=time_per_seed)
            if valid_g:
                return {'n': target_n, 'colors': colors_g}
            if colors_g is not None:
                # Use as starting point for local search
                if time.time() < deadline - 10:
                    colors_ls, valid_ls = local_search(target_n, colors_g, max_iters=30000,
                                                       time_limit=min(10.0, deadline - time.time() - 5))
                    if valid_ls:
                        return {'n': target_n, 'colors': colors_ls}

    # Fallback: local search from structured init for n=26
    if time.time() < deadline - 5:
        def init_colors_structured(n):
            m = n * (n - 1) // 2
            cols = [0] * m
            for idx in range(m):
                i, j = idx_to_edge(idx, n)
                d = (j - i) % n
                cols[idx] = d % 5
            return cols

        for restart in range(3):
            if time.time() > deadline - 3:
                break
            random.seed(777 + restart)
            colors26 = init_colors_structured(26)
            for _ in range(50):
                eidx = random.randint(0, len(colors26) - 1)
                colors26[eidx] = random.randint(0, 4)
            colors26, valid26 = local_search(26, colors26, max_iters=30000,
                                             time_limit=min(15.0, deadline - time.time() - 3))
            if valid26:
                return {'n': 26, 'colors': colors26}

    # Fallback: n=25
    if time.time() < deadline - 3:
        colors25 = init_colors_structured(25)
        colors25, valid25 = local_search(25, colors25, max_iters=20000,
                                         time_limit=min(10.0, deadline - time.time() - 2))
        if valid25:
            return {'n': 25, 'colors': colors25}

    # Final fallback: n=20
    n20 = 20
    colors20 = init_colors_structured(n20)
    if not is_valid(n20, colors20):
        colors20, _ = local_search(n20, colors20, max_iters=10000, time_limit=5.0)
    return {'n': n20, 'colors': colors20}
