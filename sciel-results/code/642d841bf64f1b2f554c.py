import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 320  # safety margin before 330s kill

    def elapsed():
        return time.time() - start_time

    # ---------- Paley(37) seed ----------
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i, j] = True
                adj[j, i] = True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    # ---------- Bitset helpers ----------
    # Represent a vertex set as a Python int bitmask.
    # adj_rows[i] is a bitmask of neighbors of vertex i.

    def build_bitmasks(adj, n):
        masks = []
        for i in range(n):
            m = 0
            row = adj[i]
            # set bits where row is True
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks.append(m)
        return masks

    # ---------- K4 / bar-K4 enumeration via bitmasks ----------
    def find_k4_bitmasks(masks, n):
        """Return list of int bitmasks, each a K4."""
        k4s = []
        for a in range(n):
            Na = masks[a]
            # iterate b in Na with b > a
            b = (Na >> (a + 1)) & ((1 << (n - a - 1)) - 1)
            # iterate over set bits of b
            while b:
                low = b & (-b)
                bidx = low.bit_length() - 1
                b = b ^ low
                if bidx <= a:
                    continue
                Nboth = Na & masks[bidx]
                # c in Nboth with c > bidx
                c_mask = Nboth >> (bidx + 1)
                while c_mask:
                    lowc = c_mask & (-c_mask)
                    cidx = lowc.bit_length() - 1
                    c_mask = c_mask ^ lowc
                    Ntriple = Nboth & masks[cidx]
                    # d in Ntriple with d > cidx
                    d_mask = Ntriple >> (cidx + 1)
                    while d_mask:
                        lowd = d_mask & (-d_mask)
                        didx = lowd.bit_length() - 1
                        d_mask = d_mask ^ lowd
                        k4s.append((1 << a) | (1 << bidx) | (1 << cidx) | (1 << didx))
        return k4s

    def find_bar_k4_bitmasks(masks, n):
        """Return list of int bitmasks, each an independent set of size 4."""
        # independent set = K4 in complement
        comp_masks = []
        full = (1 << n) - 1
        for i in range(n):
            comp_masks.append(full ^ masks[i])
        return find_k4_bitmasks(comp_masks, n)

    # ---------- Slack computation with bitmasks ----------
    def compute_slack(S_mask, k4_masks, bar_k4_masks):
        """
        S_mask: bitmask of S.
        k4_masks: list of K4 bitmasks.
        bar_k4_masks: list of bar-K4 bitmasks.
        Returns min slack (0 if invalid).
        """
        min_k4_slack = 4
        for k4 in k4_masks:
            count_in_S = (S_mask & k4).bit_count()
            slack = 4 - count_in_S
            if slack < min_k4_slack:
                min_k4_slack = slack
            if slack == 0:
                return 0
        min_bk4_slack = 4
        for bk4 in bar_k4_masks:
            count_in_S = (S_mask & bk4).bit_count()
            slack = count_in_S
            if slack < min_bk4_slack:
                min_bk4_slack = slack
            if slack == 0:
                return 0
        return min(min_k4_slack, min_bk4_slack)

    # ---------- Local search for valid S ----------
    def search_valid_S(masks, n, k4_masks, bar_k4_masks, deadline, max_iters=3000):
        """
        Find a bitmask S such that compute_slack(S, ...) > 0.
        Uses local search with random restarts.
        Returns best S_mask found (or None).
        """
        best_S = None
        best_slack = -1

        # Degree-based initial guess: S = vertices with moderate degree
        # (we want S to have no K4 and V\S to have no bar-K4)
        # Try a few initializations.
        inits = []

        # Init 1: random 50%
        S1 = 0
        for v in range(n):
            if random.random() < 0.5:
                S1 |= (1 << v)
        inits.append(S1)

        # Init 2: low-degree vertices (more likely to break K4s? actually K4 constraint is on S, so S should avoid K4s)
        # Let's try S = vertices with degree <= median
        degrees = [masks[i].bit_count() for i in range(n)]
        med = sorted(degrees)[n // 2]
        S2 = 0
        for v in range(n):
            if degrees[v] <= med:
                S2 |= (1 << v)
        inits.append(S2)

        # Init 3: high-degree vertices
        S3 = 0
        for v in range(n):
            if degrees[v] > med:
                S3 |= (1 << v)
        inits.append(S3)

        for init_S in inits:
            if elapsed() > deadline - 5:
                break
            S = init_S
            cur_slack = compute_slack(S, k4_masks, bar_k4_masks)
            if cur_slack > best_slack:
                best_slack = cur_slack
                best_S = S

            # Local search: flip single bits
            for _ in range(max_iters):
                if elapsed() > deadline - 5:
                    break
                v = random.randint(0, n - 1)
                bit = (1 << v)
                S2 = S ^ bit
                new_slack = compute_slack(S2, k4_masks, bar_k4_masks)
                if new_slack > cur_slack:
                    S = S2
                    cur_slack = new_slack
                    if cur_slack > best_slack:
                        best_slack = cur_slack
                        best_S = S
                        if best_slack >= 2:
                            return best_S
                # Occasionally random restart
                if random.random() < 0.05:
                    S = 0
                    for vv in range(n):
                        if random.random() < 0.5:
                            S |= (1 << vv)
                    cur_slack = compute_slack(S, k4_masks, bar_k4_masks)
                    if cur_slack > best_slack:
                        best_slack = cur_slack
                        best_S = S

        return best_S

    # ---------- Main construction loop with restarts ----------
    best_n = n
    best_adj = adj.copy()

    # Try multiple construction attempts with different random seeds
    for attempt in range(20):
        if elapsed() > deadline - 30:
            break

        # Reset to seed
        cur_n = q
        cur_adj = np.zeros((cur_n, cur_n), dtype=bool)
        for i in range(cur_n):
            for j in range(i + 1, cur_n):
                if (j - i) % q in residues:
                    cur_adj[i, j] = True
                    cur_adj[j, i] = True

        # Build masks
        masks = build_bitmasks(cur_adj, cur_n)
        k4_masks = find_k4_bitmasks(masks, cur_n)
        bar_k4_masks = find_bar_k4_bitmasks(masks, cur_n)

        # Try to add vertices
        stuck_count = 0
        while elapsed() < deadline - 10:
            # Recompute K4s and bar-K4s (they change as we add vertices)
            masks = build_bitmasks(cur_adj, cur_n)
            k4_masks = find_k4_bitmasks(masks, cur_n)
            bar_k4_masks = find_bar_k4_bitmasks(masks, cur_n)

            # Search for valid S
            S_mask = search_valid_S(masks, cur_n, k4_masks, bar_k4_masks,
                                    deadline - 15, max_iters=2000)

            if S_mask is not None:
                # Add new vertex with neighborhood S_mask
                new_n = cur_n + 1
                cur_adj = np.resize(cur_adj, (new_n, new_n))
                cur_adj[cur_n, :] = False
                cur_adj[:, cur_n] = False
                # Set neighbors
                for v in range(cur_n):
                    if (S_mask >> v) & 1:
                        cur_adj[cur_n, v] = True
                        cur_adj[v, cur_n] = True
                cur_n = new_n
                stuck_count = 0
                if cur_n > best_n:
                    best_n = cur_n
                    best_adj = cur_adj.copy()
            else:
                stuck_count += 1
                if stuck_count >= 3:
                    break

    # Final verification: check no K5 or independent 5
    final_adj = best_adj
    final_n = best_n
    valid = True
    # Check for K5
    for combo in combinations(range(final_n), 5):
        a, b, c, d, e = combo
        if (final_adj[a, b] and final_adj[a, c] and final_adj[a, d] and final_adj[a, e] and
            final_adj[b, c] and final_adj[b, d] and final_adj[b, e] and
            final_adj[c, d] and final_adj[c, e] and
            final_adj[d, e]):
            valid = False
            break
    if valid:
        # Check for independent 5
        for combo in combinations(range(final_n), 5):
            a, b, c, d, e = combo
            if (not final_adj[a, b] and not final_adj[a, c] and not final_adj[a, d] and not final_adj[a, e] and
                not final_adj[b, c] and not final_adj[b, d] and not final_adj[b, e] and
                not final_adj[c, d] and not final_adj[c, e] and
                not final_adj[d, e]):
                valid = False
                break

    if not valid:
        # Fall back to seed
        final_n = q
        final_adj = np.zeros((final_n, final_n), dtype=bool)
        for i in range(final_n):
            for j in range(i + 1, final_n):
                if (j - i) % q in residues:
                    final_adj[i, j] = True
                    final_adj[j, i] = True

    edges = get_edges(final_adj, final_n)
    return final_n, edges


if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
