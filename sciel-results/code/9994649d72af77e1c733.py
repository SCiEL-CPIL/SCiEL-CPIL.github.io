import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55.0

    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_fast(adj):
        """Check if no 3-edge path s-u-v-t has deg(s) == deg(t).
        For each edge (u,v), check: does there exist a degree d such that
        N(u)\\{v} contains a vertex of degree d AND N(v)\\{u} contains a vertex of degree d (possibly different)?
        If s == t is the only pair, that's fine (s != t required).
        """
        degrees = compute_degrees(adj)
        # For each vertex, build set of neighbors
        # For each edge (u,v), we need:
        #   exists d: |{s in N(u)\{v} : deg(s)==d}| >= 1 AND |{t in N(v)\{u} : deg(t)==d, t != s}| >= 1
        # Simplification: if there are at least 2 vertices of degree d total, and one is in N(u)\{v} and another in N(v)\{u}, invalid.
        # More precisely: for edge (u,v), let S = N(u)\{v}, T = N(v)\{u}.
        # Invalid if exists s in S, t in T, s != t, deg(s) == deg(t).
        # This is equivalent to: exists degree d such that S has a vertex of degree d AND T has a vertex of degree d,
        # and it's not the case that the only vertex of degree d in S∪T is s=t (but s is in S, t in T, and s≠t by construction since we check s!=t).
        # Actually if s is in S and t is in T and s==t, that would mean s is neighbor of both u and v, and s≠u,v. Then s is in both S and T.
        # We need s != t. So if S and T share a vertex of degree d, that's fine as long as there's no OTHER pair.
        # Let's just do: for each edge (u,v), collect degree counts in S and T.
        # For each degree d, if count_S[d] >= 1 and count_T[d] >= 1:
        #   if count_S[d] + count_T[d] >= 2, we need to check if the only pair is s=t.
        #   Actually, if there's a vertex x of degree d in both S and T (i.e., x in S∩T), then we can pick s=x, t=x but s!=t fails.
        #   So we need: either count_S[d] >= 1 and count_T[d] >= 1 and (count_S[d] + count_T[d] >= 2) but excluding the case where the only overlap is a single shared vertex and no other.
        # Simpler: just iterate. For n=15 this is fast enough.
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                # Edge (u,v) exists
                s_mask = au & ~(1 << v)
                t_mask = adj[v] & ~(1 << u)
                # Check all pairs s in s_mask, t in t_mask, s != t
                sm = s_mask
                while sm:
                    s_bit = sm & -sm
                    s = s_bit.bit_length() - 1
                    sm ^= s_bit
                    ds = degrees[s]
                    tm = t_mask & ~(1 << s)
                    while tm:
                        t_bit = tm & -tm
                        t = t_bit.bit_length() - 1
                        tm ^= t_bit
                        if degrees[t] == ds:
                            return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Build K_{7,8}: A = {0..6}, B = {7..14}
    adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    best_adj = adj[:]
    best_count = count_edges(best_adj)

    # Local search: try to add edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if not (best_adj[u] & (1 << v)):
                    new_adj = best_adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    if is_valid_fast(new_adj):
                        best_adj = new_adj
                        best_count += 1
                        improved = True

    # Try 2-opt: remove one edge, add another
    for _ in range(50):  # limited iterations
        if time.time() > deadline:
            break
        improved = False
        # Collect existing edges
        edges = []
        for u in range(n):
            au = best_adj[u]
            for v in range(u + 1, n):
                if au & (1 << v):
                    edges.append((u, v))
        # Collect missing edges
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (best_adj[u] & (1 << v)):
                    missing.append((u, v))
        for e1 in edges:
            if time.time() > deadline:
                break
            for e2 in missing:
                if time.time() > deadline:
                    break
                new_adj = best_adj[:]
                # Remove e1
                u1, v1 = e1
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                # Add e2
                u2, v2 = e2
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                if is_valid_fast(new_adj):
                    best_adj = new_adj
                    best_count = count_edges(best_adj)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Build final edge list
    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([int(i), int(j)])

    return final_edges
