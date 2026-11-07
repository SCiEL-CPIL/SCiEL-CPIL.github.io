"""Find worst interval start for distinct-multiple matching with n=12."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = 12
    # LCM of 1..12 = 27720
    LCM = 27720

    def can_match(m, L):
        """Check if (m, m+L] contains distinct b_i with i | b_i for i=1..12."""
        # Build candidate lists for each i
        adj = []
        for i in range(1, n + 1):
            first = (m // i + 1) * i
            last = ((m + L) // i) * i
            if first > last:
                return False
            adj.append(list(range(first, last + 1, i)))

        # DFS-based bipartite matching (Kuhn's algorithm)
        match_to_val = {}  # val -> i

        def dfs(i, visited):
            for v in adj[i]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_to_val or dfs(match_to_val[v], visited):
                    match_to_val[v] = i
                    return True
            return False

        for i in range(n):
            visited = set()
            if not dfs(i, visited):
                return False
        return True

    def compute_f(m):
        """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
        for L in range(1, 200):
            if can_match(m, L):
                return L
        return 200

    # f(m) depends only on m mod LCM
    # Precompute f for all residues 0..LCM-1
    best_m = 0
    best_f = 0

    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r

    # best_m is in 0..27719, which is within [0, 1000000]
    return int(best_m)

construct()
