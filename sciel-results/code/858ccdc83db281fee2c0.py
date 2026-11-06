import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    adj = [[True]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = False
    
    # Precompute unique bipartite splits of 8 vertices into two 4-sets
    four_subsets = list(combinations(range(8), 4))
    seen = set()
    unique_splits = []
    for fs in four_subsets:
        comp = tuple(x for x in range(8) if x not in fs)
        key = (fs, comp) if fs < comp else (comp, fs)
        if key not in seen:
            seen.add(key)
            unique_splits.append((fs, comp))
    
    def has_q3():
        for subset in combinations(range(n), 8):
            s = list(subset)
            for a_rel, b_rel in unique_splits:
                a = [s[i] for i in a_rel]
                b = [s[i] for i in b_rel]
                missing = []
                valid = True
                for av in a:
                    not_conn = [bv for bv in b if not adj[av][bv]]
                    if len(not_conn) != 1:
                        valid = False
                        break
                    missing.append((av, not_conn[0]))
                if not valid:
                    continue
                b_to_a = {}
                for av, bv in missing:
                    if bv in b_to_a:
                        valid = False
                        break
                    b_to_a[bv] = av
                if not valid or len(b_to_a) != 4:
                    continue
                for bv in b:
                    deg = sum(1 for av in a if adj[av][bv])
                    if deg != 3:
                        valid = False
                        break
                if valid:
                    for av in a:
                        for bv in b:
                            if adj[av][bv]:
                                return (av, bv)
        return None
    
    # Local search: remove edges
    while time.time() < deadline - 15:
        edge = has_q3()
        if edge is None:
            break
        i, j = edge
        adj[i][j] = adj[j][i] = False
    
    # Try to add back
    if time.time() < deadline - 5:
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    adj[i][j] = adj[j][i] = True
                    if has_q3() is not None:
                        adj[i][j] = adj[j][i] = False
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges
