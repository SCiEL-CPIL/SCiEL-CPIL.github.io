import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def time_up():
        return time.time() - t0 > DEADLINE - 3.0
    
    # Quick fallback
    u, v = edges[0]
    best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    best_size = 2
    
    # Find maximum clique using optimized Bron-Kerbosch with pivot
    def max_clique_size():
        """Find the maximum clique size using Bron-Kerbosch."""
        best_clique = []
        
        def bk(R, P_set, X):
            nonlocal best_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Pivot selection
            candidates = P_set | X
            if candidates:
                pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
                to_try = P_set - adj[pivot]
            else:
                to_try = P_set
            for v in list(to_try):
                bk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return best_clique
    
    # Find all maximal cliques above a threshold
    def all_maximal_cliques(min_size):
        cliques = []
        
        def bk(R, P_set, X):
            nonlocal cliques
            if time_up():
                return
            if not P_set and not X:
                if len(R) >= min_size:
                    cliques.append(list(R))
                return
            candidates = P_set | X
            if candidates:
                pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
                to_try = P_set - adj[pivot]
            else:
                to_try = P_set
            for v in list(to_try):
                bk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    # Path finding: try to find internally disjoint paths for all pairs in branches
    def find_disjoint_paths(branches, max_attempts=20):
        """
        Try to find pairwise internally disjoint paths for all pairs in branches.
        Returns list of paths or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        if not pairs:
            return []
        
        # If all pairs are directly connected, paths are trivial
        all_direct = all(v in adj[u] for u, v in pairs)
        if all_direct:
            return [[u, v] for u, v in pairs]
        
        # Count direct edges
        direct_pairs = [(u, v) for u, v in pairs if v in adj[u]]
        non_direct_pairs = [(u, v) for u, v in pairs if v not in adj[u]]
        
        # Start with direct pairs (no internal vertices used)
        paths = {}
        used_internal = set()
        for u, v in direct_pairs:
            paths[(u, v)] = [u, v]
        
        # For non-direct pairs, find paths avoiding used_internal and branches
        remaining = list(non_direct_pairs)
        
        # Order by constraint: pairs with fewer available paths first
        def pair_score(p):
            u, v = p
            # Count common neighbors not in branches or used
            common = (adj[u] & adj[v]) - bs - used_internal
            return len(common)
        
        remaining.sort(key=pair_score)
        
        # Backtracking search
        def bfs_path(s, t, blocked):
            """Find shortest path from s to t avoiding blocked vertices."""
            if s == t:
                return [s]
            if t in adj[s] and t not in blocked:
                return [s, t]
            parent = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                for nb in sorted(adj[curr]):
                    if nb == t:
                        if t not in blocked:
                            parent[t] = curr
                            # Reconstruct
                            path = [t]
                            node = t
                            while parent[node] is not None:
                                node = parent[node]
                                path.append(node)
                            path.reverse()
                            return path
                        else:
                            continue
                    if nb not in blocked and nb not in parent:
                        parent[nb] = curr
                        q.append(nb)
            return None
        
        def backtrack(idx, used_int):
            if idx == len(remaining):
                return True
            u, v = remaining[idx]
            # Try multiple path lengths
            for path_len in range(2, 8):  # try paths of length 2 to 7
                # BFS for path of exactly path_len
                def bfs_exact_len(s, t, max_len, blocked):
                    """Find a path of length <= max_len from s to t."""
                    if max_len == 1:
                        return [s, t] if t in adj[s] else None
                    if max_len == 0:
                        return [s] if s == t else None
                    # BFS with depth limit
                    from collections import defaultdict
                    dist = {s: 0}
                    par = {s: None}
                    q = deque([s])
                    found = False
                    while q:
                        curr = q.popleft()
                        if dist[curr] >= max_len:
                            continue
                        if curr == t:
                            found = True
                            break
                        for nb in adj[curr]:
                            if nb not in dist:
                                if nb == t or nb not in blocked:
                                    dist[nb] = dist[curr] + 1
                                    par[nb] = curr
                                    q.append(nb)
                    if not found or t not in par:
                        return None
                    path = []
                    node = t
                    while node is not None:
                        path.append(node)
                        node = par[node]
                    path.reverse()
                    return path
                
                p = bfs_exact_len(u, v, path_len, used_int | bs)
                if p is not None:
                    internal = set(p[1:-1])
                    if internal & used_int or internal & bs:
                        continue
                    # Save and recurse
                    paths[(u, v)] = p
                    for vn in internal:
                        used_int.add(vn)
                    if backtrack(idx + 1, used_int):
                        return True
                    for vn in internal:
                        used_int.discard(vn)
                    del paths[(u, v)]
            
            return False
        
        # Try with different orderings
        rng = random.Random(42)
        orderings = [remaining]
        # Add shuffled versions
        for _ in range(max_attempts - 1):
            sh = list(remaining)
            rng.shuffle(sh)
            orderings.append(sh)
        
        for ordering in orderings:
            if time_up():
                break
            # Reset
            paths = {}
            for u, v in direct_pairs:
                paths[(u, v)] = [u, v]
            used_int = set()
            
            if backtrack(0, used_int):
                # Build result in standard pair order
                result = []
                for u, v in pairs:
                    if (u, v) in paths:
                        result.append(paths[(u, v)])
                    elif (v, u) in paths:
                        result.append(list(reversed(paths[(v, u)])))
                    else:
                        return None
                return result
        return None
    
    # Phase 1: Find maximum clique
    max_cli = max_clique_size()
    if len(max_cli) > best_size:
        cli_sorted = sorted(max_cli)
        paths = find_disjoint_paths(cli_sorted)
        if paths is not None:
            best = {'branches': [int(x) for x in cli_sorted], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(cli_sorted)
    
    # Phase 2: Try all maximal cliques of size >= best_size
    if not time_up():
        cliques = all_maximal_cliques(best_size)
        cliques.sort(key=len, reverse=True)
        for clique in cliques:
            if time_up():
                break
            if len(clique) < best_size:
                break
            if len(clique) == best_size:
                # Only try if different
                if sorted(clique) == sorted(best['branches']):
                    continue
            cli_sorted = sorted(clique)
            paths = find_disjoint_paths(cli_sorted)
            if paths is not None:
                best = {'branches': [int(x) for x in cli_sorted], 'paths': [[int(x) for x in p] for p in paths]}
                best_size = len(cli_sorted)
    
    # Phase 3: Try to extend beyond clique by adding non-clique vertices
    if not time_up() and best_size < n:
        current = set(best['branches'])
        
        # Score candidates by connectivity to current branches
        def score(v):
            return sum(1 for b in current if v in adj[b])
        
        candidates = sorted([v for v in range(n) if v not in current], key=score, reverse=True)
        
        improved = True
        while improved and not time_up():
            improved = False
            for v in candidates:
                if time_up():
                    break
                new_set = current | {v}
                paths = find_disjoint_paths(sorted(new_set))
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = len(new_set)
                    current = new_set
                    candidates = sorted([x for x in range(n) if x not in current], key=score, reverse=True)
                    improved = True
                    break
    
    # Phase 4: Random search for larger sets
    if not time_up():
        deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        for k in range(best_size + 1, min(best_size + 6, n + 1)):
            if time_up():
                break
            found = False
            for _ in range(30):
                if time_up():
                    break
                subset = random.sample(deg_sorted[:min(20, n)], k)
                paths = find_disjoint_paths(sorted(subset))
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = k
                    found = True
                    break
            if not found:
                break
    
    return best
