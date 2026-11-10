"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        # For each edge (u,v): deg-sets of N(u)\{v} and N(v)\{u} must be disjoint
        for (u, v) in edge_set:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                return False
        return True

    # Start with K_{7,8} (56 edges, proven valid)
    A = list(range(7))
    B = list(range(7, 15))
    current = set()
    for a in A:
        for b in B:
            current.add((a, b))
    
    best_edges = current
    best_count = len(current)  # 56

    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_possible if e not in current]
    
    # Fast validity check with incremental degree/adj tracking
    def try_add(edge_set, e):
        u, v = e
        # Check if adding (u,v) creates a violation
        # New edge (u,v): need deg-sets of N(u)\{v} and N(v)\{u} disjoint
        # But adding (u,v) changes degrees of u,v and their neighbors' degree-sets
        # For speed, just do full check (n=15, E~60, fast enough)
        trial = edge_set | {e}
        return is_valid(trial, n)
    
    # Greedy: try adding missing edges, prefer those within same part (A or B)
    # Sort missing by "safety": edges within A or B first (less likely to create conflicts)
    def edge_priority(e):
        u, v = e
        in_A = u < 7 and v < 7
        in_B = u >= 7 and v >= 7
        if in_A or in_B:
            return 0  # intra-part, try first
        return 1  # bipartite (already present, so these are all intra-part actually)
    
    missing.sort(key=edge_priority)
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for e in missing[:]:
            if time.time() > deadline:
                break
            if e in current:
                continue
            if try_add(current, e):
                current = current | {e}
                missing.remove(e)
                added_any = True
                if len(current) > best_count:
                    best_edges = current
                    best_count = len(current)
    
    # If we got stuck, try swap moves: remove one edge, add another
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            current_list = list(current)
            import random
            random.shuffle(current_list)
            improved = False
            for rem in current_list:
                if time.time() > deadline:
                    break
                trial = current - {rem}
                for add in all_possible:
                    if add in trial:
                        continue
                    if len(trial | {add}) <= len(current):
                        continue
                    if is_valid(trial | {add}, n):
                        current = trial | {add}
                        improved = True
                        if len(current) > best_count:
                            best_edges = current
                            best_count = len(current)
                        break
                if improved:
                    break
    
    if len(current) > best_count:
        best_edges = current
        best_count = len(current)

    # Ensure validity of best result
    if not is_valid(best_edges, n):
        # Fallback to K_{7,8}
        best_edges = set()
        for a in A:
            for b in B:
                best_edges.add((a, b))
    
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
