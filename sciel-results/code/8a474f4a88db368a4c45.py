import time
import random

def construct():
    start_time = time.time()
    DEATHLINE = 120

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, edges):
        adj_C = [set() for _ in range(k)]
        adj_I = [set() for _ in range(n - k)]
        
        for u, v in edges:
            if u < k and v < k:
                adj_C[u].add(v)
                adj_C[v].add(u)
            elif u < k:
                adj_I[v - k].add(u)
            else:
                adj_I[u - k].add(v)
        
        # Calculate weights for matching
        w = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                count = 0
                for u_idx in range(n - k):
                    if i in adj_I[u_idx] and j in adj_I[u_idx]:
                        count += 1
                w[i][j] = count
                w[j][i] = count

        # Greedy + Local Search Matching
        edges_list = []
        for i in range(k):
            for j in range(i + 1, k):
                edges_list.append((w[i][j], i, j))
        edges_list.sort(reverse=True)
        
        matched = [False] * k
        matching = []
        for weight, i, j in edges_list:
            if not matched[i] and not matched[j]:
                matching.append((i, j))
                matched[i] = True
                matched[j] = True
        
        improved = True
        while improved:
            improved = False
            for idx1 in range(len(matching)):
                for idx2 in range(idx1 + 1, len(matching)):
                    a, b = matching[idx1]
                    c, d = matching[idx2]
                    w_curr = w[a][b] + w[c][d]
                    if w[a][c] + w[b][d] > w_curr:
                        matching[idx1] = (a, c)
                        matching[idx2] = (b, d)
                        improved = True
                    if w[a][d] + w[b][c] > w_curr:
                        matching[idx1] = (a, d)
                        matching[idx2] = (b, c)
                        improved = True
        
        # Handle odd k
        if k % 2 != 0:
            free_v = -1
            for i in range(k):
                if not matched[i]:
                    free_v = i
                    break
            
            if free_v != -1 and matching:
                # Merge free_v into the first pair
                a, b = matching[0]
                matching[0] = (a, b, free_v)
        
        # Assign I
        assignment = [None] * (n - k)
        for u_idx in range(n - k):
            for p_idx, p in enumerate(matching):
                # Check if u is connected to all vertices in p
                valid = True
                for v in p:
                    if v not in adj_I[u_idx]:
                        valid = False
                        break
                if valid:
                    assignment[u_idx] = p_idx
                    break
        
        cliques = []
        pair_members = [set() for _ in range(len(matching))]
        for u_idx in range(n - k):
            if assignment[u_idx] is not None:
                pair_members[assignment[u_idx]].add(u_idx + k)
        
        for p_idx, p in enumerate(matching):
            clique = list(p) + list(pair_members[p_idx])
            if len(clique) >= 2:
                cliques.append(sorted(clique))
        
        # Cover remaining edges in C and I-C
        # Edges in C not covered by matching cliques
        covered_C_edges = set()
        for p in matching:
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    covered_C_edges.add((min(p[i], p[j]), max(p[i], p[j])))
        
        for u in range(k):
            for v in range(u+1, k):
                if (u,v) not in covered_C_edges:
                    cliques.append([u, v])
                    
        # Edges I-C not covered
        for u_idx in range(n - k):
            u = u_idx + k
            assigned_pair = assignment[u_idx]
            covered_in_C = set()
            if assigned_pair is not None:
                covered_in_C = set(matching[assigned_pair])
            
            for v in adj_I[u_idx]:
                if v not in covered_in_C:
                    cliques.append(sorted([u, v]))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        if time.time() - start_time > DEATHLINE - 5:
            break
            
        cliques = solve(n, k, edges)
        results.append(cliques)

    final = []
    for clique_list in results:
        final.append([list(int(v) for v in clique) for clique in clique_list])
    
    return final
