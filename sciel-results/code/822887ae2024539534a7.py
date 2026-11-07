import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        
        # Map each K-edge to I-vertices that can cover it
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)
        
        best_cliques = None
        best_count = float('inf')
        
        for attempt in range(80):
            if time.time() > deadline - 3:
                break
            
            rng = random.Random(seed + attempt * 7919)
            
            curr_assigned = {v: set() for v in range(k, n)}
            curr_used = {v: set() for v in range(k, n)}
            
            # Greedy: process edges by scarcity
            for e in sorted(k_edges, key=lambda e: len(edge_to_vs[e])):
                u, w = e
                valid = [v for v in edge_to_vs[e] 
                         if u not in curr_used[v] and w not in curr_used[v]]
                if not valid:
                    continue
                
                # Pick v with most remaining capacity
                best_v = max(valid, key=lambda v: len(neighbors[v])//2 - len(curr_assigned[v]))
                if len(valid) > 1 and rng.random() < 0.25:
                    best_v = rng.choice(valid)
                
                curr_assigned[best_v].add(e)
                curr_used[best_v].add(u)
                curr_used[best_v].add(w)
            
            # Local search: try to assign more edges
            assigned_set = set(e for v in range(k,n) for e in curr_assigned[v])
            for _ in range(15):
                unassigned = [e for e in k_edges if e not in assigned_set]
                if not unassigned:
                    break
                rng.shuffle(unassigned)
                progress = False
                for e in unassigned[:25]:
                    u, w = e
                    valid = [v for v in edge_to_vs[e]
                             if u not in curr_used[v] and w not in curr_used[v]]
                    if valid:
                        v = rng.choice(valid)
                        curr_assigned[v].add(e)
                        curr_used[v].add(u)
                        curr_used[v].add(w)
                        assigned_set.add(e)
                        progress = True
                        break
                if not progress:
                    break
            
            # Build cliques
            cliques = []
            for v in range(k, n):
                for e in curr_assigned[v]:
                    cliques.append([e[0], e[1], v])
                N_v = set(neighbors[v])
                for u in N_v - curr_used[v]:
                    cliques.append([u, v])
            
            # Cover remaining K edges with maximal cliques
            rem = [e for e in k_edges if e not in assigned_set]
            if rem:
                rem_set = set(rem)
                adj = {u: set() for u in range(k)}
                for e in rem_set:
                    adj[e[0]].add(e[1])
                    adj[e[1]].add(e[0])
                
                while rem_set:
                    # Pick edge with max common neighbors
                    be = max(rem_set, key=lambda e: len(adj[e[0]] & adj[e[1]]))
                    clique = {be[0], be[1]}
                    changed = True
                    while changed:
                        changed = False
                        common = set(range(k))
                        for x in clique:
                            common &= adj[x]
                        common -= clique
                        if common:
                            best_x = max(common, key=lambda x: len(adj[x]))
                            clique.add(best_x)
                            changed = True
                    
                    cliques.append(sorted(clique))
                    for a, b in combinations(clique, 2):
                        rem_set.discard((min(a,b), max(a,b)))
                    for x in clique:
                        for y in list(adj[x]):
                            if y in clique:
                                adj[x].discard(y)
                                adj[y].discard(x)
            
            clean = [[int(x) for x in c] for c in cliques]
            cnt = len(clean)
            if cnt < best_count:
                best_count = cnt
                best_cliques = clean
        
        if best_cliques is None:
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
        
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
