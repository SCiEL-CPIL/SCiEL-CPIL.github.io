import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = P['n']
    edges = P['edges']
    
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    all_edges = list(edge_set)
    
    # Greedy to get upper bound
    def greedy():
        uncovered = [set() for _ in range(n)]
        for u, v in edges:
            uncovered[u].add(v)
            uncovered[v].add(u)
            
        bicliques = []
        while any(uncovered[u] for u in range(n)):
            u_star = max(range(n), key=lambda x: len(uncovered[x]))
            best_L = [u_star]
            best_R = list(uncovered[u_star])
            max_covered = len(best_R)
            
            for v in list(uncovered[u_star]):
                common = uncovered[u_star].intersection(uncovered[v])
                common.discard(u_star)
                common.discard(v)
                if not common: continue
                covered = 2 * len(common)
                if covered > max_covered:
                    max_covered = covered
                    best_L = [u_star, v]
                    best_R = list(common)
                    
            bicliques.append([best_L, best_R])
            for u in best_L:
                for v in best_R:
                    if v in uncovered[u]: uncovered[u].discard(v)
                    if u in uncovered[v]: uncovered[v].discard(u)
                    
        return bicliques
        
    greedy_res = greedy()
    upper_bound = len(greedy_res)
    
    # Backtracking
    def solve(k, uncovered, current_bicliques, edge_idx):
        if time.time() > deadline:
            return None
        if not uncovered:
            return current_bicliques
        if k == 0:
            return None
            
        # Pick an edge
        edge = next(iter(uncovered))
        u, v = edge
        
        # Try to assign to biclique i
        # Symmetry breaking: i can be 0 to min(k, edge_idx)
        # Actually, we can just try all, but prune if i > 0 and biclique i is empty and biclique i-1 is empty?
        # Better: try i from 0 to k-1.
        
        for i in range(k):
            L, R = current_bicliques[i]
            new_L = set(L)
            new_R = set(R)
            valid = True
            
            if u not in new_L and v not in new_R:
                for w in new_R:
                    if not adj[u][w]:
                        valid = False; break
                if valid:
                    for w in new_L:
                        if not adj[v][w]:
                            valid = False; break
                if valid:
                    new_L.add(u); new_R.add(v)
            elif u in new_L and v not in new_R:
                for w in new_L:
                    if not adj[v][w]:
                        valid = False; break
                if valid:
                    new_R.add(v)
            elif u not in new_L and v in new_R:
                for w in new_R:
                    if not adj[u][w]:
                        valid = False; break
                if valid:
                    new_L.add(u)
            else:
                valid = False
                
            if not valid:
                continue
                
            new_bicliques = current_bicliques.copy()
            new_bicliques[i] = (new_L, new_R)
            
            new_uncovered = set(uncovered)
            for lu in new_L:
                for rv in new_R:
                    e1 = (lu, rv) if lu < rv else (rv, lu)
                    if e1 in new_uncovered:
                        new_uncovered.remove(e1)
                        
            res = solve(k - 1, new_uncovered, new_bicliques, edge_idx + 1)
            if res is not None:
                return res
                
        return None

    for k in range(1, upper_bound + 1):
        initial_bicliques = [(set(), set()) for _ in range(k)]
        res = solve(k, set(all_edges), initial_bicliques, 0)
        if res is not None:
            result = []
            for L, R in res:
                if L and R:
                    result.append([list(L), list(R)])
            return result
            
    return greedy_res
