import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    def build_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot
            if not P and not X:
                return
            u = next(iter(P | X))
            candidates = [v for v in P if v not in adj[u]]
            
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bk(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
            
        clique_vertices = [set(c) for c in cliques]
        num_cliques = len(clique_vertices)
        
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_vertices):
            for v in c:
                vertex_cliques[v].append(i)
                
        uncovered = set(range(num_cliques))
        solution = set()
        avail_counts = [len(c) for c in clique_vertices]
        
        while uncovered:
            if time.time() > deadline:
                for idx in uncovered:
                    v = next(iter(clique_vertices[idx]))
                    solution.add(v)
                break
                
            best_v = -1
            best_score = -1.0
            
            for v in range(n):
                if v in solution:
                    continue
                
                score = 0.0
                for idx in vertex_cliques[v]:
                    if idx in uncovered:
                        if avail_counts[idx] > 0:
                            score += 1.0 / avail_counts[idx]
                        else:
                            score += 1e9 # Should not happen
                            
                if score > best_score:
                    best_score = score
                    best_v = v
                    
            if best_v == -1:
                break
                
            solution.add(best_v)
            
            # Update avail_counts and uncovered
            for idx in vertex_cliques[best_v]:
                if idx in uncovered:
                    avail_counts[idx] -= 1
                    # If avail_counts becomes 0, it means this clique is now "critical" 
                    # but it's still uncovered until we pick a vertex from it.
                    # Wait, if avail_counts becomes 0, it means ALL vertices in this clique 
                    # are in solution? No, avail_counts counts vertices NOT in solution.
                    # If it becomes 0, it means every vertex in the clique is in solution.
                    # But we just added best_v. If best_v was the last available vertex, 
                    # then the clique is covered.
                    # Actually, if best_v is in clique idx, and we add it to solution,
                    # the clique IS covered.
                    # So we should remove it from uncovered immediately.
                    uncovered.discard(idx)
                    
        return sorted(solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
