import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edge_in_clique(clique, adj):
        """Check all pairs in clique are edges."""
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def greedy_partition(n, edges, adj, rng, max_time):
        """Greedy: repeatedly find a large clique among remaining edges."""
        remaining = set(edges)
        if not remaining:
            return []
        
        # Track remaining edges per vertex for quick degree
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        while remaining:
            if time.time() > max_time:
                return None  # timeout
            
            # Pick starting vertex: prefer high degree in remaining graph
            # Randomize among top candidates
            candidates = [v for v in range(n) if rem_adj[v] != 0]
            if not candidates:
                break
            
            # Sort by degree, pick from top few randomly
            candidates.sort(key=lambda v: bin(rem_adj[v]).count('1'), reverse=True)
            top_k = min(5, len(candidates))
            start = rng.choice(candidates[:top_k])
            
            # Build clique greedily from start
            clique = [start]
            # Common neighbors of all in clique
            common = rem_adj[start]
            for v in clique[1:]:
                common &= rem_adj[v]
            
            # Add vertices from common to clique (greedy: pick those with most common neighbors)
            # Iterate: add vertex with highest degree within common
            added = True
            while added:
                added = False
                # Candidates: vertices in common not already in clique
                cands = []
                cm = common
                for v in range(n):
                    if cm & (1 << v):
                        if v not in clique:
                            cands.append(v)
                
                if not cands:
                    break
                
                # Pick the one with highest degree within common
                best_v = max(cands, key=lambda v: bin(common & rem_adj[v]).count('1'))
                clique.append(best_v)
                common &= rem_adj[best_v]
                added = True
            
            # Remove all edges within clique from remaining
            clique_set = set(clique)
            new_remaining = set()
            for u, v in remaining:
                if u in clique_set and v in clique_set:
                    # Remove this edge
                    pass
                else:
                    new_remaining.add((u, v))
            
            # Update rem_adj
            for u, v in remaining:
                if u in clique_set and v in clique_set:
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def solve_instance(n, p, seed):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        best_cliques = None
        best_count = len(edges)  # baseline: each edge is a 2-clique
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Try multiple greedy restarts with different random seeds
        rng_master = random.Random(42)
        num_restarts = 200 if n <= 32 else 50
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE:
                break
            rng = random.Random(rng_master.randint(0, 2**31))
            result = greedy_partition(n, edges, adj, rng, DEADLINE)
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
