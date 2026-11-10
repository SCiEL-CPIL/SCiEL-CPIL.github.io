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
            
            # Pick an edge with fewest common neighbors (hardest to cover)
            # Sample a few edges to find a good candidate
            edge_list = list(remaining)
            if len(edge_list) > 50:
                sample = rng.sample(edge_list, 50)
            else:
                sample = edge_list
            
            best_edge = None
            best_common_count = float('inf')
            for u, v in sample:
                common = rem_adj[u] & rem_adj[v]
                count = bin(common).count('1')
                if count < best_common_count:
                    best_common_count = count
                    best_edge = (u, v)
            
            u, v = best_edge
            clique = [u, v]
            common = rem_adj[u] & rem_adj[v]
            
            # Add vertices from common to clique greedily
            added = True
            while added:
                added = False
                cands = []
                cm = common
                for w in range(n):
                    if cm & (1 << w):
                        if w not in clique:
                            cands.append(w)
                
                if not cands:
                    break
                
                # Pick the one with highest degree within common
                best_w = max(cands, key=lambda w: bin(common & rem_adj[w]).count('1'))
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            # Remove all edges within clique from remaining
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    pass
                else:
                    new_remaining.add((u2, v2))
            
            # Update rem_adj
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
            
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
        num_restarts = 300 if n <= 32 else 80
        
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
