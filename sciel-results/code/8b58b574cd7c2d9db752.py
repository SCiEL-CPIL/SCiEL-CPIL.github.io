import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

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
        I_size = n - k
        
        # Precompute all 3-cliques of type {u, w, v} where u,w in K, v in I
        # Each such 3-clique covers K-edge (u,w) and cross-edges (u,v), (w,v)
        
        # For each v in I, the K-edges that can be covered with v are all pairs in N(v)
        # We want to select a set of (edge, v) pairs such that:
        # - No K-edge is used more than once
        # - No cross-edge (u,v) is used more than once (i.e., for fixed v, each u in K appears in at most one selected pair)
        # - Maximize the number of selected pairs
        
        # This is a 3-dimensional matching / bipartite matching variant
        # Model: left = K-edges, right = I-vertices, but with the constraint that for each v, the K-edges must form a matching in N(v)
        # Actually it's: for each v, we can select a matching in the subgraph induced by N(v) on K
        # And no K-edge can be selected for two different v's
        
        # Greedy approach: process K-edges in order of scarcity (fewest v's that can cover them)
        
        k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        
        # For each K-edge, which v's can cover it?
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)
        
        # Sort edges by scarcity
        edges_by_scarcity = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
        
        best_result = None
        best_count = float('inf')
        
        max_iters = 150
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Greedy matching
            used_k_edges = set()
            # For each v, track which K-vertices are "used" (can't be in another 3-clique with v)
            v_used_k = {v: set() for v in range(k, n)}
            # selected[v] = list of (u, w) K-edges assigned to v
            selected = {v: [] for v in range(k, n)}
            
            # Shuffle edges with same scarcity for diversity
            edges_to_process = edges_by_scarcity[:]
            random.shuffle(edges_to_process)
            # Stable sort by scarcity
            edges_to_process.sort(key=lambda e: len(edge_to_vs[e]))
            
            for e in edges_to_process:
                u, w = e
                if e in used_k_edges:
                    continue
                # Find valid v's
                candidates = [v for v in edge_to_vs[e] 
                              if u not in v_used_k[v] and w not in v_used_k[v]]
                if not candidates:
                    continue
                
                # Pick best candidate: maximize remaining flexibility
                # Heuristic: pick v with fewest remaining available K-edges
                def flexibility(v):
                    N_v = set(neighbors[v])
                    available = N_v - v_used_k[v]
                    return len(available)
                
                if random.random() < 0.15:
                    v = random.choice(candidates)
                else:
                    v = min(candidates, key=flexibility)
                
                # Assign
                v_used_k[v].add(u)
                v_used_k[v].add(w)
                selected[v].append((u, w))
                used_k_edges.add(e)
            
            # Build cliques
            cliques = []
            
            # 3-cliques from matching
            for v in range(k, n):
                for u, w in selected[v]:
                    cliques.append([u, w, v])
            
            # 2-cliques for unmatched cross-edges
            for v in range(k, n):
                N_v = set(neighbors[v])
                used_k_in_v = v_used_k[v]
                for u in N_v - used_k_in_v:
                    cliques.append([u, v])
            
            # Cover remaining K-edges with 2-cliques
            remaining_k_edges = set(k_edges) - used_k_edges
            for u, w in remaining_k_edges:
                cliques.append([u, w])
            
            # Validate
            clean = []
            for c in cliques:
                clean.append([int(x) for x in c])
            
            # Verify edge coverage
            edge_count = defaultdict(int)
            for c in clean:
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        a, b = min(c[i], c[j]), max(c[i], c[j])
                        edge_count[(a, b)] += 1
            
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            valid = True
            for e in all_edges:
                if edge_count[e] != 1:
                    valid = False
                    break
            for e, cnt in edge_count.items():
                if e not in all_edges:
                    valid = False
                    break
            
            if not valid:
                continue
            
            count = len(clean)
            if count < best_count:
                best_count = count
                best_result = clean
        
        if best_result is None:
            # Fallback: all 2-cliques
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            best_result = [[int(x) for x in c] for c in cliques]
        
        return best_result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    while len(results) < 3:
        results.append([])
        
    return results
