import random
from collections import defaultdict
import numpy as np

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        if P:
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def greedy_init(clique_masks, n):
        T = 0
        remaining = list(range(len(clique_masks)))
        while remaining:
            # Count how many remaining cliques each vertex covers
            counts = [0] * n
            for idx in remaining:
                mask = clique_masks[idx]
                # Iterate bits in mask
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    counts[v] += 1
                    m ^= lsb
            
            best_v = -1
            best_count = -1
            for v in range(n):
                if counts[v] > best_count:
                    best_count = counts[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            T |= (1 << best_v)
            
            # Remove covered cliques
            new_remaining = []
            for idx in remaining:
                if (clique_masks[idx] & (1 << best_v)) == 0:
                    new_remaining.append(idx)
            remaining = new_remaining
            
        return T

    def is_valid(T_mask, clique_masks):
        for c in clique_masks:
            if (c & T_mask) == 0:
                return False
        return True

    def local_search(T_mask, clique_masks, n):
        # 1. Remove redundant
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if (T_mask >> v) & 1:
                    new_T = T_mask ^ (1 << v)
                    if is_valid(new_T, clique_masks):
                        T_mask = new_T
                        changed = True
        
        # 2. Swap 2-for-1
        improved = True
        while improved:
            improved = False
            T_list = [i for i in range(n) if (T_mask >> i) & 1]
            for i in range(len(T_list)):
                for j in range(i+1, len(T_list)):
                    v1, v2 = T_list[i], T_list[j]
                    mask_remove = (1 << v1) | (1 << v2)
                    base_T = T_mask ^ mask_remove
                    for u in range(n):
                        if (T_mask >> u) & 1:
                            continue
                        new_T = base_T | (1 << u)
                        if is_valid(new_T, clique_masks):
                            T_mask = new_T
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 3. Swap 3-for-1
        improved = True
        while improved:
            improved = False
            T_list = [i for i in range(n) if (T_mask >> i) & 1]
            for i in range(len(T_list)):
                for j in range(i+1, len(T_list)):
                    for k in range(j+1, len(T_list)):
                        v1, v2, v3 = T_list[i], T_list[j], T_list[k]
                        mask_remove = (1 << v1) | (1 << v2) | (1 << v3)
                        base_T = T_mask ^ mask_remove
                        for u in range(n):
                            if (T_mask >> u) & 1:
                                continue
                            new_T = base_T | (1 << u)
                            if is_valid(new_T, clique_masks):
                                T_mask = new_T
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return T_mask

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
            
        clique_masks = []
        for c in relevant_cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
            
        # Initial Greedy
        T_mask = greedy_init(clique_masks, n)
        
        # Local Search
        T_mask = local_search(T_mask, clique_masks, n)
        
        # Iterated Local Search (Perturbation)
        best_T = T_mask
        best_size = bin(T_mask).count('1')
        
        import time
        start_time = time.time()
        
        # Try a few perturbations if time allows
        # For n=40, this is fast.
        for _ in range(20):
            if time.time() - start_time > 30: break # Safety per instance
            
            # Perturb: remove 1-2 random vertices
            T_list = [i for i in range(n) if (best_T >> i) & 1]
            if not T_list:
                break
            
            perturbed_T = best_T
            # Remove 1 or 2 vertices
            num_remove = random.randint(1, min(2, len(T_list)))
            to_remove = random.sample(T_list, num_remove)
            for v in to_remove:
                perturbed_T ^= (1 << v)
                
            # Fix: Greedy add
            remaining_cliques = [c for c in clique_masks if (c & perturbed_T) == 0]
            
            while remaining_cliques:
                # Pick a random uncovered clique, pick a random vertex from it?
                # Or Greedy: pick vertex covering most uncovered
                counts = [0] * n
                for c in remaining_cliques:
                    m = c
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        counts[v] += 1
                        m ^= lsb
                
                best_v = -1
                best_count = -1
                for v in range(n):
                    if counts[v] > best_count:
                        best_count = counts[v]
                        best_v = v
                
                if best_v == -1:
                    break
                
                perturbed_T |= (1 << best_v)
                
                new_remaining = []
                for c in remaining_cliques:
                    if (c & (1 << best_v)) == 0:
                        new_remaining.append(c)
                remaining_cliques = new_remaining
                
            # Local Search on perturbed
            perturbed_T = local_search(perturbed_T, clique_masks, n)
            
            if bin(perturbed_T).count('1') < best_size:
                best_T = perturbed_T
                best_size = bin(perturbed_T).count('1')
                
        return [i for i in range(n) if (best_T >> i) & 1]

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        results.append(solve_instance(n, p, seed))
        
    return results
