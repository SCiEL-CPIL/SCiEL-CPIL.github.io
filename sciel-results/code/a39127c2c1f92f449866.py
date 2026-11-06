"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
import time

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_maximal_cliques(adj, n):
        """Yield all maximal cliques using Bron-Kerbosch with pivot selection."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                P.discard(v)
                X.add(v)
                
                bk(R, P_new, X_new)
                
                P.add(v)
                X.discard(v)
                R.discard(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(cliques, n):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        m = len(relevant_cliques)
        
        # Precompute bitmasks
        vertex_clique_mask = [0] * n
        clique_vertex_mask = [0] * m
        
        for i, c in enumerate(relevant_cliques):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)
                clique_vertex_mask[i] |= (1 << v)
        
        all_cliques_mask = (1 << m) - 1
        
        def get_uncovered(S_mask):
            uncovered = all_cliques_mask
            temp = S_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                uncovered &= ~vertex_clique_mask[v]
            return uncovered

        def is_covered(S_mask):
            return get_uncovered(S_mask) == 0

        def greedy_init(rng):
            S_mask = 0
            uncovered = all_cliques_mask
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if (S_mask >> v) & 1:
                        continue
                    covered = vertex_clique_mask[v] & uncovered
                    if covered:
                        score = bin(covered).count('1')
                        if score > best_score:
                            best_score = score
                            best_v = v
                
                if best_v == -1:
                    break
                
                S_mask |= (1 << best_v)
                uncovered &= ~vertex_clique_mask[best_v]
            
            return S_mask

        def local_search(S_mask, rng):
            # 1. Remove redundant vertices
            changed = True
            while changed:
                changed = False
                temp = S_mask
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    new_mask = S_mask ^ (1 << v)
                    if is_covered(new_mask):
                        S_mask = new_mask
                        changed = True
                        break
            
            # 2. Try 2-for-1 swaps to reduce size
            improved = True
            while improved:
                improved = False
                S_list = []
                temp = S_mask
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    S_list.append(v)
                
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        v = S_list[i]
                        w = S_list[j]
                        S_minus = S_mask & ~(1 << v) & ~(1 << w)
                        
                        uncovered = get_uncovered(S_minus)
                        
                        if uncovered == 0:
                            S_mask = S_minus
                            improved = True
                            break
                        
                        for u in range(n):
                            if (S_mask >> u) & 1:
                                continue
                            if (vertex_clique_mask[u] & uncovered) == uncovered:
                                S_mask = S_minus | (1 << u)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            return S_mask

        def perturb(S_mask, rng):
            S_list = []
            temp = S_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                S_list.append(v)
            
            if not S_list:
                return S_mask
            
            v_remove = rng.choice(S_list)
            candidates = [v for v in range(n) if not ((S_mask >> v) & 1)]
            if not candidates:
                return S_mask
            
            v_add = rng.choice(candidates)
            new_mask = (S_mask & ~(1 << v_remove)) | (1 << v_add)
            
            if is_covered(new_mask):
                return new_mask
            
            for _ in range(20):
                v_remove2 = rng.choice(S_list)
                v_add2 = rng.choice(candidates)
                new_mask2 = (S_mask & ~(1 << v_remove2)) | (1 << v_add2)
                if is_covered(new_mask2):
                    return new_mask2
            return S_mask

        best_mask = greedy_init(random.Random(42))
        best_size = bin(best_mask).count('1')
        
        rng = random.Random(42)
        start_time = time.time()
        time_limit = 30.0
        
        current_mask = best_mask
        
        while time.time() - start_time < time_limit:
            # Local Search
            current_mask = local_search(current_mask, rng)
            size = bin(current_mask).count('1')
            
            if size < best_size:
                best_size = size
                best_mask = current_mask
            
            # Perturb
            current_mask = perturb(current_mask, rng)
        
        # Convert best_mask to sorted list
        result = []
        temp = best_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            result.append(v)
        return sorted(result)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cover = solve_instance(cliques, n)
        out.append(cover)
    
    return out
# EVOLVE-BLOCK-END
