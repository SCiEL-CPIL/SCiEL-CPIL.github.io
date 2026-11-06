import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n = 18
    all_vertices = list(range(n))

    # ---------- Fast bitmask helpers ----------
    def neighbors_from_bm(bm):
        """Return list of neighbor bitmasks for each vertex."""
        nb = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and (bm >> (i * n + j)) & 1:
                    mask |= (1 << j)
            nb[i] = mask
        return nb

    def max_triangle_free_induced_bm(bm, timeout=5.0):
        """
        Exact branch and bound for max triangle-free induced subgraph size.
        Optimized to use bitmasks for forbidden vertices.
        """
        best = [0]
        start = time.time()
        
        # Precompute neighbor bitmasks
        nb = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (bm >> (i * n + j)) & 1:
                    nb[i] |= (1 << j)

        # Precompute edge existence matrix for O(1) check
        is_edge = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if (bm >> (i * n + j)) & 1:
                    is_edge[i][j] = True
                    is_edge[j][i] = True

        def dfs(current_mask, candidates_mask, forbidden_list):
            if time.time() - start > timeout:
                return
            
            cur_size = bin(current_mask).count('1')
            cand_size = bin(candidates_mask).count('1')
            
            # Upper bound: current size + remaining candidates
            if cur_size + cand_size <= best[0]:
                return
            
            if cand_size == 0:
                if cur_size > best[0]:
                    best[0] = cur_size
                return

            # Branch on a candidate vertex
            # Heuristic: pick vertex with highest degree in candidate subgraph? 
            # Or just lowest index for simplicity.
            # Let's pick the lowest set bit.
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask & ~v_bit

            # Option 1: Exclude v
            dfs(current_mask, remaining, forbidden_list)

            # Option 2: Include v
            # Check if v is forbidden by current set
            # forbidden_list[v] contains bits of vertices that form a triangle with v and some u in current_mask
            # Actually, standard approach:
            # If we add v, any u in current_mask such that (u,v) is edge and there is w in current_mask 
            # such that (u,w) and (v,w) are edges... wait.
            
            # Correct Logic:
            # A set S is triangle-free iff for every u in S, no two neighbors of u in S are connected.
            # Equivalently, when adding v to S, we must ensure that for every u in S, 
            # v is not connected to any neighbor of u in S.
            # i.e., N(v) & N(u) & S must be empty for all u in S.
            # Or simpler: 
            # If we add v, then for any u in S, if (u,v) is an edge, then N(u) & N(v) & S must be empty?
            # No. If (u,v) is an edge, and there is w in S such that (u,w) and (v,w) are edges, 
            # then {u,v,w} is a triangle.
            # So, for each u in S, if (u,v) is edge, then (N(u) & N(v) & S) must be 0.
            
            # Let's check validity of adding v.
            valid = True
            for u in range(n):
                if (current_mask >> u) & 1:
                    if is_edge[u][v]:
                        # Check if any common neighbor of u and v is in current_mask
                        common = nb[u] & nb[v] & current_mask
                        if common:
                            valid = False
                            break
            
            if valid:
                new_current = current_mask | v_bit
                # Update forbidden list for remaining candidates?
                # If we add v, any candidate w that forms a triangle with v and some u in new_current
                # must be removed from candidates.
                # A triangle is formed by {u, v, w} if (u,v), (v,w), (u,w) are edges.
                # So for each w in remaining:
                # If (v,w) is edge, then N(v) & N(w) & new_current must be 0?
                # No, if (v,w) is edge, we need to ensure no u in new_current is connected to both v and w.
                # i.e., N(v) & N(w) & new_current == 0.
                
                new_remaining = 0
                for w in range(n):
                    if (remaining >> w) & 1:
                        w_bit = 1 << w
                        # Check if w can be added later
                        # w is forbidden if there exists u in new_current such that 
                        # (u,w) is edge and N(u) & N(w) & new_current is not empty?
                        # Actually, we just need to check if w creates a triangle with new_current.
                        # Triangle {u, v, w} or {u, w, v} etc.
                        # If we add w later, it must not form a triangle with any pair in new_current.
                        # Pair (u, v) in new_current: if (u,v) is edge, check if w is common neighbor.
                        # Pair (u, x) where x in old current: if (u,x) is edge, check if w is common neighbor.
                        
                        # Efficient check:
                        # w is incompatible with new_current if there exists u in new_current 
                        # such that (u,w) is edge and (N(u) & N(w) & new_current) != 0?
                        # No, that's for adding w to a set containing u.
                        # The condition for w to be addable to S is:
                        # For all u in S, if (u,w) is edge, then N(u) & N(w) & S == 0.
                        
                        can_add_w = True
                        for u in range(n):
                            if (new_current >> u) & 1:
                                if is_edge[u][w]:
                                    common = nb[u] & nb[w] & new_current
                                    if common:
                                        can_add_w = False
                                        break
                        if can_add_w:
                            new_remaining |= w_bit

                dfs(new_current, new_remaining, forbidden_list)

        dfs(0, (1 << n) - 1, [0]*n)
        return best[0]

    def greedy_triangle_free_size_bm(bm, trials=20):
        """Quick greedy estimate."""
        nb = neighbors_from_bm(bm)
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                ok = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        if (nb[v] >> current[i]) & 1 and (nb[v] >> current[j]) & 1 and \
                           (nb[current[i]] >> current[j]) & 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    current.append(v)
            best = max(best, len(current))
        return best

    # ---------- K4 check ----------
    def is_k4_free_bm(bm):
        """Check K4-freeness. For each 4-set, check if all 6 edges present."""
        for a, b, c, d in combinations(range(n), 4):
            if ((bm >> (a*n+b)) & 1) and ((bm >> (a*n+c)) & 1) and ((bm >> (a*n+d)) & 1) and \
               ((bm >> (b*n+c)) & 1) and ((bm >> (b*n+d)) & 1) and ((bm >> (c*n+d)) & 1):
                return False
        return True

    def try_add_edge(bm, u, v):
        """Return new bm if adding (u,v) keeps K4-free, else None."""
        if (bm >> (u * n + v)) & 1:
            return None
        new_bm = bm | (1 << (u * n + v)) | (1 << (v * n + u))
        # Check: common neighbors of u and v that form an edge -> K4
        nb_u = 0
        nb_v = 0
        for w in range(n):
            if w != u and w != v:
                if (bm >> (u * n + w)) & 1:
                    nb_u |= (1 << w)
                if (bm >> (v * n + w)) & 1:
                    nb_v |= (1 << w)
        common = nb_u & nb_v
        cl = [w for w in range(n) if (common >> w) & 1]
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if (bm >> (cl[i] * n + cl[j])) & 1:
                    return None
        return new_bm

    def try_remove_edge(bm, u, v):
        """Return new bm with edge (u,v) removed."""
        if not ((bm >> (u * n + v)) & 1):
            return None
        return bm & ~((1 << (u * n + v)) | (1 << (v * n + u)))

    # ---------- Initial construction ----------
    # Start with random graph and refine, or use a known good structure
    # Let's try a few random starts and pick the best initial score
    best_initial_bm = 0
    best_initial_score = 999
    
    for _ in range(5):
        if time.time() > deadline - 120:
            break
        rbm = 0
        for u, v in [(i, j) for i in range(n) for j in range(i + 1, n)]:
            if random.random() < 0.5:
                new_bm = try_add_edge(rbm, u, v)
                if new_bm is not None:
                    rbm = new_bm
        r_score = max_triangle_free_induced_bm(rbm, timeout=2.0)
        if r_score < best_initial_score:
            best_initial_score = r_score
            best_initial_bm = rbm
            
    current_bm = best_initial_bm
    current_score = best_initial_score
    best_bm = current_bm
    best_score = current_score

    # ---------- Phase 1: Simulated Annealing with bitmask ----------
    T = 3.0
    cooling = 0.999
    no_improve = 0
    edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    while time.time() < deadline - 10:
        # Pick random edge
        u, v = random.choice(edge_pairs)
        if (current_bm >> (u * n + v)) & 1:
            new_bm = try_remove_edge(current_bm, u, v)
        else:
            new_bm = try_add_edge(current_bm, u, v)

        if new_bm is None:
            T *= cooling
            no_improve += 1
            continue

        # Quick greedy check
        g_score = greedy_triangle_free_size_bm(new_bm, trials=5)
        delta_g = g_score - current_score

        if delta_g < 0 or random.random() < np.exp(-delta_g / max(T, 0.01)):
            exact_score = max_triangle_free_induced_bm(new_bm, timeout=1.0)
            delta = exact_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                current_bm = new_bm
                current_score = exact_score
                no_improve = 0
                if exact_score < best_score:
                    best_bm = new_bm
                    best_score = exact_score
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        if no_improve > 100:
            T = max(T, 0.5)
            no_improve = 0

    # ---------- Phase 2: Focused local search on best graph ----------
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u, v in edge_pairs:
            if time.time() > deadline - 5:
                break
            if (best_bm >> (u * n + v)) & 1:
                cand_bm = try_remove_edge(best_bm, u, v)
            else:
                cand_bm = try_add_edge(best_bm, u, v)
            if cand_bm is None:
                continue
            cand_score = max_triangle_free_induced_bm(cand_bm, timeout=0.5)
            if cand_score < best_score:
                best_bm = cand_bm
                best_score = cand_score
                improved = True

    # ---------- Final exact verification ----------
    final_score = max_triangle_free_induced_bm(best_bm, timeout=5.0)
    if final_score < best_score:
        best_score = final_score

    # ---------- Convert to edge list ----------
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_bm >> (i * n + j)) & 1:
                edges.append([i, j])
    return edges
