"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        # adj is list of int bitmasks
        
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R != 0:
                    # Check size
                    if bin(R).count('1') >= 2:
                        cliques.append(R)
                return
            
            # Pivot selection
            # Choose vertex from P or X with max degree in P
            best_pivot = -1
            max_count = -1
            P_temp = P
            while P_temp:
                v = (P_temp & -P_temp).bit_length() - 1
                P_temp &= P_temp - 1
                count = bin(P & adj[v]).count('1')
                if count > max_count:
                    max_count = count
                    best_pivot = v
            
            X_temp = X
            while X_temp:
                v = (X_temp & -X_temp).bit_length() - 1
                X_temp &= X_temp - 1
                count = bin(P & adj[v]).count('1')
                if count > max_count:
                    max_count = count
                    best_pivot = v
            
            if best_pivot == -1:
                candidates = P
            else:
                candidates = P & ~adj[best_pivot]
            
            while candidates:
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                
                bk(R | v_bit, P & adj[v], X & adj[v])
                P ^= v_bit
                X |= v_bit

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(n, p, seed):
        if time.time() > deadline:
            return []
            
        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Set Cover Problem:
        # Universe: cliques (represented as bitmasks of vertices)
        # Sets: vertices 0..n-1, set i contains clique c if (c >> i) & 1
        
        # Precompute for each vertex, which cliques it covers?
        # Or just iterate.
        
        best_solution = None
        best_size = float('inf')
        
        # Helper: check if S covers all cliques
        def is_valid(S_mask):
            for c in cliques:
                if (S_mask & c) == 0:
                    return False
            return True
            
        # Helper: get list of vertices from mask
        def mask_to_list(mask):
            return [i for i in range(n) if (mask >> i) & 1]
            
        # Local Search: Reduce S_mask
        def local_search_reduce(S_mask):
            changed = True
            while changed:
                changed = False
                # Try removing each vertex
                S_temp = S_mask
                while S_temp:
                    v_bit = S_temp & -S_temp
                    v = v_bit.bit_length() - 1
                    S_temp ^= v_bit
                    
                    new_S = S_mask ^ v_bit
                    if is_valid(new_S):
                        S_mask = new_S
                        changed = True
                        # Restart loop as S changed
                        break 
            return S_mask

        # Greedy Set Cover
        def greedy_init(randomize=False):
            S_mask = 0
            remaining_cliques = list(cliques)
            
            while remaining_cliques:
                if randomize:
                    # Pick random clique to prioritize? No, standard greedy is better.
                    # But we can randomize tie-breaking.
                    pass
                    
                best_v = -1
                max_cover = -1
                # To allow random tie-breaking, we can collect all bests
                candidates = []
                
                for v in range(n):
                    if (S_mask >> v) & 1:
                        continue
                    # Count how many remaining cliques v covers
                    cover_count = 0
                    for c in remaining_cliques:
                        if (c >> v) & 1:
                            cover_count += 1
                    if cover_count > max_cover:
                        max_cover = cover_count
                        candidates = [v]
                    elif cover_count == max_cover and cover_count > 0:
                        candidates.append(v)
                        
                if not candidates:
                    break # Should not happen if is_valid logic is correct, but safety
            
                if randomize and candidates:
                    v = random.choice(candidates)
                else:
                    v = candidates[0]
                    
                S_mask |= (1 << v)
                
                # Remove covered cliques
                new_remaining = []
                for c in remaining_cliques:
                    if not ((c >> v) & 1):
                        new_remaining.append(c)
                remaining_cliques = new_remaining
                
            return S_mask

        # Run multiple iterations
        # 1. Pure Greedy
        S = greedy_init(randomize=False)
        S = local_search_reduce(S)
        if bin(S).count('1') < best_size:
            best_size = bin(S).count('1')
            best_solution = S
            
        # 2. Randomized Greedy + Local Search
        iterations = 500
        for _ in range(iterations):
            if time.time() > deadline:
                break
            S = greedy_init(randomize=True)
            S = local_search_reduce(S)
            if bin(S).count('1') < best_size:
                best_size = bin(S).count('1')
                best_solution = S
                
            # 3. Perturbation on best solution
            if best_solution is not None:
                S_pert = best_solution
                # Pick a random v in S and u not in S
                S_list = mask_to_list(S_pert)
                if S_list:
                    v = random.choice(S_list)
                    # Find u not in S
                    possible_u = [u for u in range(n) if not ((S_pert >> u) & 1)]
                    if possible_u:
                        u = random.choice(possible_u)
                        S_pert = (S_pert ^ (1 << v)) | (1 << u)
                        S_pert = local_search_reduce(S_pert)
                        if bin(S_pert).count('1') < best_size:
                            best_size = bin(S_pert).count('1')
                            best_solution = S_pert
                            
        if best_solution is None:
            # Fallback: all vertices
            return list(range(n))
            
        return mask_to_list(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append([])
            continue
        res = solve_instance(n, p, seed)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
