import random
import time
import math

def construct():
    deadline = time.time() + 118  # Safety margin under 120s kill
    rng = random.Random(42)
    
    # We aim for n=35 (Exoo's bound). If we can't find a valid one, we try 34, etc.
    # But the task says "Maximize n". 
    # Strategy: 
    # 1. Try to find a valid graph on n=35 using Simulated Annealing.
    # 2. If successful, return it.
    # 3. If not, try n=34, etc., down to a known valid size (e.g., 17 Paley).
    # Given the 120s limit, we should allocate most time to the highest n.
    
    def count_k4_and_ind6(n, A):
        """
        Returns (num_k4, num_ind6).
        A: list of int bitmasks. A[i] has bit j set if i-j is an edge.
        """
        k4_count = 0
        ind6_count = 0
        
        # Precompute complement for independent sets? 
        # Independent set in G is a clique in G_bar.
        # Let's compute G_bar adjacency masks.
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
            
        # Count K4 in G
        # Iterate over edges (u, v), then common neighbors w, then check if w connects to another common neighbor?
        # Faster: Iterate over pairs (u, v) that are edges.
        # Let N(u) be neighbors of u.
        # For each edge (u, v), let C = N(u) & N(v).
        # Any pair (w, x) in C that is an edge in G forms a K4 {u, v, w, x}.
        # So for each edge (u, v), we need to count edges in G[C].
        
        # This might be slow if done naively for every flip.
        # However, for n=35, n^4 is ~ 1.5M. 
        # We only update the penalty when we flip an edge.
        # We can maintain the count incrementally?
        # Or just recompute? 
        # Recomputing K4 count:
        # For each u:
        #   For each v in N(u) with v > u:
        #     C = N(u) & N(v) & {x > v}
        #     Count edges in C.
        
        # Let's try a simple recompute first to see if it's fast enough.
        # n=35. Edges ~ 300.
        # For each edge, intersection of two 35-bit ints is fast.
        # Counting edges in a set of size k is O(k^2) or using bit tricks.
        
        # Let's optimize K4 counting:
        # For each u:
        #   N_u = A[u]
        #   For each v in N_u (v > u):
        #     C = N_u & A[v]
        #     # We need to count edges in C.
        #     # An edge in C means for x in C, A[x] & C has bits.
        #     # Sum of degrees in induced subgraph / 2.
        #     # deg_sum = 0
        #     # for x in C: deg_sum += bin(A[x] & C).count('1')
        #     # k4_contrib = deg_sum // 2
        #     # This is O(n) per edge. Total O(E*n) ~ 300 * 35 = 10k ops. Very fast.
        
        for u in range(n):
            N_u = A[u]
            # Iterate v in N_u, v > u
            v_mask = N_u & ~((1 << (u + 1)) - 1) # v > u
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                C = N_u & A[v]
                # Count edges in C
                # To avoid double counting, we only count edges (x,y) in C where x < y.
                # Or just sum degrees and divide by 2.
                deg_sum = 0
                c_temp = C
                while c_temp:
                    x_bit = c_temp & -c_temp
                    x = x_bit.bit_length() - 1
                    c_temp ^= x_bit
                    deg_sum += bin(A[x] & C).count('1')
                k4_count += deg_sum // 2
                
        # Count Independent 6-sets (Clique of size 6 in G_bar)
        # Similar logic on comp.
        for u in range(n):
            N_u_comp = comp[u]
            v_mask = N_u_comp & ~((1 << (u + 1)) - 1)
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                C = N_u_comp & comp[v]
                deg_sum = 0
                c_temp = C
                while c_temp:
                    x_bit = c_temp & -c_temp
                    x = x_bit.bit_length() - 1
                    c_temp ^= x_bit
                    deg_sum += bin(comp[x] & C).count('1')
                # We found triangles in G_bar containing u,v.
                # Each Ind6 contains 4 triangles? No.
                # A clique of size 6 in G_bar.
                # The number of triangles in a K6 is C(6,3) = 20.
                # This counting method counts each K_k multiple times.
                # Specifically, iterating u < v < ... 
                # The loop above counts each triangle (u, v, w) with u < v < w exactly once?
                # No, u is the smallest, v is next, w is in C (w > v).
                # So it counts each triangle exactly once.
                # But we want K6s.
                # This approach counts triangles, not K6s directly.
                # If we just want to know if there is ANY independent set of size 6,
                # we can use a different check or a heuristic.
                # However, for SA, we need a penalty.
                # Penalty = (# K4s) + (# Ind6s).
                # Counting Ind6s exactly is harder.
                
                # Alternative: Use a simpler penalty for Ind6.
                # If we find a triangle in G_bar, it's a "bad" sign but not a violation yet.
                # We need a penalty that is 0 iff no Ind6 exists.
                # This is hard to do with a simple sum.
                
                # Let's stick to:
                # Penalty = (# K4s) + M * (1 if exists Ind6 else 0)
                # But checking "exists Ind6" is expensive (BK).
                
                # Better heuristic penalty:
                # Sum over all 6-subsets? No, too slow.
                
                # Let's use:
                # Penalty = (# K4s) + (# Triangles in G_bar)
                # Minimizing triangles in G_bar tends to reduce alpha.
                # If alpha < 6, there are no K6 in G_bar.
                # Minimizing triangles is a good proxy.
                
                ind6_proxy += deg_sum // 2 # This is actually counting triangles in G_bar
                
        # Wait, I need to define ind6_proxy outside.
        # Let's rewrite the function properly.
        
    def get_penalty(n, A):
        k4_count = 0
        tri_comp_count = 0
        
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
            
        # Count K4 in G
        for u in range(n):
            N_u = A[u]
            v_mask = N_u & ~((1 << (u + 1)) - 1)
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                C = N_u & A[v]
                deg_sum = 0
                c_temp = C
                while c_temp:
                    x_bit = c_temp & -c_temp
                    x = x_bit.bit_length() - 1
                    c_temp ^= x_bit
                    deg_sum += bin(A[x] & C).count('1')
                k4_count += deg_sum // 2
                
        # Count Triangles in G_bar (proxy for Ind6)
        for u in range(n):
            N_u_comp = comp[u]
            v_mask = N_u_comp & ~((1 << (u + 1)) - 1)
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                C = N_u_comp & comp[v]
                deg_sum = 0
                c_temp = C
                while c_temp:
                    x_bit = c_temp & -c_temp
                    x = x_bit.bit_length() - 1
                    c_temp ^= x_bit
                    deg_sum += bin(comp[x] & C).count('1')
                tri_comp_count += deg_sum // 2
                
        # We want 0 K4s and 0 Ind6s.
        # If tri_comp_count is 0, then there are no triangles in G_bar, so no Ind3 in G, which implies no Ind6.
        # Actually, if there are no triangles in G_bar, alpha(G) <= 2? No.
        # Triangle-free G_bar means G_bar has no K3. So G has no independent set of size 3.
        # This is a very strong condition. It might be too hard to satisfy.
        # We just need alpha(G) < 6.
        # So we don't need to eliminate all triangles in G_bar.
        
        # Let's use a different penalty for Ind6.
        # We can check if there is an Ind6 using a fast heuristic or limited BK.
        # But for SA, we need a continuous-ish or at least frequent signal.
        
        # Let's try:
        # Penalty = k4_count + 10 * (1 if has_ind6(n, A) else 0)
        # The cost of has_ind6 is the bottleneck.
        
        # Let's implement a fast has_ind6 that returns quickly if found.
        
        return k4_count, tri_comp_count

    def has_ind6_fast(n, A):
        """Returns True if there is an independent set of size 6."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
            
        # Bron-Kerbosch to find K6 in comp.
        found = [False]
        
        def bk(R, P, X):
            if found[0]:
                return
            if len(R) == 6:
                found[0] = True
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < 6:
                return
            
            # Pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                # Pick u in P with max degree in P
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and not found[0]:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def total_penalty(n, A):
        k4_count, tri_comp_count = get_penalty(n, A)
        # If k4_count is 0 and tri_comp_count is 0, it's likely valid, but we need to be sure about Ind6.
        # If tri_comp_count > 0, it doesn't mean Ind6 exists, but it's a penalty.
        # If k4_count > 0, it's a penalty.
        
        # Let's combine:
        # If k4_count == 0:
        #    if has_ind6_fast: penalty = 100
        #    else: penalty = 0
        # else:
        #    penalty = k4_count + 10 * tri_comp_count
        
        if k4_count == 0:
            if has_ind6_fast(n, A):
                return 100
            else:
                return 0
        else:
            return k4_count + 10 * tri_comp_count

    def valid(n, A):
        if total_penalty(n, A) == 0:
             # Double check with exact check? 
             # total_penalty 0 implies k4_count==0 and no ind6 (via fast check).
             # The fast check is exact (BK).
             # So if penalty is 0, it is valid.
             return True
        return False

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Main Search
    # Try n from 35 down to 20.
    # For each n, run SA for a fixed time budget.
    
    best_result = None
    
    # Allocate time
    # n=35: 60s
    # n=34: 30s
    # n=33: 10s
    # n=32: 5s
    # n=31: 3s
    # n=30: 2s
    # ...
    
    budgets = {
        35: 60,
        34: 30,
        33: 10,
        32: 5,
        31: 3,
        30: 2,
        29: 1,
        28: 0.5,
        27: 0.5,
        26: 0.2,
        25: 0.2,
        24: 0.1,
        23: 0.1,
        22: 0.05,
        21: 0.05,
        20: 0.05
    }
    
    for n in range(35, 19, -1):
        if time.time() > deadline - 2:
            break
        budget = budgets.get(n, 0.1)
        
        # Initialize random graph
        # Density around 0.5
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
                    
        current_penalty = total_penalty(n, A)
        if current_penalty == 0:
            best_result = {'n': n, 'edges': to_edges(n, A)}
            return best_result
            
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        start_time = time.time()
        while time.time() - start_time < budget:
            if time.time() > deadline - 1:
                break
                
            # Pick random edge to flip
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
                
            # Flip edge (u, v)
            A[u] ^= (1 << v)
            A[v] ^= (1 << u)
            
            new_penalty = total_penalty(n, A)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty == 0:
                    best_result = {'n': n, 'edges': to_edges(n, A)}
                    return best_result
            else:
                # Accept with probability exp(-delta/T)
                delta = new_penalty - current_penalty
                if delta < 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta / T)
                
                if accept:
                    current_penalty = new_penalty
                else:
                    # Revert
                    A[u] ^= (1 << v)
                    A[v] ^= (1 << u)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Fallback: Paley(17)
    if best_result is None:
        q = 17
        squares = {x * x % q for x in range(1, q)}
        n0 = q
        A = [0] * n0
        for i in range(n0):
            for j in range(n0):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        best_result = {'n': n0, 'edges': to_edges(n0, A)}
        
    return best_result
