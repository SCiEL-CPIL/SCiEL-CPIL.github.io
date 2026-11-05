"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math

    N = 512
    # 1. Precompute prime factorizations for 1..N
    factors = [None] * (N + 1)
    factors[1] = {}
    for i in range(2, N + 1):
        x = i
        fac = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                fac[d] = fac.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            fac[x] = fac.get(x, 0) + 1
        factors[i] = fac

    # 2. Build the conflict graph using bitsets
    # neighbors[i] is a bitset of nodes j such that lcm(i, j) > N
    neighbors = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            fac_i = factors[i]
            fac_j = factors[j]
            lcm_val = 1
            primes = set(fac_i.keys()).union(fac_j.keys())
            for p in primes:
                exp_i = fac_i.get(p, 0)
                exp_j = fac_j.get(p, 0)
                max_exp = max(exp_i, exp_j)
                if max_exp > 0:
                    lcm_val *= (p ** max_exp)
                    if lcm_val > N:
                        break
            if lcm_val > N:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)

    # 3. Maximum Independent Set using Branch and Bound with Bitsets
    best_size = 0
    best_set = 0

    all_nodes = (1 << (N + 1)) - 2

    def bit_count(x):
        return bin(x).count('1')

    # Greedy coloring bound for the conflict graph
    def greedy_color_bound(cand):
        uncolored = cand
        color = 0
        while uncolored != 0:
            color += 1
            # Pick a vertex with max degree in the remaining uncolored set
            # For simplicity and speed, we just iterate through bits
            # A better heuristic: pick vertex with highest degree in uncolored
            best_v = -1
            best_deg = -1
            tmp = uncolored
            while tmp:
                v_bit = tmp & (-tmp)
                v_idx = v_bit.bit_length() - 1
                deg = bit_count(neighbors[v_idx] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v_idx
                tmp ^= v_bit
            
            if best_v == -1:
                break
                
            v_bit = 1 << best_v
            uncolored ^= v_bit
            # Remove neighbors of v from uncolored (they can share this color)
            uncolored &= ~neighbors[best_v]
            
        return color

    def solve(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        # Pruning with coloring bound
        if current_size + greedy_color_bound(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Heuristic: pick a node with high degree in the induced subgraph
        # We pick the node with the highest degree among candidates
        v_idx = -1
        max_deg = -1
        tmp = candidates
        while tmp:
            v_bit = tmp & (-tmp)
            idx = v_bit.bit_length() - 1
            deg = bit_count(neighbors[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            tmp ^= v_bit
            
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v
        new_cands_include = candidates & ~neighbors[v_idx] & ~v_bit
        solve(new_cands_include, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        new_cands_exclude = candidates & ~v_bit
        solve(new_cands_exclude, current_set, current_size)

    solve(all_nodes, 0, 0)
    
    result = []
    for i in range(1, N + 1):
        if best_set & (1 << i):
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
