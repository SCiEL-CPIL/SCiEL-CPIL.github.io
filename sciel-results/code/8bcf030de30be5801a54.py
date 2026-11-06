"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from fractions import Fraction
    import sys

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Build complement adjacency: edge if lcm(a,b) <= 256
    # We want Maximum Weight Independent Set in this graph.
    comp_adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            g = gcd(ai, bj)
            if ai * bj // g <= N:
                comp_adj[i] |= (1 << j)
                comp_adj[j] |= (1 << i)

    weights = [Fraction(1, nums[i]) for i in range(n)]
    
    # Sort nodes by weight descending
    node_order = list(range(n))
    node_order.sort(key=lambda x: weights[x], reverse=True)
    
    best_weight = Fraction(0, 1)
    best_clique = []

    def get_bound(candidates_bitset):
        """
        Returns an upper bound on the maximum weight independent set
        within the candidate set using greedy coloring.
        """
        cands = []
        tmp = candidates_bitset
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            tmp ^= lsb
            
        if not cands:
            return Fraction(0, 1)
            
        # Sort by weight descending
        cands.sort(key=lambda x: weights[x], reverse=True)
        
        color_of = {}
        max_w_per_color = {}
        
        for u in cands:
            neighbor_colors = set()
            neighbors_bits = comp_adj[u] & candidates_bitset
            tmp_nb = neighbors_bits
            while tmp_nb:
                lsb = tmp_nb & (-tmp_nb)
                nb_idx = lsb.bit_length() - 1
                if nb_idx in color_of:
                    neighbor_colors.add(color_of[nb_idx])
                tmp_nb ^= lsb
                
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[u] = c
            if c not in max_w_per_color or weights[u] > max_w_per_color[c]:
                max_w_per_color[c] = weights[u]
                
        bound = sum(max_w_per_color.values())
        return bound

    all_cands = (1 << n) - 1
    sorted_indices = node_order

    def dfs(candidates_bitset, current_weight, current_set):
        nonlocal best_weight, best_clique
        
        if not candidates_bitset:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_set[:]
            return
            
        bound = get_bound(candidates_bitset)
        if current_weight + bound <= best_weight:
            return
            
        # Select branching node: highest weight node in candidates
        v = -1
        for idx in sorted_indices:
            if candidates_bitset & (1 << idx):
                v = idx
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        new_cands_include = candidates_bitset & ~(comp_adj[v] | (1 << v))
        dfs(new_cands_include, current_weight + weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        new_cands_exclude = candidates_bitset & ~(1 << v)
        dfs(new_cands_exclude, current_weight, current_set)

    dfs(all_cands, Fraction(0, 1), [])
    
    result = [nums[i] for i in best_clique]
    return result
# EVOLVE-BLOCK-END
