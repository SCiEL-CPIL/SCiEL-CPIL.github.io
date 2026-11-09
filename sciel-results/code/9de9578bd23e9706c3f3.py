import time
import math
from math import gcd
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for safety
    
    N = 511
    # Indices 0..510 represent values 1..511
    # Value v = i + 1
    weights = [0.0] * N
    for i in range(N):
        v = i + 1
        weights[i] = 1.0 / (512 - v)
        
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if gcd(i+1, j+1) > 1
    conflict_mask = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            vj = j + 1
            if gcd(vi, vj) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Helper to check if a node is compatible with a set (represented by bitmask)
    # set_mask has bit i set if i is in the set
    # node i is compatible with set if (conflict_mask[i] & set_mask) == 0
    def is_compatible(i, set_mask):
        return (conflict_mask[i] & set_mask) == 0

    def get_weight(set_mask):
        w = 0.0
        m = set_mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            w += weights[i]
            m ^= lsb
        return w

    # Greedy Initial Solution
    def greedy_solution(order):
        set_mask = 0
        for i in order:
            if is_compatible(i, set_mask):
                set_mask |= (1 << i)
        return set_mask

    # Local Search: Add and Swap
    def local_search(initial_mask, deadline):
        cur_mask = initial_mask
        cur_w = get_weight(cur_mask)
        
        # Phase 1: Add
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find best node to add
            best_i = -1
            best_w = 0.0
            for i in range(N):
                if (cur_mask >> i) & 1:
                    continue
                if is_compatible(i, cur_mask):
                    if weights[i] > best_w:
                        best_w = weights[i]
                        best_i = i
            if best_i != -1:
                cur_mask |= (1 << best_i)
                cur_w += best_w
                improved = True
                
        # Phase 2: Swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_swap_gain = 0.0
            best_remove = -1
            best_add = -1
            
            # Try removing each node in set
            m = cur_mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                m ^= lsb
                
                # Remove i
                new_mask = cur_mask & ~(1 << i)
                
                # Find best node to add to new_mask
                best_add_i = -1
                best_add_w = 0.0
                for j in range(N):
                    if (new_mask >> j) & 1:
                        continue
                    if is_compatible(j, new_mask):
                        if weights[j] > best_add_w:
                            best_add_w = weights[j]
                            best_add_i = j
                            
                if best_add_i != -1:
                    gain = best_add_w - weights[i]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_remove = i
                        best_add = best_add_i
                        
            if best_swap_gain > 1e-9:
                cur_mask &= ~(1 << best_remove)
                cur_mask |= (1 << best_add)
                cur_w += best_swap_gain
                improved = True
                
        return cur_mask, cur_w

    # Branch and Bound on a subset of nodes
    # nodes: list of indices in 0..N-1
    def bnb_mwis(nodes, deadline):
        K = len(nodes)
        if K == 0:
            return 0, 0
        
        # Map local indices 0..K-1 to global indices
        node_to_global = [0] * K
        for i, n in enumerate(nodes):
            node_to_global[i] = n
            
        # Build local conflict masks
        local_conflict = [0] * K
        for i in range(K):
            gi = node_to_global[i]
            mask = 0
            for j in range(K):
                if i == j:
                    continue
                gj = node_to_global[j]
                if (conflict_mask[gi] >> gj) & 1:
                    mask |= (1 << j)
            local_conflict[i] = mask
            
        best_w = [0.0]
        best_mask = [0]
        
        # Greedy lower bound
        order = sorted(range(K), key=lambda x: -weights[node_to_global[x]])
        cur_mask = 0
        cur_w = 0.0
        for i in order:
            if (local_conflict[i] & cur_mask) == 0:
                cur_mask |= (1 << i)
                cur_w += weights[node_to_global[i]]
        best_w[0] = cur_w
        best_mask[0] = cur_mask
        
        # Upper bound: Sum of max weights in cliques? 
        # Simpler UB: Sum of weights of all available nodes (too loose)
        # Better UB: Greedy coloring or clique cover.
        # For small K, we can use a simple bound: 
        # Sort available nodes by weight desc. 
        # If we pick a node, we remove its neighbors.
        # UB can be computed by a recursive greedy bound, but that's expensive.
        # Let's use a simple fractional bound or just rely on the structure.
        # Actually, for K<=70, a simple B&B with good pruning works.
        # Pruning: if cur_w + sum(weights of available) <= best_w, prune.
        
        def sum_weights_available(available_mask):
            s = 0.0
            m = available_mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                s += weights[node_to_global[i]]
                m ^= lsb
            return s

        def bb(available_mask, cur_w, chosen_mask):
            if time.time() > deadline:
                return
            if available_mask == 0:
                if cur_w > best_w[0]:
                    best_w[0] = cur_w
                    best_mask[0] = chosen_mask
                return
                
            # Prune
            ub = cur_w + sum_weights_available(available_mask)
            if ub <= best_w[0]:
                return
                
            # Select node to branch on: highest weight in available
            best_i = -1
            best_wi = -1.0
            m = available_mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                if weights[node_to_global[i]] > best_wi:
                    best_wi = weights[node_to_global[i]]
                    best_i = i
                m ^= lsb
                
            if best_i == -1:
                return
                
            # Branch 1: Include best_i
            # Remove best_i and its neighbors
            new_avail = available_mask & ~(local_conflict[best_i] | (1 << best_i))
            bb(new_avail, cur_w + weights[node_to_global[best_i]], chosen_mask | (1 << best_i))
            
            # Branch 2: Exclude best_i
            bb(available_mask & ~(1 << best_i), cur_w, chosen_mask)

        full_mask = (1 << K) - 1
        bb(full_mask, 0.0, 0)
        
        # Convert local mask to global mask
        global_mask = 0
        m = best_mask[0]
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            global_mask |= (1 << node_to_global[i])
            m ^= lsb
        return best_w[0], global_mask

    # Main Logic
    best_mask = 0
    best_w = 0.0
    
    # 1. Greedy + Local Search on full graph
    order = list(range(N))
    order.sort(key=lambda x: -weights[x])
    g_mask = greedy_solution(order)
    ls_mask, ls_w = local_search(g_mask, deadline)
    if ls_w > best_w:
        best_w = ls_w
        best_mask = ls_mask
        
    # 2. B&B on Top-K + Extension + Local Search
    # Try different K values
    for K in [40, 50, 60, 70]:
        if time.time() > deadline - 10:
            break
            
        # Top K nodes by weight
        top_k_nodes = [i for i, _ in sorted(enumerate(weights), key=lambda x: -x[1])[:K]]
        
        # Run B&B on these nodes
        bw, bm = bnb_mwis(top_k_nodes, deadline)
        
        # Extend to full graph: Add any compatible node from the rest
        full_mask = bm
        # Sort remaining nodes by weight
        remaining = [i for i in range(N) if i not in top_k_nodes]
        remaining.sort(key=lambda x: -weights[x])
        
        for i in remaining:
            if is_compatible(i, full_mask):
                full_mask |= (1 << i)
                
        # Local Search on this extended solution
        ext_mask, ext_w = local_search(full_mask, deadline)
        if ext_w > best_w:
            best_w = ext_w
            best_mask = ext_mask

    # 3. Random Restarts with Local Search
    # If time permits, do some random restarts
    rng = random.Random(42)
    while time.time() < deadline - 5:
        # Random order
        order = list(range(N))
        rng.shuffle(order)
        g_mask = greedy_solution(order)
        ls_mask, ls_w = local_search(g_mask, deadline)
        if ls_w > best_w:
            best_w = ls_w
            best_mask = ls_mask

    # Convert best_mask to list of values
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result.append(i + 1)
        m ^= lsb
        
    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
