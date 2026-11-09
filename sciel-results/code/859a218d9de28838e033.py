import time
import math
from fractions import Fraction

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _compute_G(m, moduli_range):
    """Compute LCM of gcd(m, k) for all k in moduli_range, k != m."""
    g_list = []
    for k in moduli_range:
        if k == m:
            continue
        g = math.gcd(m, k)
        if g > 1:
            g_list.append(g)
    if not g_list:
        return 1
    l = 1
    for g in g_list:
        l = _lcm(l, g)
    return l

def construct():
    t0 = time.time()
    DEADLINE = 110.0
    MIN_M, MAX_M = 20, 200
    moduli = list(range(MIN_M, MAX_M + 1))
    
    # 1. Generate reduced nodes
    nodes = []
    for m in moduli:
        G = _compute_G(m, moduli)
        for r in range(G):
            weight = 1.0 / m
            nodes.append((m, r, weight))
            
    N = len(nodes)
    weights = [n[2] for n in nodes]
    ms = [n[0] for n in nodes]
    rs = [n[1] for n in nodes]
    
    # 2. Build conflict graph
    adj = [0] * N
    idx_by_m = {}
    for i in range(N):
        m = ms[i]
        if m not in idx_by_m:
            idx_by_m[m] = []
        idx_by_m[m].append(i)
        
    for m, idxs in idx_by_m.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i] |= (1 << j)
                    
    moduli_list = list(idx_by_m.keys())
    for u_idx in range(len(moduli_list)):
        m1 = moduli_list[u_idx]
        idxs1 = idx_by_m[m1]
        for v_idx in range(u_idx + 1, len(moduli_list)):
            m2 = moduli_list[v_idx]
            idxs2 = idx_by_m[m2]
            g = math.gcd(m1, m2)
            if g == 1:
                continue
            for i in idxs1:
                r1 = rs[i]
                rem1 = r1 % g
                for j in idxs2:
                    r2 = rs[j]
                    rem2 = r2 % g
                    if rem1 == rem2:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
    if time.time() > DEADLINE:
        return [[m, 0] for m in moduli[:10]]

    # 3. Sort and re-index
    sorted_indices = sorted(range(N), key=lambda i: -weights[i])
    new_weights = [weights[sorted_indices[k]] for k in range(N)]
    new_adj = [0] * N
    old_to_new = [0] * N
    for new_idx, old_idx in enumerate(sorted_indices):
        old_to_new[old_idx] = new_idx
        old_mask = adj[old_idx]
        new_mask = 0
        m_temp = old_mask
        while m_temp:
            bit = m_temp & (-m_temp)
            old_neighbor = bit.bit_length() - 1
            new_neighbor = old_to_new[old_neighbor]
            new_mask |= (1 << new_neighbor)
            m_temp ^= bit
        new_adj[new_idx] = new_mask
        
    best_score = 0.0
    best_nodes = []
    
    def get_sum_weight(mask):
        s = 0.0
        m_temp = mask
        while m_temp:
            bit = m_temp & (-m_temp)
            idx = bit.bit_length() - 1
            s += new_weights[idx]
            m_temp ^= bit
        return s

    def greedy_solution(mask):
        """Find a greedy independent set in mask."""
        s = 0.0
        selected = []
        m_temp = mask
        while m_temp:
            # Pick lowest index (highest weight)
            bit = m_temp & (-m_temp)
            v = bit.bit_length() - 1
            s += new_weights[v]
            selected.append(v)
            # Remove v and neighbors
            m_temp = m_temp & ~new_adj[v] & ~bit
        return s, selected

    def search(mask, current_sum, current_nodes):
        nonlocal best_score, best_nodes
        
        if time.time() > DEADLINE:
            return
            
        if mask == 0:
            if current_sum > best_score:
                best_score = current_sum
                best_nodes = list(current_nodes)
            return
            
        ub = current_sum + get_sum_weight(mask)
        if ub <= best_score:
            return
            
        # Greedy update
        g_val, g_nodes = greedy_solution(mask)
        if current_sum + g_val > best_score:
            best_score = current_sum + g_val
            best_nodes = current_nodes + g_nodes
            
        # Re-check bound after greedy update
        if ub <= best_score:
            return
            
        bit = mask & (-mask)
        v = bit.bit_length() - 1
        
        # Branch 1: Include v
        new_mask_include = mask & ~new_adj[v] & ~bit
        current_nodes.append(v)
        search(new_mask_include, current_sum + new_weights[v], current_nodes)
        current_nodes.pop()
        
        # Branch 2: Exclude v
        new_mask_exclude = mask & ~bit
        search(new_mask_exclude, current_sum, current_nodes)
        
    full_mask = (1 << N) - 1
    search(full_mask, 0.0, [])
    
    result = []
    for idx in best_nodes:
        old_idx = sorted_indices[idx]
        m, r, _ = nodes[old_idx]
        result.append([m, r])
        
    return result
