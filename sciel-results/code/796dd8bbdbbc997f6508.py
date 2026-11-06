import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Construct the base graph: Polarity Graph of PG(2,5)
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj_mask = [0] * n
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    
    def count_c4_add(u, v):
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        count = 0
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            count += (adj_mask[w] & nv).bit_count()
        return count

    def count_c4_remove(u, v):
        # Count C4s that contain edge (u,v)
        # A C4 containing (u,v) must have a path u-w-x-v of length 3.
        # So w is common neighbor of u and x? No.
        # The C4 is u-v-x-w-u.
        # w is adjacent to u, x is adjacent to v.
        # w and x must be adjacent? No, u-v-x-w-u implies w-u and v-x and u-v and x-w.
        # So common neighbors of u and v form the "other" pairs?
        # Actually, C4s containing (u,v) correspond to pairs (w,x) such that w~u, x~v, w~x?
        # No, u-v-x-w-u.
        # w is adjacent to u. x is adjacent to v.
        # w is adjacent to x.
        # So we need to count pairs (w,x) where w in N(u), x in N(v), w~x.
        # Note: w != v, x != u.
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        count = 0
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            # w is a neighbor of u.
            # We need x in N(v) such that w~x.
            # So count common neighbors of w and v?
            # x is in N(v) and N(w).
            # So count = popcount(N(w) & N(v)).
            # But x cannot be u? N(w) & N(v) might include u if w~u and u~v.
            # But x is the vertex after v.
            # The cycle is u-v-x-w-u.
            # x is adjacent to v and w.
            # So x is a common neighbor of v and w.
            count += (adj_mask[w] & nv).bit_count()
        return count

    def get_delta(u, v, x, y):
        lost = count_c4_remove(u, v)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        gained = count_c4_add(x, y)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        return gained - lost

    def count_total_c4s():
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj_mask[i]:
                    common = adj_mask[i] & adj_mask[j]
                    common &= ~(1 << i) & ~(1 << j)
                    c = common.bit_count()
                    total += c * (c - 1) // 2
        return total

    current_c4 = count_total_c4s()
    best_c4 = current_c4
    best_adj_mask = adj_mask[:]
    best_edge_set = set(edge_set)
    
    # 2. Adjust edge count to exactly m (91)
    
    # Helper to find edge with max C4 participation
    def find_max_c4_edge():
        max_count = -1
        max_edge = None
        for u, v in edge_set:
            c = count_c4_remove(u, v)
            if c > max_count:
                max_count = c
                max_edge = (u, v)
        return max_edge, max_count

    # Helper to find non-edge with min C4 creation
    def find_min_c4_non_edge():
        min_count = float('inf')
        min_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    c = count_c4_add(i, j)
                    if c < min_count:
                        min_count = c
                        min_edge = (i, j)
        return min_edge, min_count

    while current_edges > m:
        if time.time() > deadline - 20: break
        edge_to_remove, _ = find_max_c4_edge()
        if not edge_to_remove: break
        u, v = edge_to_remove
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edge_set.remove((u, v))
        current_edges -= 1
        current_c4 = count_total_c4s() # Recalc for accuracy after removal
        
    while current_edges < m:
        if time.time() > deadline - 20: break
        edge_to_add, _ = find_min_c4_non_edge()
        if not edge_to_add: break
        u, v = edge_to_add
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))
        current_edges += 1
        current_c4 = count_total_c4s() # Recalc for accuracy after addition
        
    if current_edges != m:
        # Fallback: force adjustment if failed
        # This should not happen if logic is correct
        pass

    best_c4 = current_c4
    best_adj_mask = adj_mask[:]
    best_edge_set = set(edge_set)

    current_edges_list = list(edge_set)
    
    # 3. Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    while time.time() < deadline - 10:
        T = 5.0
        while T > T_min and time.time() < deadline - 10:
            idx_rm = random.randint(0, len(current_edges_list) - 1)
            e_rm = current_edges_list[idx_rm]
            u, v = e_rm
            
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) not in edge_set:
                    break
            
            delta = get_delta(u, v, x, y)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                edge_set.remove(e_rm)
                current_edges_list[idx_rm] = (x, y)
                
                adj_mask[x] |= (1 << y)
                adj_mask[y] |= (1 << x)
                edge_set.add((x, y))
                
                current_c4 += delta
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj_mask = adj_mask[:]
                    best_edge_set = set(edge_set)
                    
            T *= cooling_rate

    # 4. Final Polish: 1-1 swap local search
    if time.time() < deadline - 3:
        adj_mask = best_adj_mask[:]
        edge_set = set(best_edge_set)
        current_c4 = best_c4
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            edges_list = list(edge_set)
            # Sort edges by potential impact? Just iterate
            for e_rm in edges_list:
                if time.time() > deadline - 3:
                    break
                u, v = e_rm
                # Find best replacement
                best_delta = 0
                best_repl = None
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) in edge_set:
                            continue
                        if (i, j) == (u, v):
                            continue
                        d = get_delta(u, v, i, j)
                        if d < best_delta:
                            best_delta = d
                            best_repl = (i, j)
                
                if best_repl and best_delta < 0:
                    x, y = best_repl
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    edge_set.remove(e_rm)
                    
                    adj_mask[x] |= (1 << y)
                    adj_mask[y] |= (1 << x)
                    edge_set.add((x, y))
                    
                    current_c4 += best_delta
                    improved = True
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj_mask = adj_mask[:]
                        best_edge_set = set(edge_set)
                    break # Restart loop since edge set changed

    if best_edge_set and len(best_edge_set) == m:
        return [[u, v] for u, v in best_edge_set]
    else:
        # Fallback to valid structure if any
        if len(edge_set) == m:
            return [[u, v] for u, v in edge_set]
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
