import time
import itertools
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = 140.0

def _check_c5_bitmask(adj_masks, n):
    for v0 in range(n):
        mask0 = adj_masks[v0]
        for v1 in range(n):
            if v1 == v0 or not (mask0 >> v1) & 1:
                continue
            mask1 = adj_masks[v1]
            candidates_v2 = mask1 & ~(1 << v0)
            while candidates_v2:
                v2 = (candidates_v2 & -candidates_v2).bit_length() - 1
                candidates_v2 &= candidates_v2 - 1
                mask2 = adj_masks[v2]
                candidates_v3 = mask2 & ~(1 << v0) & ~(1 << v1)
                while candidates_v3:
                    v3 = (candidates_v3 & -candidates_v3).bit_length() - 1
                    candidates_v3 &= candidates_v3 - 1
                    mask3 = adj_masks[v3]
                    candidates_v4 = mask3 & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & mask0
                    if candidates_v4:
                        return True
    return False

def _has_c5(adj, n):
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
    return _check_c5_bitmask(masks, n)

def _has_independent_set_9(adj, n):
    comp = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if j != i and j not in adj[i]:
                m |= (1 << j)
        comp[i] = m
    
    best = 0
    found = [False]
    
    def bronk(R_size, P_mask, X_mask):
        nonlocal best, found
        if found[0]:
            return
        if P_mask == 0 and X_mask == 0:
            best = max(best, R_size)
            if best >= 9:
                found[0] = True
            return
        if P_mask == 0:
            return
        if R_size + bin(P_mask).count('1') <= best:
            return
        pivot = (P_mask & -P_mask).bit_length() - 1
        candidates = P_mask & ~comp[pivot]
        while candidates:
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            candidates ^= v_bit
            if found[0]:
                return
            P2 = P_mask & comp[v]
            X2 = X_mask & comp[v]
            bronk(R_size + 1, P2, X2)
            P_mask ^= v_bit
            X_mask |= v_bit
        return
    
    bronk(0, (1 << n) - 1, 0)
    return best >= 9

def _is_valid(adj, n):
    if _has_c5(adj, n):
        return False
    if _has_independent_set_9(adj, n):
        return False
    return True

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    n = 32
    adj = [set() for _ in range(n)]
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u].add(v)
                adj[v].add(u)
                
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to add vertices
    for target_n in range(33, 36):
        if time.time() > _DEADLINE - 20:
            break
        found = False
        for attempt in range(200):
            if time.time() > _DEADLINE - 20:
                break
            new_adj = [s.copy() for s in best_adj]
            new_n = best_n + 1
            if new_n > target_n:
                break
            new_adj.append(set())
            
            # Randomly connect new vertex
            random.seed(attempt + target_n * 1000)
            k = random.randint(0, 4)
            if k > 0:
                neighbors = random.sample(range(best_n), k)
                for nb in neighbors:
                    new_adj[best_n].add(nb)
                    new_adj[nb].add(best_n)
                    
            if _is_valid(new_adj, new_n):
                best_n = new_n
                best_adj = new_adj
                found = True
                break
        if not found:
            # Try modifying edges of existing graph to allow adding vertex?
            # For now, just break if we can't add easily
            pass

    edges = []
    for u in range(best_n):
        for v in sorted(best_adj[u]):
            if u < v:
                edges.append([u, v])
    return {'n': best_n, 'edges': edges}
