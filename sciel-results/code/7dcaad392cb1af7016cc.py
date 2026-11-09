import random
import itertools
import time

# Precompute Q3 masks
def generate_q3_masks():
    std_edges = set()
    for i in range(8):
        for j in range(8):
            if i < j and bin(i ^ j).count('1') == 1:
                std_edges.add((i, j))
    
    masks = set()
    for perm in itertools.permutations(range(8)):
        new_edges = set()
        for i, j in std_edges:
            u, v = perm[i], perm[j]
            if u > v: u, v = v, u
            new_edges.add((u, v))
        
        mask = 0
        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (i, j) in new_edges:
                    mask |= (1 << idx)
                idx += 1
        masks.add(mask)
    return list(masks)

Q3_MASKS = generate_q3_masks()

def edge_index(i, j):
    if i > j: i, j = j, i
    return i*7 - i*(i-1)//2 + (j - i - 1)

Q3_MASKS_BY_EDGE = [[] for _ in range(28)]
for mask in Q3_MASKS:
    for k in range(28):
        if (mask >> k) & 1:
            Q3_MASKS_BY_EDGE[k].append(mask)

SUBSETS_CONTAINING_PAIR = {}
ALL_SUBSETS = list(itertools.combinations(range(16), 8))
for subset in ALL_SUBSETS:
    for i in range(8):
        for j in range(i+1, 8):
            u, v = subset[i], subset[j]
            if u > v: u, v = v, u
            if (u, v) not in SUBSETS_CONTAINING_PAIR:
                SUBSETS_CONTAINING_PAIR[(u, v)] = []
            SUBSETS_CONTAINING_PAIR[(u, v)].append(subset)

def construct():
    n = 16
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.9:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    start_time = time.time()
    time_limit = 115
    
    def get_local_mask(subset, adj_bits):
        local_mask = 0
        for li in range(8):
            for lj in range(li+1, 8):
                ui = subset[li]
                uj = subset[lj]
                if (adj_bits[ui] >> uj) & 1:
                    k = edge_index(li, lj)
                    local_mask |= (1 << k)
        return local_mask

    energy = 0
    for subset in ALL_SUBSETS:
        lm = get_local_mask(subset, adj_bits)
        for qm in Q3_MASKS:
            if (lm & qm) == qm:
                energy += 1

    best_energy = energy
    best_adj_bits = adj_bits[:]
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit - 10:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        is_present = (adj_bits[u] >> v) & 1
        
        delta_energy = 0
        subsets = SUBSETS_CONTAINING_PAIR.get((u, v), [])
        
        for subset in subsets:
            li = subset.index(u)
            lj = subset.index(v)
            k = edge_index(li, lj)
            lm = get_local_mask(subset, adj_bits)
            
            if is_present:
                for qm in Q3_MASKS_BY_EDGE[k]:
                    if (lm & qm) == qm:
                        delta_energy -= 1
            else:
                for qm in Q3_MASKS_BY_EDGE[k]:
                    if (lm & (qm ^ (1 << k))) == (qm ^ (1 << k)):
                        delta_energy += 1
        
        new_energy = energy + delta_energy
        
        if new_energy < energy or random.random() < pow(2.71828, -delta_energy / T):
            if is_present:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_adj_bits = adj_bits[:]
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    adj_bits = best_adj_bits
    
    while time.time() - start_time < time_limit - 1:
        found = False
        for subset in ALL_SUBSETS:
            lm = get_local_mask(subset, adj_bits)
            for qm in Q3_MASKS:
                if (lm & qm) == qm:
                    for k in range(28):
                        if (qm >> k) & 1:
                            for li in range(8):
                                for lj in range(li+1, 8):
                                    if edge_index(li, lj) == k:
                                        u = subset[li]
                                        v = subset[lj]
                                        adj_bits[u] &= ~(1 << v)
                                        adj_bits[v] &= ~(1 << u)
                                        found = True
                                        break
                                if found: break
                            if found: break
                    if found: break
            if found: break
        
        if not found:
            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
