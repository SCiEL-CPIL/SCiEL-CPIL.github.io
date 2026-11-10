import random
import time
import sys

def construct():
    deadline = time.time() + 110.0
    N = 16
    rng = random.Random(12345)
    
    # Precompute all triples
    all_triples = []
    mask_to_idx = {}
    idx_to_mask = {}
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                idx = len(all_triples)
                all_triples.append((mask, (a, b, c)))
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                
    full_mask = (1 << N) - 1

    # Precompute popcount
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute supersets for each triple
    superset_lists = []
    for mask, pts in all_triples:
        comp = full_mask ^ mask
        subs = []
        sub = comp
        while True:
            subs.append(mask | sub)
            if sub == 0:
                break
            sub = (sub - 1) & comp
        superset_lists.append(subs)

    def calc_alpha(edge_indices):
        """
        edge_indices: list of indices into all_triples
        """
        is_ind = [True] * (1 << N)
        for idx in edge_indices:
            for s in superset_lists[idx]:
                is_ind[s] = False
        
        max_alpha = 0
        for mask in range(1 << N):
            if is_ind[mask]:
                pc = popcount[mask]
                if pc > max_alpha:
                    max_alpha = pc
        return max_alpha

    def is_linear(edge_indices, new_idx):
        """
        Check if adding new_idx to edge_indices maintains linearity.
        """
        new_mask = idx_to_mask[new_idx]
        for idx in edge_indices:
            overlap = idx_to_mask[idx] & new_mask
            if bin(overlap).count('1') > 1:
                return False
        return True

    def greedy_construct():
        edges = []
        indices = list(range(len(all_triples)))
        rng.shuffle(indices)
        for idx in indices:
            if is_linear(edges, idx):
                edges.append(idx)
        return edges

    best_edges = None
    best_alpha = N + 1
    
    # Phase 1: Greedy
    for i in range(100):
        if time.time() > deadline:
            break
        edges = greedy_construct()
        alpha = calc_alpha(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            
    if best_edges is None:
        return []

    # Phase 2: Simulated Annealing
    current_edges = best_edges
    current_alpha = best_alpha
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    num_triples = len(all_triples)
    
    while time.time() < deadline:
        move_type = rng.random()
        new_edges = current_edges[:]
        
        if move_type < 0.8:
            if not current_edges: continue
            idx_to_remove = rng.randrange(len(current_edges))
            old_idx = current_edges[idx_to_remove]
            new_edges.pop(idx_to_remove)
            
            found = False
            for _ in range(100):
                cand_idx = rng.randrange(num_triples)
                if cand_idx == old_idx: continue
                if is_linear(new_edges, cand_idx):
                    new_edges.append(cand_idx)
                    found = True
                    break
            if not found: continue
        else:
            if len(current_edges) < 2: continue
            i1 = rng.randrange(len(current_edges))
            i2 = rng.randrange(len(current_edges))
            if i1 == i2: continue
            if i1 > i2: i1, i2 = i2, i1
            
            old1 = current_edges[i1]
            old2 = current_edges[i2]
            
            new_edges.pop(i1)
            new_edges.pop(i2)
            
            added = []
            for _ in range(200):
                if len(added) == 2: break
                cand_idx = rng.randrange(num_triples)
                if cand_idx in (old1, old2): continue
                if is_linear(new_edges + added, cand_idx):
                    added.append(cand_idx)
            if len(added) < 2: continue
            new_edges.extend(added)

        new_alpha = calc_alpha(new_edges)
        
        delta = new_alpha - current_alpha
        if delta < 0:
            current_edges = new_edges
            current_alpha = new_alpha
            best_edges = current_edges[:]
            best_alpha = current_alpha
        else:
            if delta == 0:
                prob = 0.5
            else:
                prob = pow(2.718281828, -delta / temperature)
            if rng.random() < prob:
                current_edges = new_edges
                current_alpha = new_alpha
        
        temperature *= cooling
        if temperature < min_temp:
            temperature = min_temp

    return [list(all_triples[idx][1]) for idx in best_edges]

