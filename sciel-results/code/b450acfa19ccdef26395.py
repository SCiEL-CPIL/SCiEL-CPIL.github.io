import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
NUM_TRIPLES = len(ALL_TRIPLES)
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

def _precompute_conflicts_optimized():
    conflict_masks = [0] * NUM_TRIPLES
    
    # Precompute the 15 pairings of indices 0,1,2,3,4,5
    base_pairings = []
    def gen_pairings(vals):
        if not vals:
            yield []
            return
        first = vals[0]
        rest = vals[1:]
        for i in range(len(rest)):
            pair = (first, rest[i])
            new_rest = rest[:i] + rest[i+1:]
            for others in gen_pairings(new_rest):
                yield (pair,) + tuple(others)
                
    for p in gen_pairings([0,1,2,3,4,5]):
        base_pairings.append(p)
        
    for six_verts in combinations(range(V), 6):
        v = list(six_verts)
        for pairing in base_pairings:
            p1 = (v[pairing[0][0]], v[pairing[0][1]])
            p2 = (v[pairing[1][0]], v[pairing[1][1]])
            p3 = (v[pairing[2][0]], v[pairing[2][1]])
            
            edges = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        edges.append(triple_to_idx[t])
                        
            for i_idx in range(8):
                u = edges[i_idx]
                for j_idx in range(i_idx + 1, 8):
                    v_idx = edges[j_idx]
                    conflict_masks[u] |= (1 << v_idx)
                    conflict_masks[v_idx] |= (1 << u_idx)
                    
    return conflict_masks

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    conflict_masks = _precompute_conflicts_optimized()
    
    # Calculate conflict degrees for heuristic
    degrees = [bin(mask).count('1') for mask in conflict_masks]
    
    best_count = 0
    best_selected_mask = 0
    
    indices = list(range(NUM_TRIPLES))
    # Pre-sorted indices by degree (ascending)
    sorted_indices = sorted(indices, key=lambda x: degrees[x])
    
    while time.time() < deadline:
        # Alternate between random and sorted heuristic
        if random.random() < 0.5:
            order = indices[:]
            random.shuffle(order)
        else:
            # Add some noise to sorted order
            order = sorted_indices[:]
            # Swap a few random elements to break ties or explore neighbors
            for _ in range(10):
                i, j = random.sample(range(NUM_TRIPLES), 2)
                order[i], order[j] = order[j], order[i]
        
        selected_mask = 0
        available = (1 << NUM_TRIPLES) - 1
        count = 0
        
        for i in order:
            if (available >> i) & 1:
                count += 1
                selected_mask |= (1 << i)
                available &= ~conflict_masks[i]
                available &= ~(1 << i)
                
        if count > best_count:
            best_count = count
            best_selected_mask = selected_mask
            
    result = []
    for i in range(NUM_TRIPLES):
        if (best_selected_mask >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result
