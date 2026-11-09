import itertools
import random
import time
import math

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute triples
triples = []
triples_by_point = [[] for _ in range(N_POINTS)]
# Precompute bitmask for each triple
triple_masks = []
# Precompute bitmask for "other two" for each point in each triple
# cover_mask[q][t_idx] = mask of the other two points

dist_maps = [{} for _ in range(N_POINTS)]
for v_idx, v in enumerate(POINTS):
    for u_idx, u in enumerate(POINTS):
        if v_idx == u_idx: continue
        d = _dist2(v, u)
        if d not in dist_maps[v_idx]:
            dist_maps[v_idx][d] = []
        dist_maps[v_idx][d].append(u_idx)

seen_triples = set()
for v_idx in range(N_POINTS):
    for d, pts in dist_maps[v_idx].items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                a = pts[i]
                b = pts[j]
                t = tuple(sorted((v_idx, a, b)))
                if t not in seen_triples:
                    seen_triples.add(t)
                    idx = len(triples)
                    triples.append(t)
                    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                    triple_masks.append(mask)
                    
                    triples_by_point[t[0]].append(idx)
                    triples_by_point[t[1]].append(idx)
                    triples_by_point[t[2]].append(idx)

# Precompute cover masks for each point
# cover_masks[q] is a list of masks for the "other two" points in each triple containing q
cover_masks = [[] for _ in range(N_POINTS)]
for t_idx, t in enumerate(triples):
    for q in t:
        # Find the other two
        others = [x for x in t if x != q]
        mask = (1 << others[0]) | (1 << others[1])
        # We need to associate this mask with the specific triple index?
        # No, for maximality, we just need to know if ANY such mask is subset of S.
        # So for each q, we store a list of masks.
        # But we need to update this list when S changes? No, the masks are static.
        # We just check if (S_mask & cover_mask) == cover_mask.
        
        # Wait, we need to add this mask to cover_masks[q]
        # But we must ensure we add it only once per triple per q.
        # The loop `for q in t` does this.
        
        # However, we need to align indices if we want to do incremental?
        # For full re-eval, we just iterate.
        
        # Let's just store the masks in a flat list per point.
        # But we need to know which mask corresponds to which triple?
        # No, for the check "is q covered", we just need to know if there EXISTS a triple.
        # So a list of masks is sufficient.
        
        # But we must be careful: if a triple is isosceles at multiple vertices, it appears multiple times?
        # No, `seen_triples` ensures unique triples.
        # So each triple is added once.
        # For each q in t, we add the mask of the other two to cover_masks[q].
        
        # We need to add this to the list.
        # But we are in the generation loop.
        pass

# Re-do cover_masks generation properly
cover_masks = [[] for _ in range(N_POINTS)]
for t_idx, t in enumerate(triples):
    for q in t:
        others = [x for x in t if x != q]
        mask = (1 << others[0]) | (1 << others[1])
        cover_masks[q].append(mask)

def is_maximal(S_mask):
    """
    Returns number of points not in S_mask that are not covered.
    """
    nonmax = 0
    for q in range(N_POINTS):
        # If q is in S, skip
        if (S_mask >> q) & 1:
            continue
        # Check if covered
        covered = False
        for mask in cover_masks[q]:
            if (S_mask & mask) == mask:
                covered = True
                break
        if not covered:
            nonmax += 1
    return nonmax

def count_iso(S_mask):
    count = 0
    for mask in triple_masks:
        if (S_mask & mask) == mask:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Greedy Init
    chosen = []
    for p in POINTS:
        idx = POINT_INDEX[p]
        valid = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if _dist2(p, POINTS[chosen[i]]) == _dist2(p, POINTS[chosen[j]]) or \
                   _dist2(p, POINTS[chosen[i]]) == _dist2(POINTS[chosen[i]], POINTS[chosen[j]]) or \
                   _dist2(p, POINTS[chosen[j]]) == _dist2(POINTS[chosen[i]], POINTS[chosen[j]]):
                    valid = False
                    break
            if not valid: break
        if valid:
            chosen.append(idx)
            
    current_mask = 0
    for idx in chosen:
        current_mask |= (1 << idx)
        
    current_iso = count_iso(current_mask)
    current_nonmax = is_maximal(current_mask)
    
    best_mask = current_mask
    best_size = len(chosen)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        p_idx = random.randint(0, N_POINTS - 1)
        
        new_mask = current_mask
        if (new_mask >> p_idx) & 1:
            new_mask ^= (1 << p_idx) # Remove
        else:
            new_mask |= (1 << p_idx) # Add
            
        new_iso = count_iso(new_mask)
        new_nonmax = is_maximal(new_mask)
        
        new_cost = bin(new_mask).count('1') + 1000 * new_iso + 100 * new_nonmax
        curr_cost = bin(current_mask).count('1') + 1000 * current_iso + 100 * current_nonmax
        
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_mask = new_mask
            current_iso = new_iso
            current_nonmax = new_nonmax
            
            if current_iso == 0 and current_nonmax == 0:
                size = bin(current_mask).count('1')
                if size < best_size:
                    best_size = size
                    best_mask = current_mask
                    
        T *= cooling
        
    # Return best valid set
    # best_mask is guaranteed to be valid (iso=0, nonmax=0) because we only update best_mask if valid.
    # And initial greedy is valid.
    
    res = []
    for i in range(N_POINTS):
        if (best_mask >> i) & 1:
            res.append(POINTS[i])
    return res
