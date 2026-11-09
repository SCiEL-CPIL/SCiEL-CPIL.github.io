"""Fresh design: targeted local search to minimize independence number of linear triple systems on 16 points."""
import random
import time
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _point_masks():
    return [1 << i for i in range(16)]

def _triple_to_mask(t, pm):
    return pm[t[0]] | pm[t[1]] | pm[t[2]]

def _is_valid(triples):
    # any two triples share at most one point
    used_pairs = set()
    for a, b, c in triples:
        pairs = [(a, b), (a, c), (b, c)]
        for p in pairs:
            if p in used_pairs:
                return False
            used_pairs.add(p)
    return True

def _find_max_independent_set(triples, pm):
    """
    Returns (max_size, max_set_mask).
    Checks sizes from 16 down to 1.
    """
    if not triples:
        return 16, (1 << 16) - 1
    
    triple_masks = [_triple_to_mask(t, pm) for t in triples]
    
    # Check sizes from 16 down
    for size in range(16, 0, -1):
        for combo in combinations(range(16), size):
            mask = 0
            for p in combo:
                mask |= pm[p]
            # Check if independent
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                return size, mask
    return 0, 0

def _greedy_build(rng, pm):
    points = list(range(16))
    rng.shuffle(points)
    triples = []
    used_pairs = set()
    
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
    
    cand = all_triples[:]
    rng.shuffle(cand)
    
    usage = [0] * 16
    max_rounds = 400
    added = 0
    while added < max_rounds:
        valid_now = []
        for t in cand:
            a, b, c = t
            if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
                continue
            valid_now.append(t)
        
        if not valid_now:
            break
        
        # Pick one, biased toward covering underused points
        # Random pick for diversity
        t = valid_now[rng.randrange(len(valid_now))]
        
        a, b, c = t
        triples.append(t)
        used_pairs.add((a, b)); used_pairs.add((a, c)); used_pairs.add((b, c))
        usage[a] += 1; usage[b] += 1; usage[c] += 1
        added += 1
        
    return triples

def _local_search(triples, pm, rng, time_budget):
    """
    Targeted local search:
    1. Find max independent set S.
    2. Try to add a triple T subset of S.
    3. If can't add, try to swap a triple with T subset of S.
    """
    cur = list(triples)
    cur_alpha, cur_set_mask = _find_max_independent_set(cur, pm)
    best_alpha = cur_alpha
    best_triples = cur
    
    start = time.time()
    no_improve_count = 0
    
    while (time.time() - start) < time_budget:
        if no_improve_count > 50:
            break
            
        # Find max independent set in current system
        cur_alpha, cur_set_mask = _find_max_independent_set(cur, pm)
        
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_triples = cur
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Generate candidates T subset of cur_set_mask
        # Points in cur_set_mask
        pts_in_set = []
        for i in range(16):
            if (cur_set_mask >> i) & 1:
                pts_in_set.append(i)
                
        if len(pts_in_set) < 3:
            # Can't form a triple from the independent set? 
            # This shouldn't happen if cur_alpha >= 3.
            # If cur_alpha < 3, we are done? No, we want to minimize.
            # If alpha is 2, we can't do better (min alpha for linear triple system on 16 pts is likely > 2).
            # Actually, if alpha is 2, it means every pair is in a triple? No, every triple is avoided.
            # Anyway, if we can't pick 3 points, we can't add a triple from S.
            pass
        else:
            # Try to add a triple T from pts_in_set
            # We need T to be valid (no pair conflict with existing triples)
            # And T must not be already in cur.
            
            # Precompute used pairs for cur
            used_pairs = set()
            for a, b, c in cur:
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
            cur_set = set(cur)
            
            # Candidate triples from pts_in_set
            cands = []
            for i in range(len(pts_in_set)):
                for j in range(i+1, len(pts_in_set)):
                    for k in range(j+1, len(pts_in_set)):
                        t = (pts_in_set[i], pts_in_set[j], pts_in_set[k])
                        if t not in cur_set:
                            cands.append(t)
                            
            rng.shuffle(cands)
            
            improved = False
            for t in cands:
                a, b, c = t
                # Check if T can be added
                if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
                    continue
                    
                # Try adding T
                new_triples = cur + [t]
                # Check validity (should be ok by construction, but double check)
                # Actually, we just checked pairs.
                
                # Compute new alpha
                new_alpha, _ = _find_max_independent_set(new_triples, pm)
                
                if new_alpha < cur_alpha:
                    cur = new_triples
                    cur_alpha = new_alpha
                    improved = True
                    break
                elif new_alpha == cur_alpha:
                    # Accept to explore?
                    # Let's accept with some probability to escape local minima
                    if rng.random() < 0.1:
                        cur = new_triples
                        # cur_alpha stays same
                        improved = True
                        break
                        
            if not improved:
                # Try swaps
                # Pick a random triple to remove
                if cur:
                    idx = rng.randrange(len(cur))
                    t_out = cur[idx]
                    cur_without = cur[:idx] + cur[idx+1:]
                    
                    # Rebuild used pairs for cur_without
                    used_pairs_new = set()
                    for a, b, c in cur_without:
                        used_pairs_new.add((a, b))
                        used_pairs_new.add((a, c))
                        used_pairs_new.add((b, c))
                    
                    for t in cands:
                        a, b, c = t
                        if (a, b) in used_pairs_new or (a, c) in used_pairs_new or (b, c) in used_pairs_new:
                            continue
                            
                        new_triples = cur_without + [t]
                        new_alpha, _ = _find_max_independent_set(new_triples, pm)
                        
                        if new_alpha < cur_alpha:
                            cur = new_triples
                            cur_alpha = new_alpha
                            improved = True
                            break
                        elif new_alpha == cur_alpha and rng.random() < 0.05:
                            cur = new_triples
                            improved = True
                            break

    return best_triples, best_alpha

def construct():
    rng = random.Random(12345)
    pm = _point_masks()
    
    best_alpha = 16
    best_triples = None
    start = time.time()
    
    restarts = 0
    while (time.time() - start) < (_LIMIT - 20):
        triples = _greedy_build(rng, pm)
        if not triples:
            restarts += 1
            continue
            
        # Post-optimization
        rem_budget = min(10.0, max(0.5, _LIMIT - 25 - (time.time() - start)))
        if rem_budget < 0.5:
            a, _ = _find_max_independent_set(triples, pm)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
            restarts += 1
            continue
            
        opt_triples, a = _local_search(triples, pm, rng, rem_budget)
        if a < best_alpha:
            best_alpha = a
            best_triples = opt_triples
        restarts += 1
        if best_alpha <= 6:
            break
            
    if best_triples is None:
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    if not _is_valid(best_triples):
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    return best_triples
