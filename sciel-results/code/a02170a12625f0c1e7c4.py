import random
import time
import sys

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their bitmasks and pair sets
    all_triples = []
    triple_data = {} # mask -> (mask, p1, p2, p3, pts)
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                data = (mask, p_ab, p_ac, p_bc, (a,b,c))
                all_triples.append(data)
                triple_data[mask] = data
                
    # Map pairs to list of triples containing them
    pair_to_triples = {}
    for data in all_triples:
        _, p1, p2, p3, _ = data
        for p in (p1, p2, p3):
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(data[0]) # store mask

    # Exact independence number calculation using branch-and-bound
    # on the shadow graph (vertices are points, edges connect pairs in same triple)
    def calc_alpha(masks):
        # Build shadow graph: adjacency bitmask for each point
        adj = [0] * n
        for m in masks:
            # Extract the 3 points from mask
            pts = []
            for i in range(n):
                if m & (1 << i):
                    pts.append(i)
            # Add edges between all pairs in this triple
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    adj[pts[i]] |= (1 << pts[j])
                    adj[pts[j]] |= (1 << pts[i])
        
        best_size = 0
        
        def search(candidates, current_size):
            nonlocal best_size
            
            if current_size + bin(candidates).count('1') <= best_size:
                return
            
            if candidates == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick a vertex with maximum degree in the candidate subgraph
            max_deg = -1
            v = -1
            c = candidates
            while c:
                lb = c & (-c)
                idx = (lb.bit_length() - 1)
                deg = bin(adj[idx] & candidates).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                c -= lb
            
            if v == -1:
                return
            
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            search(candidates & ~v_bit, current_size)
            
            # Branch 2: Include v
            new_cands = (candidates & ~v_bit) & ~adj[v]
            search(new_cands, current_size + 1)
        
        search((1 << n) - 1, 0)
        return best_size

    def generate_initial_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for data in all_triples:
            mask, p1, p2, p3, pts = data
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(data)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples, masks, len(triples)

    def local_search(triples, masks, current_alpha):
        # triples is a list of data tuples
        # masks is a list of masks
        # We will swap triples to reduce alpha
        
        used_pairs = set()
        for data in triples:
            _, p1, p2, p3, _ = data
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        best_alpha = current_alpha
        best_triples = triples
        best_masks = masks
        
        # Precompute adjacency of pairs? 
        # For a swap, we remove T_old, add T_new.
        # T_new must be valid: its pairs must not be in (used_pairs - pairs(T_old)).
        
        for _ in range(1000): # Limit iterations per restart
            if time.time() - start_time > timeout:
                break
            
            # Pick a random triple to remove
            idx = random.randint(0, len(triples)-1)
            old_data = triples[idx]
            old_mask = masks[idx]
            _, op1, op2, op3, _ = old_data
            
            # Remove from tracking
            triples.remove(old_data)
            masks.remove(old_mask)
            used_pairs.discard(op1)
            used_pairs.discard(op2)
            used_pairs.discard(op3)
            
            # Find candidates
            # A candidate is valid if none of its pairs are in used_pairs
            candidates = []
            for data in all_triples:
                mask, p1, p2, p3, pts = data
                if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                    continue
                # Also ensure we don't just re-add the same triple if it's the only option?
                # It's okay to re-add, but it won't change alpha.
                candidates.append(data)
                
            if not candidates:
                # Restore
                triples.append(old_data)
                masks.append(old_mask)
                used_pairs.add(op1)
                used_pairs.add(op2)
                used_pairs.add(op3)
                continue
                
            # Evaluate candidates
            # To save time, we can sample if candidates list is huge
            # But 560 is small.
            # We want to minimize alpha.
            
            best_cand = None
            best_cand_alpha = best_alpha # We only accept if strictly better or equal? 
            # Actually, if equal, we might want to accept to escape local minima? 
            # But for now, let's stick to hill climbing (strict improvement) to be safe, 
            # or random acceptance if equal.
            
            # Calculate alpha for current state (without old_data)
            # Wait, we removed old_data. 
            # We need to calculate alpha for (current_masks + [new_mask]).
            # Current masks are `masks`.
            
            # Optimization: If len(masks) < 37, we might want to prioritize filling?
            # But the task says "Linear triple systems...". It doesn't strictly require max size?
            # "construct() -> list of triples... in which any two triples share at most one point."
            # "The score is the largest point set containing no triple".
            # Usually, larger packings have smaller independence numbers?
            # Not necessarily. But 37 is the max.
            # Let's assume we want to maintain size 37.
            
            # To speed up, we can check if the new triple adds edges that "break" independent sets.
            # But calc_alpha is fast.
            
            # Sample candidates if too many
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
                
            for cand in candidates:
                cand_mask = cand[0]
                # Temporarily add
                masks.append(cand_mask)
                alpha = calc_alpha(masks)
                masks.pop()
                
                if alpha < best_cand_alpha:
                    best_cand_alpha = alpha
                    best_cand = cand
                elif alpha == best_cand_alpha and random.random() < 0.1:
                    # Accept equal with small prob to wander
                    best_cand = cand
                    best_cand_alpha = alpha
                    
            if best_cand:
                # Apply swap
                triples.append(best_cand)
                masks.append(best_cand[0])
                _, np1, np2, np3, _ = best_cand
                used_pairs.add(np1)
                used_pairs.add(np2)
                used_pairs.add(np3)
                
                if best_cand_alpha < best_alpha:
                    best_alpha = best_cand_alpha
                    best_triples = triples[:]
                    best_masks = masks[:]
                    
                    if best_alpha <= 4:
                        return best_triples, best_masks, best_alpha
            else:
                # No improvement, restore old triple
                triples.append(old_data)
                masks.append(old_mask)
                used_pairs.add(op1)
                used_pairs.add(op2)
                used_pairs.add(op3)
                
        return best_triples, best_masks, best_alpha

    best_alpha = n + 1
    best_triples = []
    
    # Run multiple restarts
    while time.time() - start_time < timeout:
        triples, masks, size = generate_initial_packing()
        
        if size < 37:
            # Try to fill? Or just use it?
            # Max size is 37. If we can't get 37, alpha might be higher.
            # Let's try to fill with a greedy pass again?
            # Or just skip this restart if it's too small?
            # Random greedy on 560 triples usually gets 37.
            if size < 35:
                continue

        current_alpha = calc_alpha(masks)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples
            
            if best_alpha <= 4:
                break
        
        # Local search
        if time.time() - start_time > timeout - 10:
            break
            
        triples, masks, alpha_ls = local_search(triples, masks, current_alpha)
        
        if alpha_ls < best_alpha:
            best_alpha = alpha_ls
            best_triples = triples
            
            if best_alpha <= 4:
                break

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    # Return only the triples [a, b, c]
    return [t[4] for t in best_triples]
