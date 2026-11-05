import random
import time
import itertools

def _min_vertex_cover(triples):
    """Find minimum vertex cover of the hypergraph defined by triples."""
    n = 16
    if not triples:
        return 0
    t_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
    # Sort triples by number of vertices? They all have 3.
    # Maybe sort by frequency of vertices?
    # For now, keep as is.
    
    best_cover_size = n
    
    # Use a stack for iterative backtracking to avoid recursion limit? 
    # Depth is at most 16, so recursion is fine.
    
    def backtrack(idx, current_cover_mask, cover_size):
        nonlocal best_cover_size
        if cover_size >= best_cover_size:
            return
        if idx == len(t_masks):
            if cover_size < best_cover_size:
                best_cover_size = cover_size
            return
        
        # Pruning: 
        # If the number of remaining triples is 0, update best.
        # Simple lower bound: 
        # Each vertex can cover at most 3 triples (actually less as they share vertices).
        # Not a tight bound.
        
        t = t_masks[idx]
        
        # Find vertices in t that are not in current_cover_mask? 
        # No, we are building the cover. 
        # We must pick a vertex from t that is NOT in current_cover_mask?
        # If a vertex in t is already in current_cover_mask, then t is already covered.
        # So we should have skipped t in the loop.
        # Let's ensure we skip covered triples.
        
        # Check if t is already covered
        if (t & current_cover_mask) == t:
            backtrack(idx + 1, current_cover_mask, cover_size)
            return

        # Pick a vertex from t
        # Heuristic: pick the vertex in t that appears in the most other uncovered triples
        # This is expensive to compute. Let's just pick the first one.
        # Or randomize to escape local minima? 
        # For Min Cover, deterministic is fine for B&B.
        
        vertices_in_t = [i for i in range(n) if (t >> i) & 1]
        
        for v in vertices_in_t:
            if (current_cover_mask >> v) & 1:
                continue # Already covered, but t is not covered? Contradiction.
                # Wait, if v is in cover, and v is in t, then t is covered.
                # So this case shouldn't happen if we skip covered triples.
            
            new_cover_mask = current_cover_mask | (1 << v)
            
            # Find next uncovered triple
            next_idx = idx + 1
            while next_idx < len(t_masks) and (t_masks[next_idx] & new_cover_mask) == t_masks[next_idx]:
                next_idx += 1
            
            backtrack(next_idx, new_cover_mask, cover_size + 1)
            
            # If we found a solution of size cover_size + 1, 
            # and we are trying to minimize, we can stop if we found 1? 
            # No, we want the global min.
            
    backtrack(0, 0, 0)
    return best_cover_size

def _independence_number(triples):
    n = 16
    if not triples:
        return n
    return n - _min_vertex_cover(triples)

def _is_valid_addition(triples, new_triple):
    a, b, c = new_triple
    s_new = {a, b, c}
    for t in triples:
        overlap = len(s_new & set(t))
        if overlap >= 2:
            return False
    return True

def _greedy_construct():
    n = 16
    triples = []
    used_pairs = set()
    
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    for t in all_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        valid = True
        for p in pairs:
            if p in used_pairs or (p[1], p[0]) in used_pairs:
                valid = False
                break
        if valid:
            triples.append(t)
            for p in pairs:
                used_pairs.add(p)
    
    return triples

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 110
    
    best_triples = _greedy_construct()
    best_score = _independence_number(best_triples)
    current_triples = best_triples[:]
    current_score = best_score
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if current_triples:
            idx = random.randint(0, len(current_triples) - 1)
            removed = current_triples[idx]
            current_triples.pop(idx)
            
            new_triple = None
            attempts = 0
            while attempts < 50:
                t = tuple(sorted(random.sample(range(16), 3)))
                if _is_valid_addition(current_triples, t):
                    new_triple = list(t)
                    break
                attempts += 1
            
            if new_triple:
                current_triples.append(new_triple)
                new_score = _independence_number(current_triples)
                
                delta = new_score - current_score
                if delta <= 0 or (delta > 0 and random.random() < pow(2.71828, -delta / temp)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = current_triples[:]
                else:
                    current_triples.pop()
                    current_triples.insert(idx, removed)
            else:
                current_triples.insert(idx, removed)
        
        temp *= cooling
        if temp < 0.01:
            temp = 1.0
    
    return best_triples
