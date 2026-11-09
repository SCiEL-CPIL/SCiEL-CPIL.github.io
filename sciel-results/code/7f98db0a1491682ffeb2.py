import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def check_2_colorable(n, edges):
    """
    Returns True if 2-colorable, False otherwise.
    Uses bitmask-based backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    m = len(edge_masks)
    
    # State:
    # c1: vertices colored 1
    # c0: vertices colored 0
    # un: vertices uncolored
    # We maintain invariants: c1 | c0 | un == (1 << n) - 1
    # c1 & c0 == 0
    
    all_mask = (1 << n) - 1
    c1 = 0
    c0 = 0
    un = all_mask
    
    # Precompute for each edge:
    # e_mask
    # We need to check counts quickly.
    
    def propagate(c1, c0, un):
        """
        Returns (success, c1, c0, un)
        """
        changed = True
        while changed:
            changed = False
            for em in edge_masks:
                # Check for contradiction
                # If all 5 are in c1 or all 5 in c0
                if (em & ~c1) == 0: # em subset c1
                    return False, c1, c0, un
                if (em & ~c0) == 0: # em subset c0
                    return False, c1, c0, un
                
                # Count colors in em
                # c1_count = popcount(em & c1)
                # c0_count = popcount(em & c0)
                # un_count = 5 - c1_count - c0_count
                
                # We can compute un_in_em = em & un
                un_in_em = em & un
                c1_in_em = em & c1
                c0_in_em = em & c0
                
                # If un_in_em is 0, we already checked contradiction.
                # If un_in_em has 1 bit, it's a unit clause.
                
                # Fast check for unit propagation:
                # If (em & c1) has 4 bits, and un_in_em has 1 bit -> that bit must be 0
                # If (em & c0) has 4 bits, and un_in_em has 1 bit -> that bit must be 1
                
                # popcount is fast in Python 3.10+? int.bit_count()
                c1_cnt = c1_in_em.bit_count()
                c0_cnt = c0_in_em.bit_count()
                un_cnt = 5 - c1_cnt - c0_cnt
                
                if un_cnt == 1:
                    # The single unassigned vertex in this edge
                    v_mask = un_in_em
                    if c1_cnt == 4:
                        # Must be 0
                        # Assign v_mask to 0
                        if v_mask & c1: return False, c1, c0, un # Should not happen
                        c0 |= v_mask
                        un &= ~v_mask
                        changed = True
                    elif c0_cnt == 4:
                        # Must be 1
                        c1 |= v_mask
                        un &= ~v_mask
                        changed = True
                    # else: 3-1-1 or 2-2-1, no propagation
                # If un_cnt == 0, we already checked contradiction.
                
        return True, c1, c0, un

    def select_variable(un):
        if un == 0:
            return -1
        # Heuristic: pick vertex in most constrained edges
        # For simplicity, pick lowest index or one with high degree in unassigned edges
        # Let's try: count how many edges have this vertex and are "tight"
        # Or just random? Deterministic is better for reproducibility, but random might escape local minima?
        # Backtracking is deterministic if variable order is fixed.
        # Let's use a simple heuristic: vertex appearing in most edges with unassigned vertices.
        
        scores = [0] * n
        for em in edge_masks:
            if em & un:
                # Count unassigned vertices in this edge
                cnt = (em & un).bit_count()
                if cnt <= 2: # Tight edges
                    # Add score to unassigned vertices in this edge
                    um = em & un
                    while um:
                        lsb = um & -um
                        v = lsb.bit_length() - 1
                        scores[v] += 1
                        um ^= lsb
        
        best_v = -1
        max_score = -1
        # Pick highest score, break ties by index
        for v in range(n):
            if (un >> v) & 1:
                if scores[v] > max_score:
                    max_score = scores[v]
                    best_v = v
        return best_v

    def backtrack(c1, c0, un):
        v = select_variable(un)
        if v == -1:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught it, but verify.
            for em in edge_masks:
                if (em & ~c1) == 0 or (em & ~c0) == 0:
                    return False
            return True
        
        # Try 0
        c1_new, c0_new, un_new = c1, c0 | (1 << v), un & ~(1 << v)
        ok, c1_new, c0_new, un_new = propagate(c1_new, c0_new, un_new)
        if ok and backtrack(c1_new, c0_new, un_new):
            return True
        
        # Try 1
        c1_new, c0_new, un_new = c1 | (1 << v), c0, un & ~(1 << v)
        ok, c1_new, c0_new, un_new = propagate(c1_new, c0_new, un_new)
        if ok and backtrack(c1_new, c0_new, un_new):
            return True
        
        return False

    # Symmetry breaking: Fix vertex 0 to 0
    c0 = 1
    un = all_mask & ~1
    ok, c1, c0, un = propagate(0, c0, un)
    if not ok:
        return False # Not 2-colorable
    
    return backtrack(c1, c0, un)

def minimize_edges(n, edges):
    current_edges = edges[:]
    improved = True
    while improved and time_left() > 5:
        improved = False
        idxs = list(range(len(current_edges)))
        random.shuffle(idxs)
        for i in idxs:
            if time_left() < 5:
                break
            test_edges = current_edges[:i] + current_edges[i+1:]
            if not check_2_colorable(n, test_edges):
                current_edges = test_edges
                improved = True
                break
    return current_edges

def search_small_hypergraph():
    best_edges = None
    best_n = None
    
    for n in [10, 11, 12, 13]:
        if time_left() < 10:
            break
        
        for trial in range(200):
            if time_left() < 5:
                break
            
            # Start with a dense random hypergraph
            # Probability of non-2-colorable increases with density
            # For n=10, C(10,5)=252. 
            # Try 30-60 edges.
            m = random.randint(30, 60)
            
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            edges = [list(e) for e in all_possible[:m]]
            
            if not check_2_colorable(n, edges):
                minimized = minimize_edges(n, edges)
                if best_edges is None or len(minimized) < len(best_edges):
                    best_edges = minimized
                    best_n = n
                    if len(minimized) <= 30:
                        return {'n': best_n, 'edges': best_edges}
                break # Found one, minimize, then try next n or trial?
                # Actually, if we find one, we minimize it. 
                # We should probably try multiple trials for the same n to see if we can get smaller.
                # But minimize_edges is expensive.
                # Let's just do one good trial per n to start.
                
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

def construct():
    try:
        result = search_small_hypergraph()
        return result
    except Exception:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
