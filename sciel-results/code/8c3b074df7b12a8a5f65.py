import itertools
import random
import time
import numpy as np

N = 16
TOTAL_POINTS = N * N

def _get_point_index(x, y):
    return x * N + y

def _get_coords(idx):
    return (idx // N, idx % N)

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Precomputation
    # cover_mask[a][b] is an integer bitmask of points c such that {a, b, c} is isosceles
    # a, b are indices 0..255
    cover_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute all isosceles triples to fill cover_mask
    # For efficiency, we can iterate over pairs and find third points
    # But 256^2 * 256 is 16M, which is fine.
    # Actually, we can just compute on the fly or precompute a set of forbidden triples.
    # Let's precompute cover_mask directly.
    
    for a in range(TOTAL_POINTS):
        for b in range(a + 1, TOTAL_POINTS):
            pa = points[a]
            pb = points[b]
            dab = _dist2(pa, pb)
            
            mask = 0
            for c in range(TOTAL_POINTS):
                if c == a or c == b:
                    continue
                pc = points[c]
                dac = _dist2(pa, pc)
                dbc = _dist2(pb, pc)
                
                # Isosceles if any two distances are equal
                if dab == dac or dab == dbc or dac == dbc:
                    mask |= (1 << c)
            
            cover_mask[a][b] = mask
            cover_mask[b][a] = mask
            
    best_set = None
    best_size = TOTAL_POINTS + 1
    
    all_points_mask = (1 << TOTAL_POINTS) - 1
    
    def _is_maximal(S):
        """Check if S is a maximal independent set using bitmask cover."""
        if not S:
            return False # Empty set is not maximal unless grid is empty
        
        S_set = set(S)
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
            
        # Check independence: No pair in S should cover any point in S?
        # Actually, independence means no triple in S is isosceles.
        # This is guaranteed if we construct it that way, but let's be safe.
        # If we use the cover_mask, a point c is covered by pair {a,b} if {a,b,c} is isosceles.
        # So if c in S, and c is covered by {a,b} in S, then {a,b,c} is isosceles, so S is not independent.
        
        # Check Maximality:
        # Union of covers of all pairs in S must cover all points NOT in S.
        total_cover = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                total_cover |= cover_mask[a][b]
                
        # Points not in S
        not_in_S_mask = all_points_mask & ~S_mask
        
        # If all points not in S are covered, it's maximal
        return (total_cover & not_in_S_mask) == not_in_S_mask

    def _greedy_construct():
        """Construct a maximal independent set using greedy random choice."""
        S = []
        available = set(range(TOTAL_POINTS))
        
        while available:
            # Pick a random point
            p = random.choice(list(available))
            
            # Check if p is compatible with current S
            # p is compatible if for all a, b in S, p is NOT in cover_mask[a][b]
            compatible = True
            S_set = set(S)
            for a in S:
                for b in S_set:
                    if a < b and (1 << p) & cover_mask[a][b]:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                S.append(p)
                # Remove from available all points that are incompatible with S + {p}
                # A point q is incompatible if it forms an isosceles triangle with any pair in S+{p}
                # Or simpler: just filter available by checking compatibility with new S
                new_available = set()
                for q in available:
                    if q == p:
                        continue
                    # Check q against S + {p}
                    ok = True
                    S_new = S # S already contains p
                    for a in S_new:
                        for b in S_new:
                            if a < b and (1 << q) & cover_mask[a][b]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_available.add(q)
                available = new_available
            else:
                available.remove(p)
                
        return S

    def _local_search(S):
        """Try to minimize S while maintaining maximality."""
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            # Try removing each point
            for p in list(S):
                S_candidate = S - {p}
                if _is_maximal(S_candidate):
                    S = S_candidate
                    improved = True
                    break
        return S

    # Run multiple trials
    trials = 0
    while time.time() < deadline:
        trials += 1
        S = _greedy_construct()
        S = _local_search(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            # If we find a very small set, we might want to stop or continue to find even smaller
            if best_size <= 10:
                # Continue a bit more to see if we can go lower, but time is limited
                pass

    if best_set is None:
        # Fallback: Return a valid maximal set, even if not minimal
        S = _greedy_construct()
        best_set = S

    # Convert indices back to (x, y) tuples
    return [_get_coords(p) for p in best_set]
