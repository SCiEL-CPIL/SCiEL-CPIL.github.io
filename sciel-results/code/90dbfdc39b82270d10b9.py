"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    # Generate all vectors with entries in {-1, 0, 1}
    # There are 3^11 = 177147 such vectors.
    # We will group them by their squared norm.
    # For a set of vectors with max norm squared M, the condition is u.v <= M/2.
    # Since u.v is integer, u.v <= floor(M/2).
    
    all_vectors = []
    for p in product([-1, 0, 1], repeat=D):
        v = np.array(p, dtype=np.int64)
        norm_sq = np.dot(v, v)
        if norm_sq > 0:
            all_vectors.append((norm_sq, v))
    
    # Group by norm squared
    from collections import defaultdict
    groups = defaultdict(list)
    for norm_sq, v in all_vectors:
        groups[norm_sq].append(v)
    
    best_set = []
    
    # Try each norm squared value
    for M in sorted(groups.keys()):
        candidates = groups[M]
        max_dot = M // 2
        
        # Greedy selection with random restarts
        # Since the pool is small (< 30k for any M), we can do a few restarts.
        # However, standard greedy is O(N^2 * K). N=30k is 900M ops per pass, too slow for many restarts.
        # We need a faster greedy or a smarter one.
        
        # Let's use a simpler greedy but with a fixed order first.
        # Then try a few random shuffles.
        
        def greedy_select(cands, limit):
            selected = []
            # Sort by norm? All have same norm.
            # Just iterate
            for v in cands:
                ok = True
                for w in selected:
                    if np.dot(v, w) > limit:
                        ok = False
                        break
                if ok:
                    selected.append(v)
            return selected
        
        # First pass: original order
        current_best = greedy_select(candidates, max_dot)
        if len(current_best) > len(best_set):
            best_set = current_best
            
        # A few random restarts to escape local optima of the greedy order
        # Limit restarts to keep time low. 10 restarts * 30k vectors * avg 100 selected * 11 dims
        # 10 * 30000 * 100 * 11 ~ 330M ops. Might be tight in Python.
        # Let's reduce candidate pool size or use numpy vectorization for dot products.
        
        # Optimized Greedy using NumPy
        def greedy_select_np(cands, limit):
            if not cands:
                return []
            cands_np = np.array(cands)
            n = len(cands_np)
            selected_idx = []
            selected_vecs = []
            
            for i in range(n):
                v = cands_np[i]
                if not selected_vecs:
                    selected_idx.append(i)
                    selected_vecs.append(v)
                    continue
                
                # Check dot product with all selected
                # selected_vecs is a list of vectors. Convert to array for speed?
                # Or just loop. 
                # To speed up, we can maintain an array of selected vectors.
                pass # This python loop is slow. Let's stick to the simple one but limit restarts.
        
        # Given time constraints (120s), and Python overhead, let's be conservative.
        # The previous program scored 1.89. We just need to beat that.
        # The pool of norm^2=4 has 5280 vectors.
        # The pool of norm^2=5 has 14784.
        # The pool of norm^2=6 has 29568.
        
        # Let's try M=4, 5, 6 with a few restarts.
        
        if M in [4, 5, 6]:
            for _ in range(5):
                shuffled = candidates.copy()
                np.random.shuffle(shuffled)
                current_best = greedy_select(shuffled, max_dot)
                if len(current_best) > len(best_set):
                    best_set = current_best

    if best_set:
        return np.array(best_set, dtype=np.int64)
    else:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
# EVOLVE-BLOCK-END
