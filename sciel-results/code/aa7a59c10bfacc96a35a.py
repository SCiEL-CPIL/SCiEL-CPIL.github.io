import time
import random
from itertools import combinations
import numpy as np

# Check for scipy.optimize.milp availability
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def _edge_masks(n):
    """Returns list of edges and list of masks (ints) for each edge."""
    edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
    rest_all = list(range(n))
    masks = []
    for e in edges:
        vs = list(e)
        vs_set = set(vs)
        rest = [v for v in rest_all if v not in vs_set]
        base1 = 0
        for v in vs:
            base1 |= (1 << v)
        m = 0
        r = len(rest)
        for p in range(1 << r):
            idx0 = 0
            idx1 = base1
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx0 |= (1 << v)
                    idx1 |= (1 << v)
            m |= (1 << idx0) | (1 << idx1)
        masks.append(m)
    return edges, masks

def _solve_set_cover_ilp(masks, n_colorings, deadline):
    """Solve the set cover problem using ILP if available."""
    if not HAS_MILP:
        return None
    
    num_edges = len(masks)
    # A is (n_colorings, num_edges)
    # A[i, j] = 1 if mask j has bit i set
    # We need A @ x >= 1
    
    # Construct A efficiently
    # masks are integers. We can extract bits.
    # n_colorings is 2^n.
    
    # For n=11, n_colorings=2048. num_edges=462.
    # A is 2048x462.
    
    # Using numpy for speed
    # masks are Python ints.
    # We can create a numpy array of bools.
    
    # This loop might be slow in Python if done naively.
    # But 2048 * 462 = ~1M operations.
    
    A = np.zeros((n_colorings, num_edges), dtype=np.int8)
    
    # Optimization: iterate over edges, then set bits
    for j, m in enumerate(masks):
        # m is an int with bits set at positions 0..n_colorings-1
        # We need to find which bits are set.
        # m.bit_length() <= n_colorings
        # We can use a lookup or loop.
        # Since n_colorings is small (2048), we can just loop over bits?
        # Or use numpy bit operations?
        
        # Simple loop over set bits
        # m is sparse?
        # Each edge kills 2 * 2^(n-5) colorings.
        # For n=11, 2 * 2^6 = 128 colorings per edge.
        # So each column of A has 128 ones.
        
        # Extract set bits
        mm = m
        while mm:
            # Get index of lowest set bit
            # (mm & -mm).bit_length() - 1
            idx = (mm & -mm).bit_length() - 1
            A[idx, j] = 1
            mm &= mm - 1
            
    # Objective: minimize sum(x)
    c = np.ones(num_edges)
    
    # Constraints: A x >= 1
    # scipy milp requires: A @ x >= b
    # LinearConstraint(A, lb=np.ones(n_colorings), ub=np.inf)
    
    # Bounds: x in [0, 1]
    # integrality: all 1
    
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A, lb=np.ones(n_colorings))],
            integrality=np.ones(num_edges),
            bounds=Bounds(lb=0, ub=1)
        )
        if res.status == 0: # Optimal
            x = res.x
            # x should be 0 or 1
            chosen_indices = [i for i in range(num_edges) if x[i] > 0.5]
            return chosen_indices
    except Exception:
        pass
        
    return None

def _greedy_cover(n, masks, full, deadline, rng):
    """Greedy set cover."""
    uncovered = full
    chosen = []
    remaining = list(range(len(masks)))
    while uncovered:
        best = -1
        bestc = 0
        for ei in remaining:
            c = (uncovered & masks[ei]).bit_count()
            if c > bestc:
                bestc = c
                best = ei
                if bestc == uncovered.bit_count():
                    break
        if best < 0:
            break
        chosen.append(best)
        uncovered &= ~masks[best]
        remaining.remove(best)
        if time.time() > deadline:
            break
    if uncovered == 0:
        return chosen
    return None

def _local_prune(chosen, masks, full, deadline, rng):
    """Remove redundant edges."""
    improved = True
    chosen_set = set(chosen)
    while improved and time.time() < deadline:
        improved = False
        order = list(chosen_set)
        rng.shuffle(order)
        for i in order:
            if i not in chosen_set: continue
            # Check if removing i still covers
            u = 0
            for j in chosen_set:
                if j != i:
                    u |= masks[j]
            if u == full:
                chosen_set.remove(i)
                improved = True
                break
    return list(chosen_set)

def _try_n(n, deadline, rng):
    edges, masks = _edge_masks(n)
    full = (1 << (1 << n)) - 1
    n_colorings = 1 << n
    
    best = None
    
    # Try ILP first
    ilp_res = _solve_set_cover_ilp(masks, n_colorings, deadline)
    if ilp_res is not None:
        # Verify ILP solution
        u = 0
        for i in ilp_res:
            u |= masks[i]
        if u == full:
            best = ilp_res
    
    # Try Greedy if ILP failed or to see if we can do better (unlikely)
    # Actually, if ILP succeeded, it's optimal. No need for greedy.
    if best is None:
        for _ in range(4):
            if time.time() > deadline:
                break
            chosen = _greedy_cover(n, masks, full, deadline, rng)
            if chosen is None:
                continue
            chosen = _local_prune(chosen, masks, full, deadline, rng)
            if best is None or len(chosen) < len(best):
                best = chosen
                
    if best is None:
        return None
        
    # Final verification
    u = 0
    for i in best:
        u |= masks[i]
    if u != full:
        return None
        
    return {'n': n, 'edges': [list(edges[i]) for i in best]}

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)

    fallback = {'n': 9,
                'edges': [list(e) for e in combinations(range(9), 5)]}

    best = fallback
    # Prioritize n=11 where the optimal solution is expected
    for n in (11, 10, 12, 9):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res

    return best

# Keep the checker for local testing
def _is_2_colorable(n, edges, deadline):
    # ... (same as before) ...
    pass
