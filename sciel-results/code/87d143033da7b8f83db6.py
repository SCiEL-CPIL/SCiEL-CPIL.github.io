"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _idx(x, y):
    return x * N + y

def _pt(i):
    return (i // N, i % N)

def _d2_idx(i, j):
    pi, pj = _pt(i), _pt(j)
    return (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2

def _build_conflict_graph():
    """Build conflict graph: edge (i,j) means i and j cannot both be in set
    because there exists some k such that {i,j,k} forms isosceles triangle."""
    n_pts = N * N
    # For each pair (i,j), check if there exists k such that i,j,k form isosceles
    # Actually, we need: i and j conflict if adding both would create an isosceles
    # with any existing point. But for the independent set formulation:
    # Two points i,j are incompatible if there exists a third point k such that
    # {i,j,k} is isosceles. Then in any valid set, we can't have all three.
    # But that's a 3-clique constraint, not pairwise.
    # 
    # Better approach: For local search, we track which points can be added.
    # A point p can be added to set S if for all pairs (a,b) in S, {p,a,b} is not isosceles.
    # Equivalently, p conflicts with pair (a,b) if {p,a,b} is isosceles.
    #
    # For efficiency, precompute for each point p, the set of pairs (a,b) that
    # would form isosceles with p. Then p can be added iff no such pair is in S.
    
    # Precompute: for each point p, list of (a,b) pairs where {p,a,b} is isosceles
    # This is O(N^6) which is too slow for N=16 (16^6 = 16M, actually manageable)
    # Let's use a different approach: for each point p, compute which points q
    # are "incompatible" in the sense that there exists r such that {p,q,r} is isosceles.
    # But that's not quite right either.
    
    # Actually, for the local search approach, let's precompute:
    # conflict[p] = set of points q such that there exists r where {p,q,r} is isosceles
    # This is not sufficient because the constraint is on triples, not pairs.
    
    # Let me think differently. For a set S, p can be added iff:
    # For all a,b in S with a != b, {p,a,b} is NOT isosceles.
    # 
    # Precompute for each p, a bitmask of "forbidden pairs" or better:
    # For each p, precompute the set of points q such that p and q have a common
    # "isosceles partner" r. I.e., exists r: d(p,r)=d(q,r) or d(p,r)=d(p,q) or d(q,r)=d(p,q).
    # 
    # Actually, let's just precompute for each pair (p,q), whether there exists r
    # such that {p,q,r} is isosceles. If yes, then p and q are "conflicting" in the
    # sense that they cannot both be in S along with any r. But they CAN both be in S
    # as long as no r in S completes the isosceles.
    
    # For local search, the key operation is: given set S, can we add p?
    # Answer: for all a,b in S, {p,a,b} is not isosceles.
    # 
    # Precompute: for each p, and for each a, the set of b such that {p,a,b} is isosceles.
    # Then p can be added iff for all a in S, no b in S is in iso[p][a].
    
    # Let's precompute iso[p][a] = set of b where {p,a,b} is isosceles
    n_pts = N * N
    pts = list(itertools.product(range(N), repeat=2))
    
    # iso[p_idx][a_idx] = set of b_idx such that {p,a,b} is isosceles
    iso = [dict() for _ in range(n_pts)]
    
    for p_idx in range(n_pts):
        p = pts[p_idx]
        for a_idx in range(n_pts):
            if a_idx == p_idx:
                continue
            a = pts[a_idx]
            da = _d2_idx(p_idx, a_idx)
            for b_idx in range(n_pts):
                if b_idx == p_idx or b_idx == a_idx:
                    continue
                b = pts[b_idx]
                db = _d2_idx(p_idx, b_idx)
                dab = _d2_idx(a_idx, b_idx)
                if da == db or da == dab or db == dab:
                    if a_idx not in iso[p_idx]:
                        iso[p_idx][a_idx] = set()
                    iso[p_idx][a_idx].add(b_idx)
    
    return iso, pts

def _can_add_idx(p_idx, chosen_set, iso):
    """Check if point p_idx can be added to chosen_set (as set of indices)."""
    for a_idx in chosen_set:
        if a_idx in iso[p_idx]:
            if chosen_set & iso[p_idx][a_idx]:
                return False
    return True

def _is_maximal_idx(chosen_set, iso, n_pts):
    """Check if no point can be added."""
    for p_idx in range(n_pts):
        if p_idx not in chosen_set:
            if _can_add_idx(p_idx, chosen_set, iso):
                return False
    return True

def _greedy_idx(order, iso):
    """Greedy construction using indices."""
    chosen = set()
    for p_idx in order:
        if _can_add_idx(p_idx, chosen, iso):
            chosen.add(p_idx)
    return chosen

def _local_search_remove(chosen_set, iso, n_pts, rng, max_iters=500):
    """Try to remove points while maintaining maximality."""
    current = set(chosen_set)
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        # Try removing each point
        pts_list = list(current)
        rng.shuffle(pts_list)
        for p_idx in pts_list:
            new_set = current - {p_idx}
            # Check if new_set is still maximal
            if _is_maximal_idx(new_set, iso, n_pts):
                current = new_set
                improved = True
                break
    return current

def _local_search_swap(chosen_set, iso, n_pts, rng, max_iters=200):
    """Try swap operations: remove one point, add another if possible."""
    current = set(chosen_set)
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        pts_list = list(current)
        rng.shuffle(pts_list)
        for p_idx in pts_list:
            new_set = current - {p_idx}
            # Try to add a different point
            candidates = [q for q in range(n_pts) if q not in new_set]
            rng.shuffle(candidates)
            for q_idx in candidates:
                if _can_add_idx(q_idx, new_set, iso):
                    test_set = new_set | {q_idx}
                    if _is_maximal_idx(test_set, iso, n_pts):
                        current = test_set
                        improved = True
                        break
            if improved:
                break
    return current

def _construct_maximal_from_base_idx(base_set, iso, n_pts):
    """Add points to base_set until maximal."""
    chosen = set(base_set)
    for p_idx in range(n_pts):
        if p_idx not in chosen:
            if _can_add_idx(p_idx, chosen, iso):
                chosen.add(p_idx)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s
    
    n_pts = N * N
    all_pts = list(itertools.product(range(N), repeat=2))
    
    # Build conflict structure
    iso, pts = _build_conflict_graph()
    
    best = None
    best_len = float('inf')
    
    # Initial greedy in row-major order
    order = list(range(n_pts))
    chosen = _greedy_idx(order, iso)
    if len(chosen) < best_len:
        best = chosen
        best_len = len(chosen)
    
    # Try a few more deterministic orderings
    # Diagonal order
    diag_order = sorted(range(n_pts), key=lambda i: (pts[i][0] + pts[i][1], pts[i][0]))
    chosen = _greedy_idx(diag_order, iso)
    if len(chosen) < best_len:
        best = chosen
        best_len = len(chosen)
    
    # Checkerboard order
    even_order = [i for i in range(n_pts) if (pts[i][0] + pts[i][1]) % 2 == 0]
    odd_order = [i for i in range(n_pts) if (pts[i][0] + pts[i][1]) % 2 == 1]
    chosen = _greedy_idx(even_order + odd_order, iso)
    if len(chosen) < best_len:
        best = chosen
        best_len = len(chosen)
    
    # Randomized search with local search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate random ordering
        pts_list = list(range(n_pts))
        rng.shuffle(pts_list)
        
        # Greedy
        chosen = _greedy_idx(pts_list, iso)
        
        # Make maximal
        chosen = _construct_maximal_from_base_idx(chosen, iso, n_pts)
        
        # Local search: try to remove points
        if len(chosen) < best_len + 5:  # Only if close to best
            improved = _local_search_remove(chosen, iso, n_pts, rng, max_iters=100)
            if len(improved) < len(chosen):
                chosen = improved
        
        # Try swap
        if len(chosen) < best_len + 3:
            swapped = _local_search_swap(chosen, iso, n_pts, rng, max_iters=50)
            if len(swapped) < len(chosen):
                chosen = swapped
        
        if len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
        
        # Periodic restart with different strategies
        if best_len < 20:
            # We're doing well, be more aggressive
            pass
    
    # Convert best set back to (x,y) tuples
    if best is not None:
        result = [pts[i] for i in sorted(best)]
    else:
        # Fallback
        result = [pts[i] for i in sorted(_greedy_idx(list(range(n_pts)), iso))]
    
    return result
# EVOLVE-BLOCK-END
