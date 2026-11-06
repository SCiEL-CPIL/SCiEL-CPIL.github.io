"""Rewritten: branch-and-bound with dynamic reordering for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
import sys

def _build_conflict_graph():
    """
    Build conflict graph on grid [0,N)^2.
    Edge (i,j) exists if there is a third point k such that {i,j,k} forms an isosceles triangle.
    A set S is "no-isosceles" iff S is an independent set in this graph.
    """
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    
    # Precompute squared distances
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    
    # conflict[i] = set of j such that edge (i,j) exists
    conflict = [set() for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            d_ij = dist2[i, j]
            if d_ij == 0:
                continue
            # Check if there exists k != i, j such that triangle (i,j,k) is isosceles
            # Isosceles means: d_ik == d_ij OR d_jk == d_ij OR d_ik == d_jk
            found = False
            for k in range(V):
                if k == i or k == j:
                    continue
                d_ik = dist2[i, k]
                d_jk = dist2[j, k]
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    found = True
                    break
            if found:
                conflict[i].add(j)
                conflict[j].add(i)
    
    return points, conflict

POINTS, CONFLICT = _build_conflict_graph()
V = len(POINTS)

def _is_independent(S, conflict):
    """Check if S is an independent set."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if b in conflict[a]:
                return False
    return True

def _is_maximal(S, conflict):
    """Check if S is a maximal independent set: no vertex can be added."""
    S_set = set(S)
    for v in range(V):
        if v in S_set:
            continue
        # Can we add v? Check if v conflicts with any in S
        can_add = True
        for u in S_set:
            if v in conflict[u]:
                can_add = False
                break
        if can_add:
            return False
    return True

def _branch_and_bound(deadline, best_size, best_set, rng):
    """
    Branch and bound to find minimum maximal independent set.
    State: (S_included, S_excluded, candidates)
    candidates = vertices not yet decided
    We maintain that S_included is independent.
    Maximality constraint: at the end, every vertex not in S_included must conflict with some vertex in S_included.
    
    Dynamic reordering: sort candidates by "coverage" = number of currently-uncovered vertices they block.
    """
    start_time = time.time()
    
    # Initialize
    S_included = set()
    S_excluded = set()
    candidates = set(range(V))
    
    # For efficiency, track which vertices are "covered" (conflict with some in S_included)
    # covered[v] = True if v conflicts with some u in S_included
    covered = [False] * V
    
    def _update_covered(v, add):
        """Update covered status when v is added to or removed from S_included."""
        for u in conflict[v]:
            if add:
                covered[u] = True
            else:
                # Only clear if no other included vertex covers u
                # This is expensive; we'll recompute when needed
                pass
    
    def _recompute_covered():
        """Recompute covered array from scratch based on S_included."""
        for i in range(V):
            covered[i] = False
        for v in S_included:
            for u in conflict[v]:
                covered[u] = True
    
    def _count_uncovered():
        """Count vertices not in S_included and not covered."""
        count = 0
        for v in candidates:
            if not covered[v]:
                count += 1
        return count
    
    # We'll use a recursive B&B with a stack to avoid recursion limits
    # State: (S_included, S_excluded, candidates, covered_array)
    # But copying covered_array is expensive. Let's use a different approach.
    
    # Alternative: use a priority queue / best-first search with pruning
    # Or: use a simple DFS with good ordering and pruning
    
    # Let's do DFS with dynamic ordering at each node
    # To keep it manageable, we'll limit the depth and use time checks
    
    best_size_local = best_size
    best_set_local = list(best_set)
    
    # Use a stack for iterative DFS
    # Each stack entry: (S_included, S_excluded, candidates, covered)
    # But this is memory-heavy. Let's do recursive with a depth limit.
    
    sys.setrecursionlimit(10000)
    
    def _dfs(S_included, S_excluded, candidates, covered):
        nonlocal best_size_local, best_set_local
        
        if time.time() > deadline:
            return
        
        # Pruning: if |S_included| >= best_size_local, no improvement possible
        if len(S_included) >= best_size_local:
            return
        
        # If no candidates left, check maximality
        if not candidates:
            # Check maximality: every vertex not in S_included must be covered
            for v in range(V):
                if v not in S_included and not covered[v]:
                    return  # Not maximal
            # Valid maximal independent set
            if len(S_included) < best_size_local:
                best_size_local = len(S_included)
                best_set_local = list(S_included)
            return
        
        # Pruning: if even including all remaining candidates can't make it maximal
        # i.e., there's an uncovered vertex that no candidate can cover
        # An uncovered vertex v can be covered by adding some candidate u that conflicts with v
        uncovered = [v for v in candidates if not covered[v]]
        if uncovered:
            # For each uncovered v, check if there's a candidate u that conflicts with v
            # If not, v can never be covered -> impossible to be maximal
            for v in uncovered:
                can_cover = False
                for u in candidates:
                    if u == v:
                        continue
                    if v in conflict[u]:
                        can_cover = True
                        break
                if not can_cover:
                    return  # Dead end
        
        # Dynamic ordering: sort candidates by "coverage power"
        # Coverage power of u = number of uncovered vertices that u conflicts with
        def _coverage_power(u):
            if u in S_excluded:
                return -1
            power = 0
            for v in candidates:
                if v != u and not covered[v] and v in conflict[u]:
                    power += 1
            # Also, u itself becomes "in set" so it's no longer uncovered
            # But we count the effect on others
            return power
        
        # Get ordered candidates
        ordered = sorted(candidates, key=lambda u: -_coverage_power(u))
        
        # Branch: try including the highest-coverage candidate first
        # Then try excluding it
        for idx, u in enumerate(ordered):
            if time.time() > deadline:
                return
            
            # Option 1: Include u
            if u not in S_excluded:
                new_S_included = S_included | {u}
                new_candidates = candidates - {u}
                # Update covered: all neighbors of u become covered
                new_covered = covered[:]
                for v in conflict[u]:
                    new_covered[v] = True
                _dfs(new_S_included, S_excluded, new_candidates, new_covered)
            
            # Option 2: Exclude u
            new_S_excluded = S_excluded | {u}
            new_candidates = candidates - {u}
            _dfs(S_included, new_S_excluded, new_candidates, covered)
    
    _dfs(set(), set(), set(range(V)), [False] * V)
    
    return best_size_local, best_set_local

def _local_search(S, conflict, deadline):
    """
    Local search: try to remove points from S while maintaining maximality.
    If removing a point breaks maximality, try adding a different point.
    """
    S_set = set(S)
    current_size = len(S_set)
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each point
        for v in list(S_set):
            if len(S_set) - 1 >= current_size:
                continue
            
            new_S = S_set - {v}
            
            # Check if new_S is still independent (it should be, since S was)
            # Check if new_S is maximal
            if _is_maximal(new_S, conflict):
                S_set = new_S
                current_size = len(S_set)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove v, add w
            for v in list(S_set):
                for w in range(V):
                    if w in S_set:
                        continue
                    # Check if w conflicts with any in (S_set - {v})
                    new_S = S_set - {v} | {w}
                    # Check independence
                    independent = True
                    for a in new_S:
                        for b in new_S:
                            if a != b and b in conflict[a]:
                                independent = False
                                break
                        if not independent:
                            break
                    if not independent:
                        continue
                    # Check maximality
                    if _is_maximal(new_S, conflict):
                        S_set = new_S
                        current_size = len(S_set)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return list(S_set)

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_size = V
    best_set = list(range(V))
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction to get a good initial solution
    def _greedy_construct():
        S = set()
        covered = [False] * V
        
        while True:
            if time.time() > deadline:
                break
            
            # Find uncovered vertices
            uncovered = [v for v in range(V) if v not in S and not covered[v]]
            if not uncovered:
                break
            
            # For each candidate, compute coverage
            best_u = None
            best_power = -1
            for u in range(V):
                if u in S:
                    continue
                # Check if u conflicts with any in S
                valid = True
                for v in S:
                    if u in conflict[v]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute coverage power
                power = 0
                for v in uncovered:
                    if v != u and v in conflict[u]:
                        power += 1
                # Also, u itself
                power += 1
                
                if power > best_power:
                    best_power = power
                    best_u = u
            
            if best_u is None:
                # No valid candidate, pick any uncovered
                best_u = uncovered[0]
            
            S.add(best_u)
            for v in conflict[best_u]:
                covered[v] = True
        
        return list(S)
    
    # Run greedy multiple times with randomization
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        S = set()
        covered = [False] * V
        
        while True:
            if time.time() > deadline:
                break
            
            uncovered = [v for v in range(V) if v not in S and not covered[v]]
            if not uncovered:
                break
            
            best_u = None
            best_power = -1
            for u in range(V):
                if u in S:
                    continue
                valid = True
                for v in S:
                    if u in conflict[v]:
                        valid = False
                        break
                if not valid:
                    continue
                
                power = 0
                for v in uncovered:
                    if v != u and v in conflict[u]:
                        power += 1
                power += 1
                
                # Add random noise
                power += rng.random() * 0.1
                
                if power > best_power:
                    best_power = power
                    best_u = u
            
            if best_u is None:
                best_u = uncovered[0]
            
            S.add(best_u)
            for v in conflict[best_u]:
                covered[v] = True
        
        # Local search to polish
        S_list = _local_search(S, conflict, deadline)
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list
            
            if best_size <= 12:
                break
    
    # Phase 2: Branch and bound with the best solution as upper bound
    if time.time() < deadline - 30:
        bb_size, bb_set = _branch_and_bound(deadline - 10, best_size, best_set, rng)
        if bb_size < best_size:
            best_size = bb_size
            best_set = bb_set
    
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
