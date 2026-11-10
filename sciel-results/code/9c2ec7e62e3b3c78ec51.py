# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement S.
    
    Structural approach: Model as maximum independent set on conflict graph.
    Two points conflict if they lie on a common line. S must be an independent set
    (no two points of S on the same line). Then N = F_49^2 \\ S is automatically
    a valid Nikodym set because:
    - For x in S: the line through x and any other point y in S doesn't exist (independent),
      so pick any line L through x; L \ {x} has no S points, hence L \ {x} ⊆ N.
    - For x in N: pick any line L through x; since S is independent, L ∩ S = ∅ (if x ∉ S),
      so L ⊆ N, hence L \ {x} ⊆ N.
    
    Wait - that's not quite right. Let me reconsider.
    
    Actually the condition is:
    - x in S: exists line L through x with L ∩ S = {x}
    - x in N: exists line L through x with L ∩ S = ∅
    
    If S is an independent set (no two S points on same line), then:
    - For x in S: ANY line L through x has L ∩ S ⊆ {x}. Since x ∈ L, L ∩ S = {x}. ✓
    - For x in N: ANY line L through x has L ∩ S = ∅ (since x ∉ S and no other S point on L). ✓
    
    So ANY independent set S works! We just need to maximize |S| subject to
    no two points of S lying on a common line.
    
    This is exactly a maximum independent set / maximum matching dual problem.
    In the line-point incidence structure, it's a maximum set of points with no
    two on the same line - this is equivalent to a maximum matching in the dual,
    or more directly, it's the problem of selecting the maximum number of points
    such that each line contains at most one selected point.
    
    This can be formulated as: for each of the 2450 lines, at most one point is selected.
    We want to maximize the number of selected points.
    
    This is a packing problem. A simple greedy approach:
    1. Randomly order all points
    2. For each point in order, add it to S if no line through it already has a point in S
    
    With random restarts and different orderings, we can explore the space.
    """
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 590.0  # Safety margin
    
    # Field constants
    k = 3  # alpha^2 = 3
    
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    # Precompute all lines and line membership using bitsets for speed
    # Line representation: numpy bool array of size total_points
    
    lines = []  # list of numpy bool arrays
    point_to_lines = [[] for _ in range(total_points)]  # list of line indices per point
    
    # Vertical lines: x = const
    for xa in range(p):
        for xb in range(p):
            mask = np.zeros(total_points, dtype=bool)
            for ya in range(p):
                for yb in range(p):
                    mask[get_idx(xa, xb, ya, yb)] = True
            lines.append(mask)
            for ya in range(p):
                for yb in range(p):
                    point_to_lines[get_idx(xa, xb, ya, yb)].append(len(lines) - 1)
    
    # Slanted lines: y = m*x + b for each m in F_49
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
    
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
    
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            seen = set()
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = add((xa, xb), t)
                                    y = add((ya, yb), mul(m, t))
                                    pts.append(get_idx(x[0], x[1], y[0], y[1]))
                            key = frozenset(pts)
                            if key not in seen:
                                seen.add(key)
                                mask = np.zeros(total_points, dtype=bool)
                                for pt in pts:
                                    mask[pt] = True
                                lines.append(mask)
                                for pt in pts:
                                    point_to_lines[pt].append(len(lines) - 1)
    
    num_lines = len(lines)
    
    # Precompute: for each point, which lines contain it (as indices)
    # Already have point_to_lines
    
    # Greedy construction with random ordering
    def greedy_construct(order):
        """Build S by processing points in given order, adding if no conflict."""
        S = set()
        # Track for each line whether it already has a point in S
        line_occupied = np.zeros(num_lines, dtype=bool)
        
        for x in order:
            # Check if any line through x is already occupied
            can_add = True
            for l_idx in point_to_lines[x]:
                if line_occupied[l_idx]:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                for l_idx in point_to_lines[x]:
                    line_occupied[l_idx] = True
        
        return S
    
    # Local search: try swaps (remove one, add another)
    def local_search(S, time_limit):
        """Improve S via swap operations."""
        S = set(S)
        line_occupied = np.zeros(num_lines, dtype=bool)
        for x in S:
            for l_idx in point_to_lines[x]:
                line_occupied[l_idx] = True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all possible single swaps: remove x from S, add y to S
            # y must not conflict with S \ {x}
            # This is expensive, so use random sampling
            S_list = list(S)
            random.shuffle(S_list)
            
            for x in S_list:
                if time.time() >= time_limit:
                    break
                # Temporarily remove x
                for l_idx in point_to_lines[x]:
                    line_occupied[l_idx] = False
                S.remove(x)
                
                # Try to add a random point not in S
                # Sample candidates
                for _ in range(20):
                    y = random.randint(0, total_points - 1)
                    if y in S:
                        continue
                    # Check if y can be added
                    can_add = True
                    for l_idx in point_to_lines[y]:
                        if line_occupied[l_idx]:
                            can_add = False
                            break
                    if can_add:
                        S.add(y)
                        for l_idx in point_to_lines[y]:
                            line_occupied[l_idx] = True
                        improved = True
                        break
                
                # If no improvement found for this x, put x back
                if x not in S:
                    S.add(x)
                    for l_idx in point_to_lines[x]:
                        line_occupied[l_idx] = True
        
        return S
    
    # Run multiple random restarts
    best_S = set()
    best_size = 0
    
    num_restarts = 200
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create random order
        order = list(range(total_points))
        rng.shuffle(order)
        
        # Greedy construction
        S = greedy_construct(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # Local search refinement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Allocate a fraction of remaining time to local search
            ls_time = min(remaining_time * 0.3, 30.0)
            S = local_search(S, time.time() + ls_time)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
        
        if time.time() > deadline - 10:
            break
    
    # Build N = F_49^2 \\ best_S
    points = []
    for x in range(total_points):
        if x not in best_S:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
