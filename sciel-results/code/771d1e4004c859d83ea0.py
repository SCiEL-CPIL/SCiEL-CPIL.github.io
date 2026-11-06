import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute weights as Fractions
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
    
    # Build conflict graph: conflict[x] = set of y that conflict with x
    # x and y conflict if there exists z in [1,N] such that {x,y,z} forms a 3-AP
    # A 3-AP: a, a+d, a+2d for d != 0
    # So x,y conflict if:
    # - x,y are endpoints: (x+y) even, mid=(x+y)//2 in [1,N], mid != x,y
    # - x is middle: z = 2x-y in [1,N], z != x,y
    # - y is middle: z = 2y-x in [1,N], z != x,y
    
    conflict = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            # Check if x,y can be in a 3-AP
            is_conflict = False
            # Case 1: x,y are endpoints
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    is_conflict = True
            # Case 2: x is middle (y > x, so z = 2x - y < x)
            if not is_conflict:
                z = 2 * x - y
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            # Case 3: y is middle (z = 2y - x > y)
            if not is_conflict:
                z = 2 * y - x
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            
            if is_conflict:
                conflict[x].add(y)
                conflict[y].add(x)
    
    # Convert to bitmasks for fast operations
    # bitmask[i] has bit j set if i conflicts with j
    bitmask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        bitmask[i] = mask
    
    # Order by decreasing weight (increasing index)
    # We process vertices in order: 1, 2, 3, ..., N
    # This naturally gives higher weight first
    
    # Precompute suffix sums for upper bounds
    # suffix[i] = sum of weights[j] for j from i to N
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
    
    best_score = Fraction(0)
    best_set = set()
    
    # We'll use iterative deepening or a priority queue approach
    # Actually, let's use a recursive B&B with good pruning
    
    # Key optimization: when we include vertex v, we remove v and all its neighbors
    # from consideration. When we exclude v, we just move to next.
    
    # Use a list to represent the current available set (as indices)
    # For efficiency, we'll track which vertices are still available
    
    sys.setrecursionlimit(10000)
    
    def dfs(available, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if not available:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper bound: current_score + sum of weights of all available
        # This is a loose bound but fast
        ub = current_score
        for v in available:
            ub += weights[v]
        if ub <= best_score:
            return
        
        # Pick a vertex to branch on
        # Strategy: pick the vertex with highest weight among available
        # Or use a heuristic: pick vertex that conflicts with many high-weight vertices
        # For now, pick the smallest index (highest weight) available
        v = min(available)
        
        # Option 1: Include v
        new_available = [u for u in available if u != v and u not in conflict[v]]
        new_score = current_score + weights[v]
        new_set = current_set | {v}
        
        if new_score > best_score:
            # Check if this alone beats best (quick update)
            pass
        
        dfs(new_available, new_score, new_set)
        
        # Option 2: Exclude v
        remaining = [u for u in available if u != v]
        dfs(remaining, current_score, current_set)
    
    # Start with all vertices available
    initial_available = list(range(1, N + 1))
    dfs(initial_available, Fraction(0), set())
    
    return list(best_set)
