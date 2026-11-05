"""Rewritten program: Maximum Independent Set on conflict graph for Property P."""
import random

N = 256

def _build_conflict_bitsets():
    """
    Build conflict bitsets based on the directive:
    Edge (u, v) with u < v exists if there exists an integer a in [2, u-1]
    such that a divides (u + v).
    
    We ignore a=1 because optimal sets will not contain 1 (if 1 is in S,
    S can have size at most 2, which is suboptimal).
    """
    # conflicts[i] is a bitset of vertices j such that (i, j) is an edge.
    # We only consider vertices 2..N for the main search, but bitset covers 0..N.
    conflicts = [0] * (N + 1)
    
    for u in range(2, N + 1):
        for v in range(u + 1, N + 1):
            s = u + v
            # Check if any a in [2, u-1] divides s
            has_conflict = False
            # Optimization: iterate a from 2 to u-1
            for a in range(2, u):
                if s % a == 0:
                    has_conflict = True
                    break
            
            if has_conflict:
                conflicts[u] |= (1 << v)
                conflicts[v] |= (1 << u)
                
    return conflicts

def construct():
    """
    Find a large valid subset of [1, 256] using MIS heuristic on the conflict graph.
    """
    random.seed(42)
    
    # Precompute conflicts
    conflicts = _build_conflict_bitsets()
    
    best_size = 0
    best_set = []
    
    # Fallback: The seed solution
    fallback = list(range(192, 257))
    best_size = len(fallback)
    best_set = fallback
    
    # Run MIS heuristic with random restarts
    # We search in 2..N. Vertex 1 is isolated in the graph but implies strict constraints
    # in the original problem, so we exclude it from the main set construction.
    
    num_restarts = 200
    
    for _ in range(num_restarts):
        # Random order of vertices 2..N
        order = list(range(2, N + 1))
        random.shuffle(order)
        
        S_mask = 0
        S_list = []
        
        for v in order:
            # Check if v has any conflict with current S
            # conflicts[v] contains all neighbors of v
            if (conflicts[v] & S_mask) == 0:
                S_mask |= (1 << v)
                S_list.append(v)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
    return best_set
