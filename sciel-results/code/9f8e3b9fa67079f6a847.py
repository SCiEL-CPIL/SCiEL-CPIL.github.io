"""CSP-based search for edge-colourings of K16 where every K4 sees >=5 colours.

Strategy:
  1. Use a structured initial colouring (sum mod 17) which is known valid with 17 colours.
  2. Try to reduce the number of distinct colours via local search (hill-climbing with restarts).
  3. Use a CSP backtracking approach with forward checking to find valid colourings
     using exactly k colours, for k = 13, 14, 15, ...
  4. Return the best (minimum colour count) valid colouring found within the time budget.

Key insight: The sum-mod-17 construction uses 17 colours. We need to find if 13, 14, or 15
colours suffice. The lower bound is 13, upper bound 15.

For the CSP approach:
- Variables: 120 edges
- Domain: {0, 1, ..., k-1} for each edge
- Constraints: For every 4-clique, the 6 edges must span at least 5 distinct colours.
  Equivalently, NOT (the 6 edges use <= 4 distinct colours).

We use a backtracking search with:
- MRV (Minimum Remaining Values) variable ordering
- Forward checking: when assigning a colour to an edge, check all K4s containing that edge
- If any K4 would have <= 4 colours given the current assignment, prune.

Given the large search space, we rely on:
- Good initial ordering (edges sorted by some heuristic)
- Random restarts
- Time budget management
"""

import time
import random
import sys
from itertools import combinations

# Precompute all K4s (sets of 4 vertices) and their 6 edges
# Vertex set: 0..15
# Edges: (i,j) for 0<=i<j<16, indexed lexicographically

def _build_edge_index():
    """Build mapping from (i,j) to linear index 0..119."""
    idx = {}
    count = 0
    for i in range(16):
        for j in range(i+1, 16):
            idx[(i,j)] = count
            count += 1
    return idx

def _build_k4s():
    """Build list of all K4s, each as a list of 6 edge indices."""
    eidx = _build_edge_index()
    k4s = []
    for quad in combinations(range(16), 4):
        edges = []
        for a, b in combinations(quad, 2):
            edges.append(eidx[(a, b)])
        k4s.append(edges)
    return k4s

# Precompute at module level for speed
_EDGE_IDX = _build_edge_index()
_K4S = _build_k4s()
_NUM_K4S = len(_K4S)  # C(16,4) = 1820

def _validate(colours):
    """Check if a colouring is valid: every K4 has >= 5 distinct colours."""
    for edges in _K4S:
        c = set()
        for e in edges:
            c.add(colours[e])
            if len(c) >= 5:
                break
        else:
            if len(c) < 5:
                return False
    return True

def _count_colours(colours):
    """Count distinct colours used."""
    return len(set(colours))

def _initial_colouring():
    """Sum-mod-17 initial colouring (known valid, 17 colours)."""
    return [(i + j) % 17 for i in range(16) for j in range(i+1, 16)]

def _local_search(colours, max_iters=50000, timeout=None, start_time=None):
    """Hill-climbing: try to reduce number of colours by recoloring edges.
    
    Returns (colours, num_colours) after optimization.
    """
    if start_time is None:
        start_time = time.time()
    colours = list(colours)
    n = len(colours)
    best = _count_colours(colours)
    current = best
    
    for it in range(max_iters):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        # Pick a random edge and try a random colour
        e = random.randint(0, n-1)
        old_c = colours[e]
        new_c = random.randint(0, best - 1)
        
        if new_c == old_c:
            continue
        
        colours[e] = new_c
        new_count = _count_colours(colours)
        
        if _validate(colours):
            if new_count < current:
                current = new_count
                best = new_count
            # Accept if same or fewer colours
        else:
            # Revert
            colours[e] = old_c
    
    return colours, best

def _csp_search(k, timeout=30.0, start_time=None, max_nodes=2000000):
    """CSP backtracking to find a valid colouring of K16 using exactly k colours.
    
    Returns colouring list of length 120 if found, else None.
    """
    if start_time is None:
        start_time = time.time()
    
    n = 120
    assignment = [-1] * n
    
    # For each K4, track the colours used so far
    # k4_colours[k4_idx] = set of colours assigned to edges in that K4
    k4_colours = [set() for _ in range(_NUM_K4S)]
    
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(n)]
    for ki, edges in enumerate(_K4S):
        for e in edges:
            edge_k4s[e].append(ki)
    
    # Variable ordering: try to order edges so that constrained ones come first
    # Heuristic: edges that appear in more K4s first, but all edges appear in C(14,2)=91 K4s
    # So use random order with some structure
    order = list(range(n))
    random.shuffle(order)
    
    node_count = 0
    deadline = start_time + timeout
    
    def backtrack(pos):
        nonlocal node_count
        if (time.time() > deadline) or (node_count > max_nodes):
            return False
        
        node_count += 1
        
        if pos == n:
            return True
        
        e = order[pos]
        
        # Try each colour
        for c in range(k):
            # Check: if we assign colour c to edge e, does any K4 containing e
            # now have all its edges assigned and use < 5 colours?
            feasible = True
            for ki in edge_k4s[e]:
                k4_edges = _K4S[ki]
                # Check if all edges in this K4 are assigned
                all_assigned = True
                for ee in k4_edges:
                    if assignment[ee] == -1:
                        all_assigned = False
                        break
                
                if all_assigned:
                    # Check distinct colours
                    cs = set(assignment[ee] for ee in k4_edges)
                    if len(cs) < 5:
                        feasible = False
                        break
                else:
                    # Even if not all assigned, check if the current colours + c
                    # could possibly reach 5. If the number of distinct colours
                    # already used in this K4 (including c) is such that even
                    # with maximum additional distinct colours it can't reach 5...
                    # Actually, since unassigned edges can take any colour, we can
                    # always hope they add new colours. So only check if all assigned.
                    pass
            
            if not feasible:
                continue
            
            # Also check: for each K4 containing e, if we add c, do we have
            # a conflict where the K4 can never reach 5 colours?
            # This is a weaker check. For stronger pruning:
            # If a K4 has all edges assigned except e, and the other 5 edges
            # use exactly 4 colours, then c must be a 5th colour.
            for ki in edge_k4s[e]:
                k4_edges = _K4S[ki]
                other_edges = [ee for ee in k4_edges if ee != e]
                # Check if all other edges are assigned
                all_other_assigned = all(assignment[ee] != -1 for ee in other_edges)
                if all_other_assigned:
                    other_colours = set(assignment[ee] for ee in other_edges)
                    if len(other_colours) <= 4:
                        # We need c to be different from all other_colours to get 5
                        if c in other_colours:
                            feasible = False
                            break
                        # Also need at least 5 total: other_colours has <= 4, plus c
                        # If other_colours has exactly 4 and c is new, we get 5. OK.
                        # If other_colours has < 4, we get < 5. Fail.
                        if len(other_colours) < 4:
                            feasible = False
                            break
            
            if not feasible:
                continue
            
            # Assign
            assignment[e] = c
            for ki in edge_k4s[e]:
                k4_colours[ki].add(c)
            
            if backtrack(pos + 1):
                return True
            
            # Unassign
            assignment[e] = -1
            for ki in edge_k4s[e]:
                k4_colours[ki].discard(c)
        
        return False
    
    result = backtrack(0)
    if result:
        return assignment
    return None

def construct():
    """Find minimum-colour edge-colouring of K16 where every K4 sees >= 5 colours.
    
    Returns 120 integer colours in lexicographic order of pairs (i,j).
    """
    start_time = time.time()
    TIME_LIMIT = 140.0  # Leave 10s margin before 150s kill
    
    # Start with known valid 17-colour solution
    colours = _initial_colouring()
    best_colours = list(colours)
    best_count = _count_colours(colours)  # 17
    
    # Phase 1: Local search to reduce from 17
    # Try to get down to 15 or 16
    for attempt in range(5):
        if (time.time() - start_time) > TIME_LIMIT * 0.4:
            break
        c, cnt = _local_search(list(colours), max_iters=20000, 
                               timeout=TIME_LIMIT * 0.15, start_time=time.time())
        if cnt < best_count:
            best_count = cnt
            best_colours = list(c)
    
    # Phase 2: CSP search for exact k colours, k = 13, 14, 15
    # Try in increasing order of k (we want minimum)
    for k in [13, 14, 15]:
        if (time.time() - start_time) > TIME_LIMIT * 0.7:
            break
        if k >= best_count:
            continue
        
        # Multiple restarts for CSP
        found = None
        remaining_time = TIME_LIMIT - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        num_restarts = min(3, max(1, int(remaining_time / 15)))
        for r in range(num_restarts):
            if (time.time() - start_time) > TIME_LIMIT * 0.85:
                break
            random.seed(42 + r * 1000 + k)
            result = _csp_search(k, timeout=min(12.0, remaining_time - 2.0),
                                 start_time=time.time())
            if result is not None:
                found = result
                break
        
        if found is not None:
            # Verify
            if _validate(found):
                best_colours = found
                best_count = k
                # We found a solution with k colours; since we tried in increasing order,
                # this is optimal (or as good as we can find)
                break
    
    # Final verification
    if not _validate(best_colours):
        # Fallback to initial
        best_colours = _initial_colouring()
        best_count = 17
    
    return best_colours
