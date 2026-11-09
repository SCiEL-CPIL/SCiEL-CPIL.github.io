import random
from math import gcd
import time
from itertools import combinations

N = 128

def compute_bad_triples():
    """
    Precompute all bad triples (a, b, c) with a < b < c
    where gcd(a,b) = gcd(b,c) = gcd(a,c)
    """
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad_triples.add((a, b, c))
    return bad_triples

BAD_TRIPLES = compute_bad_triples()

# Build conflict graph: edge (u, v) exists if there's a w such that {u,v,w} is a bad triple
# Actually, we need: two elements u, v conflict if they cannot both be in the set
# because there exists some w that would form a bad triple with both.
# But that's not quite right. The constraint is: no three distinct elements have equal pairwise GCDs.
# So the conflict is on triples, not pairs.
# 
# Let me reconsider: We want to find a subset S such that no three elements in S form a bad triple.
# This is a 3-uniform hypergraph independent set problem.
#
# The inspiration suggests modeling as a 2-uniform conflict graph where an edge exists between u, v
# if there exists a w such that gcd(u,v) = gcd(u,w) = gcd(v,w).
# But this doesn't directly give us the constraint we want.
#
# Let me think differently: The constraint is that for any three distinct elements a, b, c in S,
# NOT (gcd(a,b) = gcd(b,c) = gcd(a,c)).
#
# One approach: For each pair (a, b), define the set of "forbidden" elements c such that
# gcd(a,c) = gcd(b,c) = gcd(a,b). If both a and b are in S, then none of these c can be in S.
#
# This is still a 3-uniform constraint. The branch-and-bound approach would work directly on the hypergraph.

element_conflicts = {i: set() for i in range(1, N + 1)}
for (a, b, c) in BAD_TRIPLES:
    element_conflicts[a].add((b, c))
    element_conflicts[b].add((a, c))
    element_conflicts[c].add((a, b))

def can_add(x, current_set):
    """Check if x can be added to current_set without forming a bad triple."""
    for u, v in element_conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

def greedy_build(order):
    """Build a set greedily following the given order."""
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s, deadline, max_iter=100):
    """
    Improve set s using local search.
    """
    s = set(s)
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                return s
            if can_add(x, s):
                s.add(x)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try removing 1 element and adding 2
        list_s = list(s)
        random.shuffle(list_s)
        
        found_improvement = False
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            addable_1 = []
            for x in range(1, N + 1):
                if x not in s_without_u:
                    if can_add(x, s_without_u):
                        addable_1.append(x)
            
            if not addable_1:
                continue
                
            random.shuffle(addable_1)
            
            for x1 in addable_1:
                if time.time() > deadline:
                    return s
                
                s_with_x1 = s_without_u | {x1}
                
                addable_2 = []
                for x2 in range(1, N + 1):
                    if x2 not in s_with_x1:
                        if can_add(x2, s_with_x1):
                            addable_2.append(x2)
                            
                if addable_2:
                    s.remove(u)
                    s.add(x1)
                    s.add(addable_2[0])
                    found_improvement = True
                    break
            
            if found_improvement:
                break
        
        if found_improvement:
            improved = True
            continue
            
        # Try removing 2 elements and adding 3
        list_s = list(s)
        random.shuffle(list_s)
        found_improvement = False
        
        for i in range(min(len(list_s), 10)):
            u = list_s[i]
            for j in range(i + 1, min(len(list_s), 10)):
                v = list_s[j]
                if time.time() > deadline:
                    return s
                
                s_without_uv = s - {u, v}
                
                temp_s = set(s_without_uv)
                added = []
                
                for x in range(1, N + 1):
                    if x not in temp_s:
                        if can_add(x, temp_s):
                            temp_s.add(x)
                            added.append(x)
                            if len(added) >= 3:
                                break
                            
                if len(added) >= 3:
                    s.remove(u)
                    s.remove(v)
                    for x in added[:3]:
                        s.add(x)
                    found_improvement = True
                    break
            
            if found_improvement:
                break
        
        if found_improvement:
            improved = True
            continue
            
        # If stuck, try a kick: remove 3 random elements, then try to fill
        if time.time() < deadline:
             if len(s) >= 3:
                 to_remove = random.sample(list(s), 3)
                 for x in to_remove:
                     s.remove(x)
                 for x in range(1, N + 1):
                     if x not in s:
                         if can_add(x, s):
                             s.add(x)
                 improved = True

    return s

def branch_and_bound_mis(vertices, adj, deadline, best_size=0):
    """
    Branch and bound for maximum independent set.
    vertices: list of vertices to consider
    adj: adjacency list (set of neighbors for each vertex)
    deadline: time limit
    best_size: current best size found
    Returns: (best_size, best_set)
    """
    if time.time() > deadline:
        return best_size, set()
    
    if not vertices:
        return best_size, set()
    
    # Upper bound: current best + number of remaining vertices
    if best_size + len(vertices) <= best_size:
        return best_size, set()
    
    # Pick a vertex to branch on (use degree heuristic)
    # Choose vertex with highest degree in the induced subgraph
    max_deg = -1
    pivot = None
    for v in vertices:
        deg = sum(1 for u in vertices if u in adj[v])
        if deg > max_deg:
            max_deg = deg
            pivot = v
    
    if pivot is None:
        return best_size, set()
    
    # Branch 1: Include pivot
    # If we include pivot, we can't include any of its neighbors
    remaining = [v for v in vertices if v != pivot and v not in adj[pivot]]
    size1, set1 = branch_and_bound_mis(remaining, adj, deadline, best_size)
    
    # Branch 2: Exclude pivot
    remaining2 = [v for v in vertices if v != pivot]
    size2, set2 = branch_and_bound_mis(remaining2, adj, deadline, best_size)
    
    # Return the better result
    if size1 >= size2:
        return size1, set1
    else:
        return size2, set2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = set()
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(element_conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(element_conflicts[x])),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        s = local_search(s, deadline, max_iter=50)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Strategy 2: Random restarts with local search
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline, max_iter=30)
        
        if len(s) > len(best_set):
            best_set = s.copy()
    
    return list(best_set)
