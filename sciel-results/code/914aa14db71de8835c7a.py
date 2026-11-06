"""Exact MIS solver for sum-free subset problem using branch-and-bound on conflict graph."""
# EVOLVE-BLOCK-START
import time
from random import Random
from functools import lru_cache

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _build_conflict_graph(S):
    """
    Build conflict graph where edge (u,v) exists if u+v is in S, or 2u is in S, or u is sum of two others.
    Actually, for sum-free: no a+b=c with a,b,c in set.
    Two elements x,y conflict if:
    - x+y is in S (then x,y,x+y can't all be in set)
    - 2x is in S (then x,x,2x can't all be in set) - but this is about x itself
    - x can be written as sum of two elements in S
    
    Better: define conflict between x and y if {x,y} cannot both be in a sum-free set.
    This happens if:
    1. x + y is in S (then x, y, x+y form a triple)
    2. x = a + b for some a,b in S (but this doesn't directly conflict x with y)
    
    Actually, the cleanest approach: two elements x, y are incompatible if there exists z in S 
    such that {x, y, z} contains a sum triple. The sum triples involving x,y are:
    - x + y = z (z in S)
    - x + z = y (z in S) 
    - y + z = x (z in S)
    - 2x = y or 2y = x (if x=y, then 2x=x means x=0, excluded)
    
    So x and y conflict if:
    - (x+y) in S
    - (y-x) in S and y-x != 0 (i.e., x + (y-x) = y)
    - (x-y) in S and x-y != 0 (i.e., y + (x-y) = x)
    - x == y and 2*x in S (but we handle distinct elements)
    
    For a set S, build graph where vertex = element, edge between x,y if they can't coexist.
    """
    S_set = set(S)
    n = len(S)
    # Adjacency list
    adj = [set() for _ in range(n)]
    elem_to_idx = {v: i for i, v in enumerate(S)}
    
    for i in range(n):
        for j in range(i + 1, n):
            x, y = S[i], S[j]
            # Check if x+y is in S
            if (x + y) in S_set:
                adj[i].add(j)
                adj[j].add(i)
            # Check if |x-y| is in S (x + (y-x) = y or y + (x-y) = x)
            diff = abs(x - y)
            if diff != 0 and diff in S_set:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj, elem_to_idx, S

def _mis_branch_and_bound(adj, n, deadline):
    """
    Exact maximum independent set via branch-and-bound.
    adj: list of sets, adj[i] = neighbors of i
    Returns size of MIS.
    """
    best_size = [0]
    
    # Use greedy coloring for upper bound
    def greedy_color_bound(candidates):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns number of colors = upper bound on independence number.
        """
        if not candidates:
            return 0
        
        color_class = []
        colors = {}
        num_colors = 0
        
        for v in candidates:
            # Find first available color
            used = set()
            for u in adj[v]:
                if u in candidates and u in colors:
                    used.add(colors[u])
            
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
            color_class.append(v)
        
        return num_colors
    
    def branch(candidates, current_size):
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        # Upper bound: current_size + number of vertices in candidates
        # Better: current_size + greedy color bound
        ub = current_size + len(candidates)
        if ub <= best_size[0]:
            return
        
        # Even better upper bound using coloring
        color_bound = greedy_color_bound(candidates)
        if current_size + color_bound <= best_size[0]:
            return
        
        # Pick vertex with max degree in candidate subgraph
        v = max(candidates, key=lambda x: len(adj[x] & set(candidates)))
        
        # Branch 1: include v
        # Remove v and all its neighbors from candidates
        new_candidates = [u for u in candidates if u != v and u not in adj[v]]
        branch(new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        new_candidates = [u for u in candidates if u != v]
        branch(new_candidates, current_size)
    
    branch(list(range(n)), 0)
    return best_size[0]

def _mis_with_reconstruction(adj, n, deadline):
    """
    Find MIS and reconstruct the actual set.
    Returns list of vertex indices in MIS.
    """
    best_set = [set()]
    
    def greedy_color_bound(candidates):
        if not candidates:
            return 0
        colors = {}
        num_colors = 0
        for v in candidates:
            used = set()
            for u in adj[v]:
                if u in candidates and u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
        return num_colors
    
    def branch(candidates, current_set, current_size):
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > len(best_set[0]):
                best_set[0] = set(current_set)
            return
        
        ub = current_size + len(candidates)
        if ub <= len(best_set[0]):
            return
        
        color_bound = greedy_color_bound(candidates)
        if current_size + color_bound <= len(best_set[0]):
            return
        
        v = max(candidates, key=lambda x: len(adj[x] & set(candidates)))
        
        # Branch 1: include v
        new_candidates = [u for u in candidates if u != v and u not in adj[v]]
        branch(new_candidates, current_set + [v], current_size + 1)
        
        # Branch 2: exclude v
        new_candidates = [u for u in candidates if u != v]
        branch(new_candidates, current_set, current_size)
    
    branch(list(range(n)), [], 0)
    return list(best_set[0])

def _greedy_independent_set(adj, n):
    """Simple greedy: pick vertices in order of low degree, add if no conflict with already picked."""
    picked = set()
    remaining = list(range(n))
    # Sort by degree
    remaining.sort(key=lambda v: len(adj[v]))
    
    for v in remaining:
        if v in picked:
            continue
        # Check if v conflicts with any already picked
        can_add = True
        for u in adj[v]:
            if u in picked:
                can_add = False
                break
        if can_add:
            picked.add(v)
    
    return list(picked)

def _greedy_odd(S):
    return [x for x in S if x % 2 != 0]

def _greedy_by_key(S, key_func):
    """Greedy construction: add elements in order of key_func if they don't conflict."""
    S_set = set()
    result = []
    for x in sorted(S, key=key_func):
        # Check if x can be added
        can_add = True
        for c in S_set:
            if (x + c) in S_set or (x - c) in S_set or (c - x) in S_set:
                can_add = False
                break
            if x == c and 2*x in S_set:
                can_add = False
                break
        if can_add:
            S_set.add(x)
            result.append(x)
    return result

def _local_search(S, rng, deadline):
    """Improve a sum-free subset of S via local search."""
    S_set = set(S)
    all_elements = set(S)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    if not _is_sum_free(S_set):
        S_set = set(_greedy_by_key(list(S), lambda v: -abs(v)))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
    
    while time.time() < deadline:
        improved = False
        
        for x in list(excluded):
            if time.time() >= deadline:
                break
            # Check if x can be added
            can_add = True
            for c in S_set:
                if (x + c) in S_set or (x - c) in S_set or (c - x) in S_set:
                    can_add = False
                    break
            if can_add:
                S_set.add(x)
                excluded.remove(x)
                improved = True
                if len(S_set) > best:
                    best = len(S_set)
                    best_set = set(S_set)
        
        if not improved:
            break
    
    return list(best_set)

def construct():
    deadline = time.time() + 145.0
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for idx, (size, seed) in enumerate(instances):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        remaining_time = deadline - time.time()
        num_remaining = len(instances) - idx
        inst_budget = remaining_time / num_remaining - 2.0
        inst_deadline = time.time() + inst_budget
        
        # Build conflict graph
        adj, elem_to_idx, S_list = _build_conflict_graph(S)
        n = len(S)
        
        # Try exact MIS solver with time limit
        mis_deadline = min(time.time() + inst_budget * 0.7, inst_deadline)
        mis_indices = _mis_with_reconstruction(adj, n, mis_deadline)
        
        # Convert indices to values
        mis_set = [S_list[i] for i in mis_indices]
        
        # Verify
        if not _is_sum_free(mis_set):
            # Fallback to greedy
            mis_set = _greedy_independent_set(adj, n)
            mis_set = [S_list[i] for i in mis_set]
        
        # Local search to improve
        local_deadline = min(time.time() + inst_budget * 0.25, inst_deadline - 1.0)
        result = _local_search(mis_set, rng, local_deadline)
        
        # Final validation
        if not _is_sum_free(result):
            # Fallback to odd numbers
            result = _greedy_odd(S)
            if not _is_sum_free(result):
                result = _greedy_by_key(S, lambda v: -abs(v))
        
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
