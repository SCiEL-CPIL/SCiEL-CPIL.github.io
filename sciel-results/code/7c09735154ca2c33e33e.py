"""Rewritten program using MIS-inspired branch-and-bound for dissociated reciprocal sets."""
import itertools
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_dissociated(denominators):
    """
    Check if the set of reciprocals of 'denominators' has distinct subset sums.
    """
    k = len(denominators)
    if k == 0:
        return True
    L = _lcm_list(denominators)
    ints = [L // d for d in denominators]
    
    sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += ints[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _find_minimal_bad_sets(universe, max_size=5):
    """
    Find all minimal subsets of 'universe' that are NOT dissociated,
    up to size max_size.
    """
    bad_sets = []
    n = len(universe)
    
    # Precompute LCM for all subsets up to max_size for speed
    # Actually, let's just compute on the fly but cache if needed.
    # For n=22, checking all subsets of size 5 is C(22,5) = 26334.
    # Each check involves LCM and 2^5=32 sums. Very fast.
    
    for size in range(3, max_size + 1):
        for comb in itertools.combinations(universe, size):
            if not _is_dissociated(list(comb)):
                # Check if it's minimal: no proper subset is bad
                is_minimal = True
                for i in range(size):
                    sub = list(comb[:i] + comb[i+1:])
                    if not _is_dissociated(sub):
                        is_minimal = False
                        break
                if is_minimal:
                    bad_sets.append(set(comb))
    
    return bad_sets

def _build_conflict_graph(universe, bad_sets):
    """
    Build a conflict graph where an edge (u, v) exists if u and v
    are contained in some bad set.
    """
    n = len(universe)
    index_map = {v: i for i, v in enumerate(universe)}
    # Adjacency list
    adj = [set() for _ in range(n)]
    
    for b in bad_sets:
        nodes = [index_map[x] for x in b]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                adj[nodes[i]].add(nodes[j])
                adj[nodes[j]].add(nodes[i])
                
    return adj

def _greedy_mis(universe, adj):
    """
    Greedy approximation for Maximum Independent Set.
    Order vertices by degree (ascending), pick if no conflict with already picked.
    """
    n = len(universe)
    # Sort by degree
    order = sorted(range(n), key=lambda i: len(adj[i]))
    picked = set()
    for i in order:
        # Check if i conflicts with any picked node
        conflict = False
        for j in picked:
            if j in adj[i]:
                conflict = True
                break
        if not conflict:
            picked.add(i)
    return [universe[i] for i in picked]

def _max_dissociated_subset(universe, bad_sets):
    """
    Find the largest dissociated subset using Branch-and-Bound.
    """
    n = len(universe)
    best_set = set()
    best_size = 0
    
    # Pre-index bad sets for fast subset check
    # A set S is invalid if it contains any b in bad_sets.
    
    # Order universe to improve pruning.
    # Heuristic: process numbers that are in many bad sets first?
    # Or just natural order? Let's try natural order for simplicity, 
    # but maybe sorting by "constraint tightness" helps.
    
    # For now, use natural order 1..22.
    # We can also use the greedy MIS solution as a lower bound.
    adj = _build_conflict_graph(universe, bad_sets)
    greedy_sol = _greedy_mis(universe, adj)
    if greedy_sol:
        best_set = set(greedy_sol)
        best_size = len(best_set)
        
    def backtrack(index, current_set, current_size):
        nonlocal best_set, best_size
        
        # Pruning
        if current_size + (n - index) <= best_size:
            return
            
        if index == n:
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
            return
        
        x = universe[index]
        
        # Option 1: Include x
        # Check if current_set U {x} contains any bad set
        new_set = current_set | {x}
        valid = True
        for b in bad_sets:
            if b.issubset(new_set):
                valid = False
                break
        
        if valid:
            backtrack(index + 1, new_set, current_size + 1)
            
        # Option 2: Exclude x
        backtrack(index + 1, current_set, current_size)
        
    backtrack(0, set(), 0)
    return list(best_set)

def construct():
    universe = list(range(1, 23))
    # Find minimal bad sets up to size 5
    bad_sets = _find_minimal_bad_sets(universe, max_size=5)
    # Find max subset avoiding them
    result = _max_dissociated_subset(universe, bad_sets)
    
    # Ensure non-empty
    if not result:
        return [1]
        
    # Verify the result is actually dissociated
    if not _is_dissociated(result):
        # Fallback to a known good set if something went wrong
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return result

# EVOLVE-BLOCK-END
