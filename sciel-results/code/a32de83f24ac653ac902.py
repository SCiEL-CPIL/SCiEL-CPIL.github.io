import time
import random
from math import gcd

NMAX = 256

def _valid_addition(x, S_list):
    """
    Check if x can be added to set S_list without violating the constraint.
    For all distinct a,b,c in final set, a does not divide b*c.
    When adding x, check triples involving x:
    - (x, b, c): x | b*c?
    - (b, x, c): b | x*c?
    - (c, x, b): c | x*b?
    for all b,c in S_list, b != c.
    """
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _precompute_conflicts():
    """
    Precompute pairwise conflicts.
    Two numbers u, v are in conflict if there exists w in [2, 256], w != u, w != v,
    such that w | u*v.
    Note: The problem constraint is on triples (a,b,c) distinct.
    If we build a set S, and u, v in S, they are "compatible" only if no w in S
    (w != u, w != v) divides u*v.
    
    Wait, the "2-uniform conflict graph" idea from the inspiration suggests:
    If we define a graph where an edge (u,v) exists if they CANNOT both be in the set
    along with ANY other element? No, that's not right.
    
    Let's re-read the constraint:
    "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
    
    This is a 3-uniform hypergraph constraint.
    A set S is valid if for all distinct a,b,c in S, a !| b*c.
    
    The inspiration suggests reducing to a 2-uniform conflict graph where an edge
    exists between u, v if there exists a THIRD integer w in [2, 256] (not necessarily in S?)
    such that w | uv.
    
    If w | uv, and w is in S, and u, v are in S, and u!=v!=w!=u, then the constraint is violated.
    So, if we select u and v, we must NOT select any w that divides uv (where w != u, v).
    
    This means:
    If u, v are in S, then S cannot contain any w such that w | uv and w != u, v.
    
    This is equivalent to saying:
    For any pair u, v in S, the set of "forbidden" third elements is F(u,v) = { w in [2, 256] \ {u,v} | w | uv }.
    If S contains u and v, then S \ {u,v} must be disjoint from F(u,v).
    
    This doesn't immediately reduce to a simple independent set in a graph on [2, 256] because
    the conflict depends on the third element.
    
    However, we can use this for a "pressure" score in greedy selection.
    If we pick u, it "blocks" many other numbers if they form pairs with existing numbers
    that have common divisors.
    
    Let's stick to the direct _valid_addition check but optimize the search.
    The current program uses numpy which might be slower due to overhead for small N.
    Pure Python list operations with early exit might be faster for N~50.
    
    Let's refine the search strategy:
    1. Greedy initialization with different orderings (largest, smallest, random, weighted).
    2. Local search with Add/Swap/Remove moves.
    3. Pre-compute divisibility relationships to speed up checks?
       _valid_addition is O(N^2). N is small.
       We can precompute for each x, the list of y such that x | y (or similar).
    """
    pass

def _greedy(order, deadline):
    """
    Greedy construction.
    order: 'largest', 'smallest', 'random', 'weighted'
    """
    chosen = []
    chosen_set = set()
    
    if order == 'largest':
        candidates = list(range(NMAX, 1, -1))
    elif order == 'smallest':
        candidates = list(range(2, NMAX + 1))
    elif order == 'random':
        candidates = list(range(2, NMAX + 1))
        random.shuffle(candidates)
    elif order == 'weighted':
        # Favor numbers with fewer divisors/multiples?
        # Or just random with seed
        candidates = list(range(2, NMAX + 1))
        random.shuffle(candidates)
    else:
        candidates = list(range(2, NMAX + 1))
        random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
            
    return chosen

def _local_search(S, deadline, max_iters=5000, rng_seed=42):
    """
    Local search to improve the set size.
    Moves:
    1. Add a random valid element.
    2. Swap: remove a random element, add a random valid element.
    3. Remove: remove a random element (to escape local optima, though we want max size).
    """
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    # Pre-compute potential candidates for speed?
    # Just iterate through all numbers 2..256 not in S.
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:10]:  # Try a few
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        elif move_type < 0.8:
            # Try to swap
            if S:
                idx = rng.randrange(len(S))
                removed = S[idx]
                S_temp = S[:idx] + S[idx+1:]
                S_temp_set = set(S_temp)
                
                candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
                if candidates:
                    rng.shuffle(candidates)
                    swapped = False
                    for x in candidates[:10]:
                        if _valid_addition(x, S_temp):
                            S = S_temp + [x]
                            S_set = set(S)
                            # Size is same, but structure changed.
                            # We don't update best_size here unless we can add more later.
                            # But for now, we just accept the swap to explore.
                            swapped = True
                            break
                    if not swapped:
                        S = S_temp + [removed]
                        S_set = set(S)
        else:
            # Try to remove (rarely, to escape deep local minima where swaps don't help)
            if S:
                idx = rng.randrange(len(S))
                removed = S[idx]
                S.pop(idx)
                S_set.discard(removed)
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_S = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = ['largest', 'smallest', 'random', 'random', 'random', 'weighted', 'weighted']
    for i, ord in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        seed = i * 1000
        random.seed(seed)
        S_init = _greedy(ord, deadline - 5)
        if len(S_init) > best_size:
            best_S = S_init
            best_size = len(S_init)
        
        # Local search
        S_improved = _local_search(S_init, deadline - 5, max_iters=2000, rng_seed=seed)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    # Strategy 2: More random restarts if time remains
    while time.time() < deadline - 2:
        seed = int(time.time() * 1000) % 100000
        random.seed(seed)
        S_init = _greedy('random', deadline - 2)
        if len(S_init) > best_size:
            best_S = S_init
            best_size = len(S_init)
        
        S_improved = _local_search(S_init, deadline - 2, max_iters=1000, rng_seed=seed)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    if not best_S:
        best_S = [2, 3]
        
    return [int(x) for x in best_S]
