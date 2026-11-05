import time
from math import gcd
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Strategy:
    1. Precompute gcd matrix and identify all "bad" triples (where all three pairwise gcds are equal).
    2. Use a fast conflict-checking mechanism for local search.
    3. Perform multiple greedy constructions with different orderings.
    4. Use efficient local search (add-only, then swap-based) to improve.
    5. Use simulated annealing with fast neighborhood evaluation.
    
    Key optimization: maintain a "conflict count" array for each element relative to the current set,
    so we can quickly determine if adding an element would create a bad triple.
    """
    
    DEADLINE = time.time() + 82.0  # safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # For fast conflict checking: 
    # When we have a set S, adding x creates a bad triple iff there exist u,v in S
    # such that gcd(x,u) == gcd(x,v) == gcd(u,v).
    # 
    # We can precompute for each pair (u,v) the set of x that would form a bad triple.
    # But that's expensive. Instead, for a given x and set S, we check:
    # For each u in S, compute g1 = gcd(x,u). For each v in S, v != u, compute g2 = gcd(x,v).
    # If g1 == g2 == gcd(u,v), then (x,u,v) is bad.
    #
    # Faster approach: for a fixed x, group elements of S by gcd(x, .).
    # If two elements u, v in the same group (same gcd with x) also have gcd(u,v) equal to that value,
    # then (x,u,v) is bad.
    #
    # Even faster: precompute for each x, for each possible gcd value g, the list of elements
    # that have gcd(x, elem) = g. Then for a set S, we just need to check if for any g,
    # there are two elements u,v in S ∩ group_g such that gcd(u,v) = g.
    
    # Precompute: for each x, group elements by gcd(x, elem)
    # groups[x][g] = list of elements y such that gcd(x,y) = g
    groups = [None] * (N + 1)
    for x in range(1, N + 1):
        g_map = {}
        for y in range(1, N + 1):
            if x == y:
                continue
            g = gcd_mat[x][y]
            if g not in g_map:
                g_map[g] = []
            g_map[g].append(y)
        groups[x] = g_map
    
    # For each x and each gcd value g, precompute which pairs (u,v) in groups[x][g]
    # have gcd(u,v) == g. These are the "conflicting pairs" for x with gcd value g.
    # conflicting_pairs[x][g] = set of frozenset({u,v}) where gcd(x,u)=gcd(x,v)=gcd(u,v)=g
    conflicting_pairs = [None] * (N + 1)
    for x in range(1, N + 1):
        cp = {}
        for g, elems in groups[x].items():
            pairs = set()
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    u, v = elems[i], elems[j]
                    if gcd_mat[u][v] == g:
                        pairs.add(frozenset((u, v)))
            if pairs:
                cp[g] = pairs
        conflicting_pairs[x] = cp
    
    def has_conflict(x, selected):
        """Check if adding x to selected creates a bad triple."""
        if not selected:
            return False
        # For each gcd value g that x shares with some elements in selected,
        # check if there's a conflicting pair in selected.
        for g, pairs in conflicting_pairs[x].items():
            # Check if any pair in 'pairs' is fully contained in selected
            for pair in pairs:
                u, v = tuple(pair)
                if u in selected and v in selected:
                    return True
        return False
    
    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if not has_conflict(x, selected):
                selected.add(x)
        return selected
    
    def local_search_add_only(selected, deadline=DEADLINE):
        """Try to add any excluded element that doesn't create conflicts."""
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x not in selected and not has_conflict(x, selected):
                    selected.add(x)
                    improved = True
        return selected
    
    def local_search_swap(selected, deadline=DEADLINE, max_iters=50):
        """Try swap moves: remove one element, add another."""
        selected = set(selected)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in selected]
            
            # Try removing each element and adding the best excluded element
            for x in list(selected):
                if time.time() > deadline:
                    break
                temp = selected - {x}
                # Try to add excluded elements to temp
                new_sel = set(temp)
                added_any = False
                for y in excluded:
                    if y == x:
                        continue
                    if not has_conflict(y, new_sel):
                        new_sel.add(y)
                        added_any = True
                # Try adding more
                for y in excluded:
                    if y == x or y in new_sel:
                        continue
                    if not has_conflict(y, new_sel):
                        new_sel.add(y)
                
                if len(new_sel) > len(selected):
                    selected = new_sel
                    improved = True
                    break
            if not improved:
                break
        return selected
    
    def simulated_annealing(initial_set, deadline=DEADLINE, max_iters=10000):
        """Simulated annealing to maximize set size."""
        import math
        
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        temp = 10.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if not current:
                break
            
            # Generate neighbor: randomly remove one element
            to_remove = random.choice(list(current))
            new_set = set(current)
            new_set.remove(to_remove)
            
            # Try to add excluded elements
            excluded = [x for x in range(1, N + 1) if x not in new_set]
            random.shuffle(excluded)
            for x in excluded[:30]:
                if not has_conflict(x, new_set):
                    new_set.add(x)
            
            old_size = len(current)
            new_size = len(new_set)
            delta = new_size - old_size
            
            if delta >= 0:
                current = new_set
                if new_size > best_size:
                    best = set(new_set)
                    best_size = new_size
            else:
                prob = math.exp(delta / max(temp, 1e-6))
                if random.random() < prob:
                    current = new_set
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best
    
    # --- Main search ---
    
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Phase 1: Greedy builds with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(elements, key=lambda x: sum(1 for y in elements if y != x and gcd_mat[x][y] > 1)),
        sorted(elements, key=lambda x: -sum(1 for y in elements if y != x and gcd_mat[x][y] > 1)),
    ]
    
    for order in orderings:
        if time.time() > DEADLINE:
            break
        s = greedy_build(order)
        s = local_search_add_only(s)
        s = local_search_swap(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Phase 2: Randomized greedy + local search
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        s = local_search_add_only(s)
        s = local_search_swap(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Phase 3: Simulated annealing
    if time.time() < DEADLINE:
        sa_set = simulated_annealing(best_set)
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set
    
    # Phase 4: Final local search
    if time.time() < DEADLINE:
        s = local_search_add_only(best_set)
        s = local_search_swap(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
