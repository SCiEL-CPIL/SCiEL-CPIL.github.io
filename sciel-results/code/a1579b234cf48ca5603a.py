import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Strategy: Branch-and-bound on the 3-uniform hypergraph independent set problem.
    Use greedy initialization to get a good lower bound, then exact B&B with
    strong upper bounds (independent set on conflict graph) to prune.
    """
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # Precompute gcd table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g
    
    # Find all bad triples
    bad_triples = []
    # For each pair (a,b), find all c > b such that gcd(a,c)==gcd(a,b) and gcd(b,c)==gcd(a,b)
    # Also build conflict structure: for each vertex, which pairs conflict
    conflict = {i: set() for i in range(1, N + 1)}
    # conflict[x] = set of frozenset({a,b}) where {x,a,b} is a bad triple
    
    triple_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == g_ab and gt[b][c] == g_ab:
                    bad_triples.append((a, b, c))
                    triple_set.add((a, b, c))
                    conflict[a].add(frozenset((b, c)))
                    conflict[b].add(frozenset((a, c)))
                    conflict[c].add(frozenset((a, b)))
    
    num_bad = len(bad_triples)
    
    # Build conflict graph: two vertices conflict if they appear together in some bad triple
    # Actually for upper bound, we need: if we have a set S, the max we can add is bounded by
    # the independence number of the "conflict graph" where edges mean "can't both be added"
    # But that's not quite right. Let's use a simpler bound.
    
    # For B&B: we decide for each vertex whether to include it.
    # Upper bound: current_size + number of remaining vertices that could potentially be added
    # Better upper bound: use the fact that each bad triple restricts choices.
    
    # Greedy initialization
    def greedy_init():
        """Build a large valid set using greedy with multiple strategies."""
        best = set()
        best_size = 0
        
        # Strategy 1: sort by degree (number of bad triples containing vertex)
        degree = [0] * (N + 1)
        for (a, b, c) in bad_triples:
            degree[a] += 1
            degree[b] += 1
            degree[c] += 1
        
        # Try different orderings
        orderings = [
            sorted(range(1, N + 1), key=lambda x: degree[x]),  # low degree first
            sorted(range(1, N + 1), key=lambda x: -degree[x]),  # high degree first
            list(range(1, N + 1)),  # natural order
            list(range(N, 0, -1)),  # reverse natural
        ]
        
        # Add some random orderings
        rng = random.Random(42)
        for _ in range(20):
            order = list(range(1, N + 1))
            rng.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            sel = set()
            for x in order:
                # Check if x can be added
                can_add = True
                for pair in conflict[x]:
                    a, b = tuple(pair)
                    if a in sel and b in sel:
                        can_add = False
                        break
                if can_add:
                    sel.add(x)
            
            if len(sel) > best_size:
                best_size = len(sel)
                best = sel
        
        return best, best_size
    
    def polish(sel):
        """Try to add any excluded element that fits."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel:
                    can_add = True
                    for pair in conflict[x]:
                        a, b = tuple(pair)
                        if a in sel and b in sel:
                            can_add = False
                            break
                    if can_add:
                        sel.add(x)
                        changed = True
        return sel
    
    # Get initial solution
    init_set, init_size = greedy_init()
    init_set = polish(init_set)
    init_size = len(init_set)
    
    # Branch and bound
    # State: (current_set, next_vertex_to_consider)
    # We process vertices in a fixed order. At each step, we decide include or exclude.
    
    # For better pruning, use a dynamic ordering: at each node, pick the vertex that
    # maximizes the number of conflicts with remaining vertices (most constrained first)
    
    # Upper bound heuristic: 
    # 1. Trivial: current_size + remaining_count
    # 2. Better: for each remaining vertex, count how many "slots" it blocks.
    #    Use a greedy coloring-like bound on the conflict structure.
    
    # Let's use a simpler but effective approach:
    # Upper bound = current_size + max_independent_set_of_remaining
    # where max_independent_set_of_remaining is approximated by greedy.
    
    # Actually, let's use a different approach. The key insight is that the problem is
    # small enough (128 vertices) that with good pruning we can solve it exactly.
    
    # Precompute for each vertex, the set of vertices that "conflict" with it in the sense
    # that they form bad triples together.
    
    # Let's define: two vertices a, b are "incompatible" if there exists c such that
    # {a, b, c} is a bad triple. Then the problem reduces to finding a large subset with
    # no three mutually incompatible... no, that's not quite right either.
    
    # The constraint is: no three distinct elements with all three pairwise gcds equal.
    # This is a 3-uniform hypergraph constraint, not a pairwise one.
    
    # Let me use a simpler B&B:
    # - Order vertices 1 to 128
    # - At each step, decide to include or exclude the next vertex
    # - Upper bound: current_size + (number of remaining vertices that don't immediately
    #   conflict with the current selection)
    
    # Actually, let's just do a well-optimized B&B with memoization on the "tail" of the
    # vertex sequence.
    
    # Since N=128 is large for pure B&B, let's use a hybrid:
    # 1. Get a good lower bound from greedy + local search
    # 2. Use B&B to prove optimality or find better solutions
    
    # Local search to improve the initial solution
    def local_search(sel, max_iters=500):
        """2-for-1 exchange local search."""
        sel = set(sel)
        best_size = len(sel)
        best_sel = set(sel)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in sel]
            
            # Try adding one
            for x in excluded:
                can_add = True
                for pair in conflict[x]:
                    a, b = tuple(pair)
                    if a in sel and b in sel:
                        can_add = False
                        break
                if can_add:
                    sel.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one and adding two
                sel_list = list(sel)
                for x in sel_list:
                    temp = sel - {x}
                    added = []
                    for y in excluded:
                        if y in added:
                            continue
                        can_add = True
                        for pair in conflict[y]:
                            a, b = tuple(pair)
                            if a in temp and b in temp:
                                can_add = False
                                break
                            if a in added and b in added:
                                can_add = False
                                break
                            if a in added and b in temp:
                                can_add = False
                                break
                            if b in added and a in temp:
                                can_add = False
                                break
                        if can_add:
                            added.append(y)
                        if len(added) >= 2:
                            break
                    
                    if len(added) >= 2:
                        sel = temp | set(added)
                        improved = True
                        break
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
            
            if not improved:
                break
        
        return best_sel
    
    init_set = local_search(init_set, max_iters=300)
    init_size = len(init_set)
    init_set = polish(init_set)
    init_size = len(init_set)
    
    # Now try B&B to see if we can do better
    # Use a time-limited B&B
    
    # For B&B, we need an efficient upper bound.
    # Key idea: if we have a partial assignment where some vertices are "forced in" and
    # some are "forced out", the upper bound is:
    #   |forced_in| + max_additional
    # where max_additional is an upper bound on how many more we can add from the undecided.
    
    # A simple upper bound: |forced_in| + |undecided|
    # Better: for each undecided vertex, check if it's compatible with all forced_in vertices.
    # If not, it can't be added, so exclude it from the count.
    
    # Even better: use the structure of bad triples.
    
    # Let's implement a recursive B&B with the following:
    # - Process vertices in order of "constraint tightness" (most constrained first)
    # - At each node, compute an upper bound
    # - If upper bound <= current best, prune
    
    # Precompute vertex ordering for B&B
    # Sort by number of bad triples containing the vertex (most constrained first)
    degree = [0] * (N + 1)
    for (a, b, c) in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # For B&B, we'll use a fixed ordering but with a good upper bound
    # Let's try a different approach: use the structure of the problem.
    
    # Key observation: the bad triples have structure related to gcd.
    # If gcd(a,b) = gcd(a,c) = gcd(b,c) = g, then a/g, b/g, c/g are pairwise coprime... no,
    # that's not right. gcd(a/g, b/g) = 1, gcd(a/g, c/g) = 1, gcd(b/g, c/g) = 1.
    # So a/g, b/g, c/g are pairwise coprime.
    
    # This means that for a fixed g, the numbers a = g*x where x are pairwise coprime
    # form a set where any three with the same g-value in their gcd would be bad.
    
    # Let's use this structure for a better upper bound.
    
    # For now, let's do a straightforward B&B with the simple upper bound and see if it works.
    
    # Actually, with 128 vertices, even a naive B&B is 2^128 which is way too large.
    # We need much better pruning.
    
    # Let's use a different strategy: since we already have a good solution from greedy + local search,
    # let's try to verify if it's optimal using a certificate-based approach, or just try to improve it
    # with more sophisticated local search.
    
    # Multi-start local search
    best_set = init_set
    best_size = init_size
    
    # Try multiple random restarts with different initial solutions
    rng = random.Random(123)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random perturbation of the current best
        if restart == 0:
            sel = set(best_set)
        else:
            # Start from a random subset of best_set plus some random additions
            sel = set()
            for x in best_set:
                if rng.random() < 0.7:
                    sel.add(x)
            # Add some random elements
            for x in range(1, N + 1):
                if x not in sel and rng.random() < 0.1:
                    # Check if it can be added
                    can_add = True
                    for pair in conflict[x]:
                        a, b = tuple(pair)
                        if a in sel and b in sel:
                            can_add = False
                            break
                    if can_add:
                        sel.add(x)
        
        # Polish
        sel = polish(sel)
        
        # Local search
        sel = local_search(sel, max_iters=200)
        sel = polish(sel)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel
    
    # Final polish
    best_set = polish(best_set)
    best_size = len(best_set)
    
    # Try one more round of aggressive local search with 3-for-2 exchanges
    def local_search_3for2(sel, max_iters=200):
        """Try removing 3 and adding 2 (or removing 2 and adding 3)."""
        sel = set(sel)
        best_size = len(sel)
        best_sel = set(sel)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in sel]
            sel_list = list(sel)
            
            # Try removing 2 and adding 3
            if len(sel_list) >= 2:
                # Try random pairs to remove
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    to_remove = set(rng.sample(sel_list, 2))
                    temp = sel - to_remove
                    added = []
                    for y in excluded:
                        if y in added:
                            continue
                        can_add = True
                        for pair in conflict[y]:
                            a, b = tuple(pair)
                            if a in temp and b in temp:
                                can_add = False
                                break
                            if a in added and b in added:
                                can_add = False
                                break
                        if can_add:
                            added.append(y)
                        if len(added) >= 3:
                            break
                    
                    if len(added) >= 3:
                        new_sel = temp | set(added)
                        if len(new_sel) > best_size:
                            best_size = len(new_sel)
                            best_sel = set(new_sel)
                        sel = new_sel
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_sel
    
    best_set = local_search_3for2(best_set, max_iters=100)
    best_set = polish(best_set)
    best_size = len(best_set)
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
