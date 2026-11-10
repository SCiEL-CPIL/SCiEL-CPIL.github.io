import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    num_bad = len(bad_triples)
    
    # Build conflict graph: edge (x,y) exists if x,y appear together in some bad triple
    conflict = [[False] * n for _ in range(n)]
    for (a, b, c) in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        conflict[ia][ib] = conflict[ib][ia] = True
        conflict[ia][ic] = conflict[ic][ia] = True
        conflict[ib][ic] = conflict[ic][ib] = True
    
    # Precompute adjacency lists
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and conflict[i][j]:
                adj[i].append(j)
    
    # Also track which triples each pair is in
    pair_triples = defaultdict(list)
    for t, (a, b, c) in enumerate(bad_triples):
        ia, ib, ic = a - 1, b - 1, c - 1
        pair_triples[(ia, ib)].append(t)
        pair_triples[(ib, ia)].append(t)
        pair_triples[(ia, ic)].append(t)
        pair_triples[(ic, ia)].append(t)
        pair_triples[(ib, ic)].append(t)
        pair_triples[(ic, ib)].append(t)
    
    # For each element, list of bad triples it participates in
    elem_triples = [[] for _ in range(n)]
    for t, (a, b, c) in enumerate(bad_triples):
        elem_triples[a-1].append(t)
        elem_triples[b-1].append(t)
        elem_triples[c-1].append(t)
    
    triple_elems = [(a-1, b-1, c-1) for (a, b, c) in bad_triples]
    
    def count_violations(selected):
        """Count the number of bad triples that are fully selected."""
        cnt = 0
        for t in range(num_bad):
            a, b, c = triple_elems[t]
            if selected[a] and selected[b] and selected[c]:
                cnt += 1
        return cnt
    
    def add_violations(selected, x):
        """If we add element x, how many new violations are created?"""
        cnt = 0
        for t in elem_triples[x]:
            a, b, c = triple_elems[t]
            others = [v for v in (a, b, c) if v != x]
            if selected[others[0]] and selected[others[1]]:
                cnt += 1
        return cnt
    
    # Phase 1: Greedy construction with multiple orderings
    best_size = 0
    best_sel = [False] * n
    
    # Ordering 1: by degree (fewest constraints first)
    degree = [len(adj[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda i: (degree[i], i))
    
    sel = [False] * n
    for x in order1:
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 2: reverse degree
    order2 = sorted(range(n), key=lambda i: (-degree[i], i))
    sel = [False] * n
    for x in order2:
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 3: natural order
    sel = [False] * n
    for x in range(n):
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 4: reverse natural
    sel = [False] * n
    for x in range(n-1, -1, -1):
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Greedy addition pass on best
    changed = True
    while changed and time.time() < deadline - 40:
        changed = False
        for x in range(n):
            if not best_sel[x] and add_violations(best_sel, x) == 0:
                best_sel[x] = True
                changed = True
    best_size = sum(best_sel)
    
    # Phase 2: Tomita-style branch-and-bound for maximum independent set
    # on the conflict graph. The conflict graph has an edge between x,y if
    # they co-occur in some bad triple. A valid solution is an independent set
    # in this graph (no edge has both endpoints selected).
    #
    # Wait - that's NOT right. The constraint is that no BAD TRIPLE is fully
    # selected. Two elements can be in a bad triple together as long as the
    # third isn't also selected. So the conflict graph model is too strong.
    #
    # The correct model: we need a subset with no hyperedge (bad triple) fully
    # contained. This is a 3-uniform hypergraph independent set.
    #
    # For branch-and-bound on hypergraph independent set:
    # - Branch on vertices: include or exclude
    # - If we include vertex v, we must ensure no bad triple containing v
    #   has all other two vertices also included.
    # - Bound: current_count + |undecided| (optimistic)
    # - Stronger bound: current_count + max_independent_set(undecided_subgraph)
    #   where undecided_subgraph uses the conflict graph as upper bound.
    
    # Use bitsets for speed
    # Represent sets as Python integers (bit i set means element i is in set)
    
    # Build conflict as bitmasks
    conflict_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            conflict_mask[i] |= (1 << j)
    
    # For each element, bitmask of elements that form a bad triple with it
    # (i.e., the other two elements in any bad triple containing i)
    # Actually, for the hypergraph: if we include i, we need to track which
    # pairs (j,k) are "forbidden" - both j and k cannot be included.
    
    # Precompute: for each element i, list of (j,k) pairs such that (i,j,k) is bad
    elem_pairs = [[] for _ in range(n)]
    for t in range(num_bad):
        a, b, c = triple_elems[t]
        elem_pairs[a].append((b, c))
        elem_pairs[b].append((a, c))
        elem_pairs[c].append((a, b))
    
    # Branch and bound with memoization-like pruning
    # State: (undecided_mask, forbidden_pairs_mask) is too complex.
    # Simpler: branch on vertices, track selected set, use conflict graph bound.
    
    # Actually, let's use a cleaner approach:
    # The problem is: maximize |S| subject to for all (a,b,c) in bad_triples,
    # not (a in S and b in S and c in S).
    #
    # Branch and bound:
    # - Order vertices by some heuristic
    # - At each step, branch: include current vertex or exclude it
    # - If include: check it doesn't complete any bad triple with already-included vertices
    #   If it would, we must exclude it.
    # - Bound: |selected| + |remaining vertices that could potentially be added|
    #   Upper bound on addable: use greedy coloring on the induced conflict subgraph
    #   of remaining vertices as an upper bound on their independent set size.
    
    # For the coloring bound, we need the conflict graph on remaining vertices.
    # But the conflict graph is too strong (it forbids pairs that can coexist
    # if the third element of their triple is absent).
    
    # Let me use a different approach: since n=128 and the structure is specific,
    # let's do a smarter local search + exact verification.
    
    # Phase 3: Improved local search with proper state management
    random.seed(42)
    
    def greedy_fill(sel):
        """Add all elements that can be added without creating violations."""
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if not sel[x] and add_violations(sel, x) == 0:
                    sel[x] = True
                    changed = True
        return sum(sel)
    
    def local_search_v2(sel, max_iters=50):
        """Local search with 1-swap and 1-remove/k-add moves."""
        sz = greedy_fill(sel)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 8:
            improved = False
            iters += 1
            
            # Try removing one element and re-filling
            for r in range(n):
                if not sel[r]:
                    continue
                sel[r] = False
                new_sz = greedy_fill(sel)
                if new_sz > sz:
                    sz = new_sz
                    improved = True
                else:
                    sel[r] = True
                    # Re-fill in case removal changed what's addable
                    sz = greedy_fill(sel)
        
        return sz
    
    # Apply local search to best
    best_size = local_search_v2(best_sel, max_iters=80)
    
    # Phase 4: Multiple random restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 8:
            break
        
        if restart % 3 == 0:
            # Random ordering
            order = list(range(n))
            random.shuffle(order)
        elif restart % 3 == 1:
            # Random perturbation of best
            sel = best_sel[:]
            for _ in range(3):
                included = [i for i in range(n) if sel[i]]
                excluded = [i for i in range(n) if not sel[i]]
                if included:
                    r = random.choice(included)
                    sel[r] = False
                if excluded:
                    a = random.choice(excluded)
                    if add_violations(sel, a) == 0:
                        sel[a] = True
            order = None  # Use perturbed solution directly
        else:
            # Weighted random: prefer low-degree elements
            weights = [1.0 / (1 + degree[i]) for i in range(n)]
            total_w = sum(weights)
            order = list(range(n))
            for i in range(n - 1, 0, -1):
                r = random.random() * total_w
                cum = 0
                for j in range(i + 1):
                    cum += weights[order[j]]
                    if cum >= r:
                        order[j], order[i] = order[i], order[j]
                        break
                total_w -= weights[order[i]]
        
        if order is not None:
            sel = [False] * n
            for x in order:
                if add_violations(sel, x) == 0:
                    sel[x] = True
        else:
            sel = sel[:]  # Already perturbed
        
        sz = greedy_fill(sel)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]
        
        sz = local_search_v2(sel, max_iters=40)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]
    
    # Phase 5: Simulated annealing-style refinement
    if time.time() < deadline - 10:
        temp = 5.0
        best_sel_sa = best_sel[:]
        best_size_sa = best_size
        for sa_iter in range(2000):
            if time.time() > deadline - 8:
                break
            if sa_iter % 200 == 199:
                temp *= 0.7
            
            # Random move: flip a random element
            x = random.randint(0, n - 1)
            old_state = best_sel_sa[x]
            best_sel_sa[x] = not old_state
            
            # Check validity
            valid = True
            for t in elem_triples[x]:
                a, b, c = triple_elems[t]
                if best_sel_sa[a] and best_sel_sa[b] and best_sel_sa[c]:
                    valid = False
                    break
            
            if valid:
                new_size = sum(best_sel_sa)
                if new_size >= best_size_sa:
                    best_size_sa = new_size
                    if new_size > best_size:
                        best_size = new_size
                        best_sel = best_sel_sa[:]
                else:
                    # Accept with probability exp(-delta/temp)
                    if random.random() > (0.001 ** ((best_size_sa - new_size) / temp)):
                        best_sel_sa[x] = old_state
            else:
                best_sel_sa[x] = old_state
    
    # Phase 6: Final exact verification and repair
    # Iteratively remove violations until none remain
    fixed = True
    while fixed and time.time() < deadline - 3:
        fixed = False
        for t in range(num_bad):
            a, b, c = triple_elems[t]
            if best_sel[a] and best_sel[b] and best_sel[c]:
                # Remove the element with highest degree to minimize future conflicts
                degs = [len(adj[a]), len(adj[b]), len(adj[c])]
                max_idx = degs.index(max(degs))
                best_sel[[a, b, c][max_idx]] = False
                fixed = True
    
    # Final greedy fill after repairs
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for x in range(n):
            if not best_sel[x] and add_violations(best_sel, x) == 0:
                best_sel[x] = True
                changed = True
    
    # Final verification - ensure no bad triple is fully selected
    for t in range(num_bad):
        a, b, c = triple_elems[t]
        if best_sel[a] and best_sel[b] and best_sel[c]:
            best_sel[c] = False
            best_sel[b] = False
            best_sel[a] = False
    
    # Re-verify after aggressive fix
    for t in range(num_bad):
        a, b, c = triple_elems[t]
        if best_sel[a] and best_sel[b] and best_sel[c]:
            best_sel[c] = False
    
    result = [x + 1 for x in range(n) if best_sel[x]]
    if not result:
        result = [1]
    
    return result
