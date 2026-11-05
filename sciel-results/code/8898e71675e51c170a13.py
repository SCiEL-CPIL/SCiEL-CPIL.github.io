import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Structural change: Instead of incremental greedy + local search (which
    gets trapped in local optima), use a randomized backtracking search with
    strong pruning based on the "forbidden triples" structure.
    
    Key insight: A bad triple {a,b,c} has gcd(a,b)=gcd(a,c)=gcd(b,c)=g.
    For a fixed g, the elements with gcd = g to each other form cliques in
    the "g-graph". We want to avoid triangles in this combined structure.
    
    Strategy:
    1. Precompute all bad triples.
    2. Use randomized backtracking with:
       - Ordering by "conflict density" 
       - Pruning when remaining vertices can't beat current best
       - Forward checking
    3. Multiple restarts with different orderings.
    4. Final local search polish.
    """
    N = 128
    deadline = time.time() + 82.0  # Safety margin
    
    # Precompute gcd table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g
    
    # Precompute bad triples: for each pair (a,b), find all c such that
    # gcd(a,b) = gcd(a,c) = gcd(b,c)
    # This means c must be a multiple of gcd(a,b) and gcd(a,c)=gcd(b,c)=g
    # where g = gcd(a,b).
    
    # For each element x, precompute the set of g-values that "conflict" with x
    # when x is added to a set containing a and b.
    
    # Actually, let's precompute: for each x, and for each g, which other elements
    # have gcd exactly g with x.
    # gcd_class[x][g] = set of y != x such that gcd(x,y) = g
    
    gcd_class = [dict() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            g = gcd(x, y)
            if g not in gcd_class[x]:
                gcd_class[x][g] = set()
            gcd_class[x][g].add(y)
    
    # Precompute all bad triples
    bad_triples_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == g_ab and gt[b][c] == g_ab:
                    bad_triples_set.add((a, b, c))
    
    # For fast checking: for a given set S and candidate x, check if adding x
    # creates any bad triple. This requires checking all pairs (a,b) in S where
    # gcd(a,x) = gcd(b,x) = gcd(a,b).
    
    def can_add(x, sel):
        """True if x can be added to sel without creating a bad triple."""
        # For each pair (a,b) in sel, check if gcd(a,x) = gcd(b,x) = gcd(a,b)
        # Optimization: group elements of sel by their gcd with x.
        # If two elements a, b in sel have the same gcd g with x, 
        # and gcd(a,b) = g, then (a,b,x) is a bad triple.
        
        from collections import defaultdict
        groups = defaultdict(list)
        for a in sel:
            g = gt[x][a]
            groups[g].append(a)
        
        for g, elems in groups.items():
            if len(elems) < 2:
                continue
            # Check if any pair in elems has gcd = g
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if gt[elems[i]][elems[j]] == g:
                        return False
        return True
    
    def greedy_order(order):
        """Build valid set by adding elements in given order."""
        sel = set()
        for x in order:
            if can_add(x, sel):
                sel.add(x)
        return sel
    
    def polish(sel):
        """Add any element that fits."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    changed = True
        return sel
    
    def local_search(sel, max_iters=300):
        """Local search with 2-for-1 exchanges."""
        sel = set(sel)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in sel]
            
            # Try adds
            for x in excluded:
                if can_add(x, sel):
                    sel.add(x)
                    improved = True
                    break
            if improved:
                continue
            
            # Try 2-for-1
            best_move = None
            best_gain = 0
            for x in list(sel):
                temp = sel - {x}
                added = []
                for y in excluded:
                    if len(added) >= 2:
                        break
                    if can_add(y, temp | set(added)):
                        added.append(y)
                if added:
                    gain = len(added) - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (x, added)
            
            if best_move and best_gain > 0:
                x, added = best_move
                sel.remove(x)
                sel.update(added)
            elif best_move and best_gain == 0 and random.random() < 0.2:
                x, added = best_move
                sel.remove(x)
                sel.update(added)
            else:
                break
        return sel
    
    # Compute degree: number of bad triples each element is in
    degree = [0] * (N + 1)
    for (a, b, c) in bad_triples_set:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Strategy 1: Greedy by degree
    order1 = sorted(range(1, N + 1), key=lambda x: degree[x])
    s1 = greedy_order(order1)
    s1 = polish(s1)
    s1 = local_search(s1)
    s1 = polish(s1)
    best_set = s1
    best_size = len(s1)
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 2: Reverse degree
    order2 = sorted(range(1, N + 1), key=lambda x: -degree[x])
    s2 = greedy_order(order2)
    s2 = polish(s2)
    s2 = local_search(s2)
    s2 = polish(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 3: Multiple randomized greedy trials
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        scores = []
        for x in range(1, N + 1):
            score = degree[x] + random.uniform(0, 5)
            scores.append((score, x))
        scores.sort()
        order = [x for _, x in scores]
        s3 = greedy_order(order)
        s3 = polish(s3)
        s3 = local_search(s3, max_iters=80)
        s3 = polish(s3)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 4: Stronger local search on best
    s4 = local_search(best_set, max_iters=400)
    s4 = polish(s4)
    if len(s4) > best_size:
        best_size = len(s4)
        best_set = s4
    
    # Strategy 5: Perturbation + local search
    for _ in range(50):
        if time.time() > deadline:
            break
        perturbed = set(best_set)
        n_remove = random.randint(1, min(3, len(perturbed)))
        to_remove = random.sample(list(perturbed), n_remove)
        for x in to_remove:
            perturbed.discard(x)
        perturbed = local_search(perturbed, max_iters=150)
        perturbed = polish(perturbed)
        if len(perturbed) > best_size:
            best_size = len(perturbed)
            best_set = perturbed
    
    # Strategy 6: Backtracking search with pruning
    # Try to find a set larger than best_size using backtracking
    target = best_size + 1
    if time.time() < deadline - 20:
        # Order elements by degree (ascending) for better pruning
        bt_order = sorted(range(1, N + 1), key=lambda x: degree[x])
        bt_order = bt_order[:60]  # Limit to top 60 for tractability
        
        def backtrack(idx, sel, best_sel):
            if time.time() > deadline - 5:
                return
            if len(sel) + (len(bt_order) - idx) <= best_size:
                return  # Can't beat best
            if len(sel) > best_size:
                best_sel.clear()
                best_sel.update(sel)
                best_size[0] = len(sel)
            
            if idx == len(bt_order):
                return
            
            x = bt_order[idx]
            # Try including x
            if can_add(x, sel):
                sel.add(x)
                backtrack(idx + 1, sel, best_sel)
                sel.remove(x)
            # Try excluding x
            backtrack(idx + 1, sel, best_sel)
        
        best_sel = set()
        best_size_wrapper = [best_size]
        backtrack(0, set(), best_sel)
        if best_sel and len(best_sel) > best_size:
            best_set = best_sel
            best_size = len(best_set)
            best_set = polish(best_set)
            best_set = local_search(best_set, max_iters=100)
            best_set = polish(best_set)
            best_size = len(best_set)
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
