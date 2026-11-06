import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    N = 256
    
    # Build the 3-AP conflict graph
    # Two numbers conflict if they are part of a 3-AP with some third number
    # Actually, we need: a set S has no 3-AP means no three distinct elements a,b,c in S with a+c=2b
    # So the "conflict" is: if we have both a and c in S, we cannot have b=(a+c)/2 in S
    # For branch-and-bound on independent set: vertices are numbers 1..N
    # Edge between u and v if they "conflict" - but actually the constraint is on triples
    
    # Let's think differently: we want a subset S with no 3-AP.
    # This is equivalent to: for any a < b < c in S, we need a + c != 2b.
    # 
    # For branch-and-bound, we can use the following approach:
    # Order vertices by weight (1/x) descending.
    # Use a greedy coloring bound: color the remaining vertices such that
    # adjacent vertices have different colors. The number of colors gives
    # an upper bound on the size of an independent set in the remaining subgraph.
    # But we want max weight, not max size. So we use weighted coloring bound.
    
    # Actually, let's use a different approach: branch on whether to include or exclude
    # each vertex, ordered by weight. Use a greedy upper bound.
    
    # Precompute: for each pair (a, c), the middle b = (a+c)/2 if it's an integer
    # and in [1, N]. If a and c are in S, then b cannot be in S.
    
    # Let's precompute conflicts: for each x, which pairs (a, c) with a < c and a+c = 2x
    # If both a and c are in S, then x cannot be in S.
    # Also if x is in S, then for each such pair, at most one of a, c can be in S.
    
    # For the branch-and-bound, we'll use the following:
    # - Order vertices by 1/x descending (most valuable first)
    # - At each step, decide to include or exclude the next vertex
    # - Upper bound: sum of weights of all remaining vertices that can still be included
    #   (i.e., not conflicting with already-included vertices)
    
    # Let's precompute for each x, the set of vertices that conflict with x
    # Two vertices conflict if they form a 3-AP with some third vertex
    # Actually, the constraint is: if x is in S, then for any (a, c) with a+c=2x,
    # we cannot have both a and c in S.
    # This is not a simple pairwise conflict.
    
    # Alternative: model as maximum weight independent set in a graph where
    # vertices are 1..N, and there's an edge between u and v if u and v are
    # "incompatible" - i.e., there exists w such that {u,v,w} forms a 3-AP.
    # But that's not quite right either.
    
    # Let me reconsider. A set S has no 3-AP iff for all distinct a, b, c in S,
    # a + c != 2b.
    # 
    # This is equivalent to: S is an independent set in the following hypergraph:
    # vertices = {1, ..., N}, hyperedges = {a, b, c} where a+c = 2b and a < b < c.
    # We want a maximum weight set with no hyperedge fully contained.
    #
    # For branch-and-bound, we can use:
    # - Order by weight descending
    # - Branch: include or exclude current vertex
    # - Bound: sum of weights of vertices that can still be added without creating a 3-AP
    #   with already-included vertices
    
    # Precompute: for each x, the set of pairs (a, c) such that a + c = 2x, a < x < c
    # If x is included, then for each such pair, at most one of a, c can be included.
    
    # Let's precompute midpoints: for each x, list of (a, c) with a + c = 2x
    midpoints = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            c = 2 * x - a
            if c > x and c <= N:
                midpoints[x].append((a, c))
    
    # Also, for each x, precompute which other vertices would be "blocked" if x is in S
    # This is complex. Let's use a simpler bound.
    
    # Greedy upper bound: given a set of already-included vertices S_included,
    # and a set of remaining vertices R, compute an upper bound on the additional
    # weight we can get from R.
    # Simple bound: sum of 1/x for x in R that don't immediately conflict with S_included.
    # But "immediately conflict" is hard to define for 3-AP.
    
    # Better bound: use a greedy algorithm to find a feasible subset of R,
    # and use its weight as a lower bound, but we need an upper bound.
    
    # Let's use the following approach:
    # 1. Start with a good initial solution using greedy + local search
    # 2. Use branch-and-bound with the following bound:
    #    - Sort remaining vertices by weight descending
    #    - For each remaining vertex, check if it can be added to S_included
    #      without creating a 3-AP. If yes, add it to the bound sum.
    #    - This gives a valid upper bound (it's actually the weight of a feasible
    #      solution, so it's a lower bound on the optimum, not upper).
    
    # I'm overcomplicating this. Let me use a different strategy:
    # Use a weighted maximum independent set formulation where we define
    # conflicts properly.
    
    # Define: two numbers u and v conflict if there exists w such that
    # {u, v, w} forms a 3-AP. Then S has no 3-AP iff S is an independent set
    # in this conflict graph.
    
    # Check: if u, v, w form a 3-AP with u < v < w, then u+w = 2v.
    # In the conflict graph, we'd have edges (u,v), (v,w), (u,w).
    # If S contains u, v, w, then all three edges are violated.
    # But if S contains only u and v (not w), is that allowed? Yes, as long as
    # there's no w in S with u+w=2v.
    # So the conflict graph approach doesn't work directly.
    
    # Let me just use the original local search approach but with better initialization
    # and more aggressive local search.
    
    # Precompute conflict_pairs[x] = list of (a, b) pairs such that {a, b, x} is a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x, N - x + 1)):
            a, b = x - d, x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is largest: (x-2d, x-d, x)
        for d in range(1, (x + 1) // 2 + 1):
            a, b = x - 2*d, x - d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is smallest: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            a, b = x + d, x + 2*d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))
    
    def can_add(x, in_set):
        """Check if x can be added given current set represented by bool array (1-indexed)."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True
    
    # Greedy initialization: add numbers in order of 1/x descending
    in_set = [False] * (N + 1)
    S = set()
    for x in range(1, N + 1):  # 1/x is largest for small x
        if can_add(x, in_set):
            in_set[x] = True
            S.add(x)
    
    def compute_score(S):
        return sum(1.0 / x for x in S)
    
    best_set = set(S)
    best_score = compute_score(best_set)
    best_in_set = in_set[:]
    
    # Local search with multiple strategies
    improved = True
    round_count = 0
    no_improve_rounds = 0
    
    while improved and time.time() < deadline:
        improved = False
        round_count += 1
        
        # Strategy 1: Remove-and-readd
        sorted_by_value = sorted(best_set, key=lambda x: -1.0/x)
        
        for x in sorted_by_value:
            if time.time() > deadline:
                break
            
            in_set_temp = best_in_set[:]
            in_set_temp[x] = False
            S_temp = best_set - {x}
            
            new_S = set(S_temp)
            new_in_set = in_set_temp[:]
            
            for y in range(1, N + 1):
                if new_in_set[y]:
                    continue
                if can_add(y, new_in_set):
                    new_in_set[y] = True
                    new_S.add(y)
            
            new_score = compute_score(new_S)
            if new_score > best_score + 1e-15:
                best_set = new_S
                best_score = new_score
                best_in_set = new_in_set
                improved = True
                no_improve_rounds = 0
        
        # Strategy 2: Single swap
        if time.time() < deadline:
            for x in sorted(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                best_y = -1
                best_new_score = best_score
                
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        new_score = best_score - 1.0/x + 1.0/y
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_y = y
                
                if best_y > 0 and best_new_score > best_score + 1e-15:
                    best_set = S_temp | {best_y}
                    best_score = best_new_score
                    best_in_set = in_set_temp[:]
                    best_in_set[best_y] = True
                    improved = True
                    no_improve_rounds = 0
        
        # Strategy 3: Remove two, re-fill
        if time.time() < deadline:
            elements = sorted(best_set, reverse=True)
            n_elem = len(elements)
            max_pairs_to_try = min(30, n_elem)
            
            for i in range(max_pairs_to_try):
                if time.time() > deadline:
                    break
                for j in range(i + 1, min(i + 10, n_elem)):
                    if time.time() > deadline:
                        break
                    x1, x2 = elements[i], elements[j]
                    
                    in_set_temp = best_in_set[:]
                    in_set_temp[x1] = False
                    in_set_temp[x2] = False
                    S_temp = best_set - {x1, x2}
                    
                    new_S = set(S_temp)
                    new_in_set = in_set_temp[:]
                    
                    for y in range(1, N + 1):
                        if new_in_set[y]:
                            continue
                        if can_add(y, new_in_set):
                            new_in_set[y] = True
                            new_S.add(y)
                    
                    new_score = compute_score(new_S)
                    if new_score > best_score + 1e-15:
                        best_set = new_S
                        best_score = new_score
                        best_in_set = new_in_set
                        improved = True
                        no_improve_rounds = 0
        
        # Strategy 4: Multi-add after removal
        if time.time() < deadline:
            for x in sorted(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                candidates = []
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        candidates.append(y)
                
                best_improvement = 0
                best_additions = []
                
                for y in candidates:
                    new_score = best_score - 1.0/x + 1.0/y
                    if new_score > best_score + best_improvement:
                        best_improvement = new_score - best_score
                        best_additions = [y]
                
                if len(candidates) > 1 and time.time() < deadline - 5:
                    for i in range(min(8, len(candidates))):
                        for j in range(i + 1, min(8, len(candidates))):
                            y1, y2 = candidates[i], candidates[j]
                            test_in_set = in_set_temp[:]
                            test_in_set[y1] = True
                            if can_add(y2, test_in_set):
                                new_score = best_score - 1.0/x + 1.0/y1 + 1.0/y2
                                if new_score > best_score + best_improvement:
                                    best_improvement = new_score - best_score
                                    best_additions = [y1, y2]
                
                if best_additions and best_improvement > 1e-15:
                    new_set = S_temp | set(best_additions)
                    new_in_set = in_set_temp[:]
                    for y in best_additions:
                        new_in_set[y] = True
                    best_set = new_set
                    best_score += best_improvement
                    best_in_set = new_in_set
                    improved = True
                    no_improve_rounds = 0
        
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds >= 3:
                break
    
    # Final polish
    if time.time() < deadline - 2:
        final_improved = True
        while final_improved and time.time() < deadline - 1:
            final_improved = False
            for x in sorted(best_set):
                if time.time() > deadline - 1:
                    break
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                new_S = set(S_temp)
                new_in_set = in_set_temp[:]
                
                for y in range(1, N + 1):
                    if new_in_set[y]:
                        continue
                    if can_add(y, new_in_set):
                        new_in_set[y] = True
                        new_S.add(y)
                
                new_score = compute_score(new_S)
                if new_score > best_score + 1e-15:
                    best_set = new_S
                    best_score = new_score
                    best_in_set = new_in_set
                    final_improved = True
    
    return list(best_set)
