import math
import time
import random
from collections import deque

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 256
    max_sum = 2 * n  # 512
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must not be a square
    # Filter candidates
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    m = len(candidates)
    
    # Map value -> index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build conflict bitmasks
    # conflict[i] is a bitmask of indices j > i (or all j != i) that conflict with i
    # For B&B, we need: for each vertex, which other vertices conflict
    conflict = [0] * m  # bitmask: bit j set means vertex j conflicts with vertex i
    
    # For each square sum s, all pairs (a,b) with a+b=s conflict
    # Also diagonal already handled by filtering
    for s in squares:
        # Find all pairs (a,b) in candidates with a + b = s, a <= b
        # a ranges from max(1, s-n) to min(n, s-1)
        a_lo = max(1, s - n)
        a_hi = min(n, s - 1)
        # Collect all candidate values that can form a pair summing to s
        vals_in_square = []
        for a in range(a_lo, a_hi + 1):
            b = s - a
            if a <= b and a in val_to_idx and b in val_to_idx:
                vals_in_square.append((a, b))
        # All these pairs conflict with each other (clique)
        # For each pair (a,b), a conflicts with b
        # Also, if multiple pairs share a vertex, that vertex conflicts with all others in the square
        # Actually, the conflict is: for square s, the set of all candidate values v where there exists w in candidates with v+w=s
        # forms a clique? No. Only pairs that sum to s conflict.
        # Two values a and b conflict if a+b is a square.
        # So for each square s, for each pair (a,b) with a+b=s, mark conflict.
        
        # More efficiently: for each candidate a, if s-a is also a candidate and s-a >= a, they conflict
        for a in range(a_lo, a_hi + 1):
            b = s - a
            if a > b:
                break
            ia = val_to_idx.get(a)
            ib = val_to_idx.get(b)
            if ia is not None and ib is not None and ia != ib:
                conflict[ia] |= (1 << ib)
                conflict[ib] |= (1 << ia)
            elif ia is not None and ib is not None and ia == ib:
                # a == b, diagonal already filtered, so this shouldn't happen
                pass

    # For B&B, we want: given a set of available vertices (bitmask), 
    # and we pick vertex v, the new available set removes v and all its neighbors.
    # neighbor_mask[v] = conflict[v] (all vertices conflicting with v)
    
    # Upper bound via clique cover: 
    # The graph is the union of cliques (one per square sum).
    # For each square s, let C_s be the set of candidate vertices that have a partner summing to s.
    # In any independent set, at most 1 vertex from C_s can be chosen? No!
    # Wait: if a and b are in C_s (a+b=s), they conflict. But a and c where a+c is a different square don't conflict via s.
    # The clique is: all pairs (a,b) with a+b=s. But is {a,b,c} a clique if a+b=s, b+c=s? 
    # If a+b=s and b+c=s, then a=c. So the structure for a fixed s is a matching!
    # For a fixed square s, the conflicting pairs form a matching (each vertex has at most one partner in that square).
    # So the graph is a union of matchings (one per square).
    
    # This means the graph is a union of matchings, so it's a graph with bounded structure.
    # The chromatic number is at most the maximum degree + 1, but more importantly,
    # we can use the fact that each square contributes a matching.
    
    # Better upper bound: for each vertex, its degree. The independence number <= n - (sum of degrees)/2 ... no.
    # Use: alpha(G) <= n - tau(G) where tau is minimum vertex cover.
    # For bipartite graphs, tau = maximum matching (Kőnig's theorem).
    # Is our graph bipartite? Not necessarily.
    
    # Let's use a simpler approach: greedy coloring for upper bound, but with bitmasks for speed.
    
    # Precompute: for each vertex, sorted list of neighbors by index (for greedy coloring)
    # Actually, let's use a different strategy: 
    # Since the graph is sparse (union of ~20 matchings), the maximum degree is at most 20.
    # We can use a simple B&B with the following upper bound:
    # alpha <= number of vertices in the subgraph (trivial)
    # Or: alpha <= sum over connected components of alpha(component)
    
    # Let me reconsider. The key insight is that this is a sparse graph.
    # Let's use a randomized local search with bit-parallel operations, which should be fast.
    
    # Actually, let me implement a clean B&B with bitmasks:
    
    def solve_bnb(available_mask, time_budget):
        """Branch and bound to find max independent set.
        available_mask: bitmask of vertices still available.
        Returns (size, solution_mask)"""
        best_size = 0
        best_mask = 0
        
        # Greedy initial solution
        def greedy_solution(rem):
            sol = 0
            cnt = 0
            while rem:
                # Pick vertex with minimum degree in rem
                best_v = -1
                best_d = m + 1
                r = rem
                while r:
                    # Get lowest set bit
                    lb = r & (-r)
                    v = lb.bit_length() - 1
                    d = (conflict[v] & rem).bit_count()
                    if d < best_d:
                        best_d = d
                        best_v = v
                        if d == 0:
                            break
                    r ^= lb
                if best_v < 0:
                    break
                sol |= (1 << best_v)
                cnt += 1
                rem ^= (1 << best_v)
                rem &= ~conflict[best_v]
            return sol, cnt
        
        init_sol, init_cnt = greedy_solution(available_mask)
        best_size = init_cnt
        best_mask = init_sol
        
        def time_up():
            return time.time() - start > time_budget
        
        # Upper bound: for a set of available vertices, use greedy coloring
        def color_bound(rem):
            if rem == 0:
                return 0
            # Greedy coloring: assign colors, count max color
            # Order by degree descending
            vertices = []
            r = rem
            while r:
                lb = r & (-r)
                v = lb.bit_length() - 1
                vertices.append(v)
                r ^= lb
            # Sort by degree in rem (descending)
            vertices.sort(key=lambda v: -(conflict[v] & rem).bit_count())
            color_of = {}
            max_color = 0
            for v in vertices:
                used_colors = set()
                # Check neighbors already colored
                nb = conflict[v] & rem
                r2 = nb
                while r2:
                    lb2 = r2 & (-r2)
                    u = lb2.bit_length() - 1
                    if u in color_of:
                        used_colors.add(color_of[u])
                    r2 ^= lb2
                c = 0
                while c in used_colors:
                    c += 1
                color_of[v] = c
                max_color = max(max_color, c + 1)
            return max_color
        
        def search(rem, current_size, current_mask):
            nonlocal best_size, best_mask
            if rem == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            if time_up():
                return
            ub = current_size + color_bound(rem)
            if ub <= best_size:
                return
            
            # Pick vertex with highest degree in rem
            best_v = -1
            best_d = -1
            r = rem
            while r:
                lb = r & (-r)
                v = lb.bit_length() - 1
                d = (conflict[v] & rem).bit_count()
                if d > best_d:
                    best_d = d
                    best_v = v
                r ^= lb
            
            if best_v < 0:
                return
            
            v_bit = 1 << best_v
            
            # Branch 1: include best_v
            new_rem_incl = (rem ^ v_bit) & ~conflict[best_v]
            if current_size + 1 + color_bound(new_rem_incl) > best_size:
                search(new_rem_incl, current_size + 1, current_mask | v_bit)
            
            # Branch 2: exclude best_v
            new_rem_excl = rem ^ v_bit
            if current_size + color_bound(new_rem_excl) > best_size:
                search(new_rem_excl, current_size, current_mask)
        
        search(available_mask, 0, 0)
        return best_size, best_mask
    
    def mask_to_set(mask):
        s = set()
        r = mask
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            s.add(candidates[v])
            r ^= lb
        return s
    
    # Full bitmask
    full_mask = (1 << m) - 1
    
    # Run B&B with time budget
    time_budget = deadline - 10.0
    best_size, best_mask = solve_bnb(full_mask, time_budget)
    best_solution = mask_to_set(best_mask)
    
    # Local search: try to improve
    def local_search(sol, time_budget):
        sol = set(sol)
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            # Try removing one and adding multiple
            for a in list(sol):
                if time.time() - start > time_budget:
                    break
                trial = sol - {a}
                # Find all candidates that can be added
                addable = []
                for b in candidates:
                    if b in trial:
                        continue
                    if 2 * b in squares:
                        continue
                    # Check if b conflicts with any in trial
                    ok = True
                    for c in trial:
                        if b + c in squares:
                            ok = False
                            break
                    if ok:
                        addable.append(b)
                # Greedily add
                added = []
                for b in addable:
                    if all(b + c not in squares for c in (trial | set(added))):
                        added.append(b)
                if len(trial) + len(added) > len(sol):
                    sol = trial | set(added)
                    improved = True
                    break
        return sol
    
    best_solution = local_search(best_solution, deadline - 5.0)
    
    # Randomized restarts
    if time.time() - start < deadline - 3.0:
        rng = random.Random(123)
        for restart in range(30):
            if time.time() - start > deadline - 3.0:
                break
            # Randomized greedy
            rem = set(candidates)
            r_sol = []
            while rem:
                # Randomly weighted by degree
                scored = []
                for node in rem:
                    d = len([x for x in rem if x + node in squares and x != node])
                    score = d + rng.random()
                    scored.append((score, node))
                scored.sort()
                best = scored[0][1]
                r_sol.append(best)
                rem.discard(best)
                rem = {x for x in rem if x + best not in squares}
            r_sol = set(r_sol)
            r_sol = local_search(r_sol, deadline - 3.0)
            if len(r_sol) > len(best_solution):
                best_solution = r_sol
    
    # Final verification
    ok = True
    lst = sorted(best_solution)
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    
    if not ok:
        # Fallback: deterministic greedy
        rem = set(candidates)
        fb = []
        while rem:
            mn = None
            md = None
            for node in rem:
                d = len([x for x in rem if x + node in squares and x != node])
                if md is None or d < md:
                    md = d
                    mn = node
            fb.append(mn)
            rem.discard(mn)
            rem = {x for x in rem if x + mn not in squares}
        best_solution = set(fb)
    
    return sorted(best_solution)
