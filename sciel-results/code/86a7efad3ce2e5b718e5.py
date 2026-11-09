import time
from math import gcd
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    # For each pair, find the gcd, then check all other numbers with same gcd to both
    # More efficient: for each gcd value g, collect all pairs with that gcd
    # Then any triple where all three pairs have gcd=g is bad
    
    # Build adjacency: for each pair (a,b), store gcd
    # For each g, build list of pairs with gcd g
    # Then for each g, any three elements that form a triangle in the graph of pairs-with-gcd-g is bad
    
    # Actually, let's just enumerate all triples directly - there are C(128,3) = 341,376 triples
    # This is manageable
    
    bad_triple_set = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            gab = gcd_table[a][b]
            for c in range(b+1, n+1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bad_triple_set.add((a, b, c))
    
    # For each vertex, compute which other vertices it's in a bad triple with
    # And more importantly, for branch-and-bound, we need:
    # For each vertex v, and each pair (u,w) not containing v, 
    # does adding v create a bad triple?
    
    # Better approach: For branch-and-bound on independent set in 3-uniform hypergraph:
    # State: (remaining_vertices, current_size)
    # When we decide to include vertex v, we must exclude any pair (u,w) 
    # such that (v,u,w) is a bad triple
    # When we decide to exclude v, we just remove it
    
    # For efficient pruning, precompute for each vertex v:
    # conflicts[v] = set of pairs (u,w) such that (v,u,w) is bad
    # This means if we include v, we can't include both u and w
    
    # Actually, let's think differently. Use a recursive backtracking with:
    # - Vertex ordering by degree (number of bad triples containing it)
    # - At each step, try including or excluding the next vertex
    # - Prune when current_size + remaining_count <= best_size
    
    # Precompute vertex degrees (number of bad triples containing each vertex)
    vertex_degree = [0]*(n+1)
    for (a,b,c) in bad_triple_set:
        vertex_degree[a] += 1
        vertex_degree[b] += 1
        vertex_degree[c] += 1
    
    # Order vertices by degree (most constrained first)
    vertices = list(range(1, n+1))
    vertices.sort(key=lambda v: vertex_degree[v], reverse=True)
    
    # For branch-and-bound, we need efficient conflict checking
    # When we have a set S of included vertices, and we want to add v:
    # v can be added iff for all u,w in S, (v,u,w) is NOT a bad triple
    # Equivalently, no bad triple contains v and two elements of S
    
    # Precompute for each v: bad_pairs_with_v[v] = set of (u,w) pairs 
    # such that (v,u,w) is a bad triple (with u<w, and u,w != v)
    bad_pairs_with_v = [set() for _ in range(n+1)]
    for (a,b,c) in bad_triple_set:
        bad_pairs_with_v[a].add((b,c))
        bad_pairs_with_v[b].add((a,c))
        bad_pairs_with_v[c].add((a,b))
    
    # Also precompute: for each pair (u,w), which v's make (v,u,w) bad
    # This helps when we have u,w in S and are considering v
    bad_v_for_pair = {}
    for (a,b,c) in bad_triple_set:
        for pair in [(b,c),(a,c),(a,b)]:
            if pair not in bad_v_for_pair:
                bad_v_for_pair[pair] = set()
            bad_v_for_pair[pair].add(a if pair==(b,c) else b if pair==(a,c) else c)
    
    # Actually, let's use a simpler but effective approach:
    # Greedy + local search with better initialization
    
    def is_valid_set(s):
        """Check if a set is valid."""
        lst = sorted(s)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i+1, m):
                b = lst[j]
                gab = gcd_table[a][b]
                for k in range(j+1, m):
                    c = lst[k]
                    if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                        return False
        return True
    
    def valid_addition(selected_list, x):
        """Check if adding x to selected_list maintains validity."""
        for i in range(len(selected_list)):
            a = selected_list[i]
            for j in range(i+1, len(selected_list)):
                b = selected_list[j]
                gab = gcd_table[a][b]
                if gcd_table[a][x] == gab and gcd_table[b][x] == gab:
                    return False
        return True
    
    # Branch and bound for maximum independent set
    # We'll use a recursive approach with memoization on the remaining set (as frozenset)
    # But that's too large. Instead, use greedy upper bound.
    
    def greedy_upper_bound(remaining_set):
        """Get an upper bound on the size of a valid set from remaining_set."""
        # Simple bound: just len(remaining_set)
        # Better: greedy independent set size as lower bound on what's achievable
        # But we need an upper bound for pruning.
        # The tightest simple upper bound is len(remaining_set)
        # A better bound: for each vertex in remaining, count how many other vertices
        # it conflicts with (in the sense that they can't both be in the independent set)
        # This is complex for 3-uniform hypergraphs.
        # Use len(remaining_set) as the bound.
        return len(remaining_set)
    
    # Let's use a different approach: iterative improvement with multiple strategies
    
    best_set = set()
    best_size = 0
    
    def try_improve(initial_set, rng):
        """Try to improve initial_set using local search."""
        nonlocal best_set, best_size
        
        current = set(initial_set)
        current_list = list(current)
        
        # First, try to add any valid elements
        added = True
        while added:
            added = False
            for x in range(1, n+1):
                if x not in current:
                    if valid_addition(current_list, x):
                        current.add(x)
                        current_list.append(x)
                        added = True
        
        # Try swaps: remove one, add one or more
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(current_list)):
                orig = current_list[i]
                trial = [x for j, x in enumerate(current_list) if j != i]
                trial_set = set(trial)
                
                # Try adding any single element
                for x in range(1, n+1):
                    if x not in trial_set:
                        if valid_addition(trial, x):
                            new_list = trial + [x]
                            if len(new_list) > len(current_list):
                                current = set(new_list)
                                current_list = new_list
                                improved = True
                                break
                if improved:
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        return current
    
    # Strategy 1: Random greedy starts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        rng = Random(restart * 1000003 + 7)
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        current = set()
        current_list = []
        for x in candidates:
            if valid_addition(current_list, x):
                current.add(x)
                current_list.append(x)
        
        try_improve(current, rng)
    
    # Strategy 2: Structured starts
    # Numbers that are powers of 2 times a prime > 2
    # Or numbers with distinct prime factorization patterns
    
    # Try: all numbers that are 1 mod some value
    for mod in [2, 3, 4, 5, 6, 7]:
        if time.time() > deadline - 5:
            break
        seed = set(x for x in range(1, n+1) if x % mod == 1)
        if seed:
            try_improve(seed, Random(mod * 1000))
    
    # Try: numbers with exactly k prime factors
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    for k in [0, 1, 2, 3]:
        if time.time() > deadline - 5:
            break
        seed = set(x for x in range(1, n+1) if count_prime_factors(x) == k)
        if seed:
            try_improve(seed, Random(k * 1000 + 42))
    
    # Strategy 3: Use the bad triple structure to guide selection
    # Vertices with low degree (in few bad triples) are "safer" to include
    # Greedy: always pick the vertex with lowest degree among remaining valid choices
    
    if time.time() < deadline - 5:
        remaining = set(range(1, n+1))
        current = set()
        current_list = []
        
        while remaining:
            # Find the vertex in remaining that can be added and has lowest degree
            best_candidate = None
            best_deg = float('inf')
            
            for v in remaining:
                if valid_addition(current_list, v):
                    deg = vertex_degree[v]
                    if deg < best_deg:
                        best_deg = deg
                        best_candidate = v
            
            if best_candidate is None:
                break
            
            current.add(best_candidate)
            current_list.append(best_candidate)
            remaining.discard(best_candidate)
        
        try_improve(current, Random(999999))
    
    # Final verification and return
    if not best_set:
        return [1]
    
    # Verify the result is valid
    if not is_valid_set(best_set):
        # Fallback to a known valid set
        return [1]
    
    return sorted(best_set)
