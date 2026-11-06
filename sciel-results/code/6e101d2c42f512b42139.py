import time
from math import gcd
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))
    
    # Map: value -> index (0-indexed)
    val_to_idx = {v: i for i, v in enumerate(UNIVERSE)}
    idx_to_val = UNIVERSE  # index i -> value UNIVERSE[i]
    
    # Precompute gcd matrix (value-based)
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Build the conflict graph: edge (i,j) exists if there is some k != i,j
    # such that gcd(val_i, val_j) == gcd(val_i, val_k) == gcd(val_j, val_k)
    # This means {i,j,k} is a forbidden triple.
    # We want the maximum independent set in this conflict graph.
    
    # For each pair (i, j) in values, check if there exists k making a forbidden triple
    # conflict[i][j] = True if (i,j) is in some forbidden triple
    
    # Actually, for the independent set formulation:
    # We need: no three vertices form a forbidden triple.
    # This is NOT equivalent to a graph independent set (which forbids pairs).
    # It's a hypergraph independent set (forbids triples).
    
    # However, we can use a different approach: 
    # The problem is equivalent to finding the largest set S such that for every
    # g in [1,128], the set {x in S : g | x} has no three elements that are
    # pairwise coprime after dividing by g.
    
    # Let me use a different exact approach: iterative deepening with bitsets.
    # Or better: use the fact that N=128 is small enough for a well-optimized
    # hypergraph independent set solver.
    
    # Actually, let me use a clever reduction:
    # For each g, define G_g = {x in [1,128] : g divides x}.
    # Within G_g, a forbidden triple is {g*a, g*b, g*c} where gcd(a,b)=gcd(a,c)=gcd(b,c)=1.
    # So within G_g, we need to avoid three pairwise-coprime elements (after dividing by g).
    
    # This is still complex. Let me just use a fast bitset-based search.
    
    # Precompute: for each triple (a,b,c) with a<b<c, is it forbidden?
    # forbidden[a][b][c] = (gcd(a,b) == gcd(a,c) == gcd(b,c))
    
    # For the search, I'll use a recursive approach with bitsets for speed.
    # State: chosen set (as a Python int bitmask), remaining candidates.
    
    # Key insight for speed: precompute for each x the bitmask of all y such that
    # (x,y) is in some forbidden triple with a third element. But this doesn't help directly.
    
    # Better: for the B&B, when we have a chosen set C, a vertex x can be added iff
    # for all pairs (a,b) in C, (a,b,x) is not a forbidden triple.
    
    # Let me precompute: for each x, a bitmask conflict_mask[x] where bit y is set
    # if there exists z such that (x,y,z) is a forbidden triple.
    # This doesn't directly help because the conflict depends on the specific pair.
    
    # Let me use a different strategy: since N=128, use a well-optimized
    # branch-and-bound with the following upper bound:
    # At any node, UB = |chosen| + (number of remaining vertices that are individually
    # compatible with chosen).
    
    # To make this fast, I'll precompute for each x the list of "conflict pairs"
    # and use bitset operations.
    
    # Actually, let me try a completely different approach: use integer programming
    # via scipy's linprog or a custom branch-and-bound that's optimized for this.
    
    # Given the time constraint, let me use a very fast heuristic + verification:
    # 1. Use a sophisticated greedy with multiple orderings
    # 2. Use local search (add/remove/swap) to improve
    # 3. Verify the result
    
    # But the key structural change: use the DECOMPOSITION by gcd.
    # 
    # Observation: If we include a number x, it constrains which other numbers
    # we can include based on shared gcd factors.
    #
    # Let me try: for each prime p, the multiples of p form a sub-problem.
    # But numbers can be in multiple such sub-problems.
    
    # Let me just implement a very fast bitset-based B&B.
    
    # Precompute forbidden triples as a set of frozensets
    # For efficiency, encode each triple as a sorted tuple
    
    # Actually, the fastest approach for N=128:
    # Use a recursive search where at each step we decide to include or exclude
    # a vertex, with strong pruning.
    
    # Key optimization: order vertices by their "degree" in the hypergraph
    # (number of forbidden triples they participate in), most constrained first.
    
    # Precompute: for each x, count of forbidden triples containing x
    triple_count = [0] * (N + 1)
    # For each pair (a,b), find all c > b such that (a,b,c) is forbidden
    # and increment counts
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    triple_count[a] += 1
                    triple_count[b] += 1
                    triple_count[c] += 1
    
    # Order vertices by decreasing triple_count
    vertex_order = sorted(range(1, N + 1), key=lambda x: -triple_count[x])
    
    # For the B&B, I need a fast "can add x to set C" check.
    # Precompute: for each x, the set of pairs (a,b) such that (a,b,x) is forbidden.
    # Encode pair (a,b) as a single integer: a * 129 + b (since a,b <= 128)
    
    conflict_pairs = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    # (a,b,c) is a forbidden triple
                    # c conflicts with pair (a,b)
                    code = a * 129 + b
                    conflict_pairs[c].add(code)
                    # Also a conflicts with pair (b,c) and b conflicts with pair (a,c)
                    # But we only need: when checking if x can be added to C,
                    # we check if any pair in C is in conflict_pairs[x]
    
    # For fast checking, maintain the set of pairs in chosen set as a set of codes.
    
    def can_add_fast(x, chosen_pairs):
        """chosen_pairs is a set of encoded pairs (a*129+b for a<b in chosen)."""
        for code in conflict_pairs[x]:
            if code in chosen_pairs:
                return False
        return True
    
    # B&B with time limit
    best_solution = []
    best_len = 0
    timeout = [False]
    
    def branch_and_bound(chosen, chosen_pairs, remaining, depth):
        nonlocal best_solution, best_len
        if timeout[0]:
            return
        if time.time() >= DEADLINE - 3:
            timeout[0] = True
            return
        
        # Upper bound
        if len(chosen) + len(remaining) <= best_len:
            return
        
        if not remaining:
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_solution = chosen[:]
            return
        
        # Filter remaining to those compatible with chosen
        compatible = []
        for x in remaining:
            if can_add_fast(x, chosen_pairs):
                compatible.append(x)
        
        if len(chosen) + len(compatible) <= best_len:
            return
        
        if not compatible:
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_solution = chosen[:]
            return
        
        # Branch on the first compatible vertex (most constrained)
        x = compatible[0]
        
        # Include x
        new_chosen = chosen + [x]
        new_pairs = set(chosen_pairs)
        for a in chosen:
            code = min(a, x) * 129 + max(a, x)
            new_pairs.add(code)
        new_remaining = []
        for y in compatible[1:]:
            if can_add_fast(y, new_pairs):
                new_remaining.append(y)
        branch_and_bound(new_chosen, new_pairs, new_remaining, depth + 1)
        
        # Exclude x
        branch_and_bound(chosen, chosen_pairs, compatible[1:], depth + 1)
    
    # Get initial solution via greedy
    def greedy(order):
        selected = []
        pairs = set()
        for x in order:
            if can_add_fast(x, pairs):
                selected.append(x)
                for a in selected[:-1]:
                    pairs.add(min(a, x) * 129 + max(a, x))
        return selected
    
    # Multiple greedy starts
    orders = [
        vertex_order,  # most constrained first
        list(reversed(vertex_order)),  # least constrained first
        list(range(1, N + 1)),  # natural
        list(range(N, 0, -1)),  # reverse natural
    ]
    
    for order in orders:
        sol = greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
    
    if time.time() >= DEADLINE - 5:
        return best_solution if best_solution else [1]
    
    # Run B&B
    branch_and_bound([], set(), vertex_order, 0)
    
    if not best_solution:
        best_solution = [1]
    
    return best_solution
