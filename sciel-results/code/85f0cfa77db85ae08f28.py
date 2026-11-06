import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 180
    
    # 1. Precompute the conflict graph
    # We need a set of 3 distinct elements {a, b, c} such that 1/a = 1/b + 1/c.
    # This implies a < b and a < c.
    # c = ab / (b-a). So (b-a) must divide ab.
    # Since gcd(b-a, a) = gcd(b, a), let g = gcd(a, b).
    # b-a = g * k. Then c = a(b-a)/k ... wait.
    # c = ab / (b-a). Let d = b-a. c = ab/d.
    # For c to be integer, d must divide ab.
    
    # We represent the graph with N nodes (0 to N-1 corresponding to 1 to N).
    # An edge (u, v) exists if there exists w such that {u, v, w} is a Schur triple.
    # Actually, the constraint is "no THREE DISTINCT elements".
    # So if we pick {u, v}, we cannot pick any w that completes the triple with u and v.
    # This is a 3-uniform hypergraph independent set problem.
    # However, we can convert it to a graph MIS problem by noting that:
    # If we include u and v, we must exclude all w in Conflict(u, v).
    # This is not a standard graph MIS because the conflict depends on pairs.
    
    # BUT, we can use a different perspective.
    # Let's stick to the hypergraph approach but use a specific search strategy.
    # Or, simpler: The current code uses a suffix heuristic.
    # A better heuristic for "dense" sets in [1, N] avoiding reciprocal relations
    # is often found by starting with a random permutation and greedily adding,
    # OR by using a SAT solver approach (DPLL).
    
    # Given the time limit (90s) and N=180, a full exponential search is risky.
    # However, the "suffix" strategy found ~140 elements (score 1.65).
    # Let's try a **Multi-Start Local Search with Bitmasks** which is much faster than set operations.
    
    # Precompute conflict masks for each pair? No, that's N^2 * N bits = 180^3 bits ~ 4MB. Feasible.
    # But checking validity of a set S is: for all (u, v) in S, (S & ConflictMask[u][v]) == 0?
    # No, that's O(|S|^2).
    
    # Let's use the property: 1/a = 1/b + 1/c <=> c = ab/(b-a).
    # We can precompute for each pair (a, b) with a < b, the unique c (if integer and <= N).
    # If c exists and c != a, c != b, then {a, b, c} is a forbidden triple.
    
    # Strategy: 
    # 1. Generate a random permutation of [1..N].
    # 2. Greedily build a set S. When considering x, check if adding x creates a triple with any two existing elements.
    # 3. To make this fast, we can precompute for each x, the list of pairs (a, b) such that {a, b, x} is a triple.
    #    If S contains a and b, we cannot add x.
    
    # Let's precompute `conflicts_with_x[x]` = list of pairs (a, b) that form a triple with x.
    # Actually, for a fixed x, the triples are (a, b, x) where 1/a = 1/b + 1/x => 1/b = 1/a - 1/x = (x-a)/(ax) => b = ax/(x-a).
    # So for a fixed x, and a < x, if (x-a) divides ax, then b = ax/(x-a) is an integer.
    # We need b != x, b != a.
    
    # Let's build a structure: 
    # `pair_conflict[a][b]` = c, if {a,b,c} is a triple.
    # Then, when adding x to S, we must ensure that for all a, b in S, pair_conflict[a][b] != x.
    # This check is O(|S|^2). With |S| ~ 140, that's ~20k checks per insertion.
    # 180 insertions * 20k = 3.6M ops per run. Very fast.
    
    # To improve the result, we can do:
    # 1. Multiple random restarts with different permutations.
    # 2. Local search: Try to swap an element out and two in? Or just re-order the greedy process.
    
    # Let's implement a fast greedy with random restarts.
    
    # Precomputation
    N_val = 180
    # pair_conflict[a][b] = c (1-indexed) or 0 if none
    # We only need a < b.
    pair_conflict = [[0] * (N_val + 1) for _ in range(N_val + 1)]
    
    for a in range(1, N_val + 1):
        for b in range(a + 1, N_val + 1):
            d = b - a
            if d > 0 and (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N_val and c != a and c != b:
                    pair_conflict[a][b] = c
                    # Note: The triple is {a, b, c}. The equation is 1/a = 1/b + 1/c.
                    # This implies a < b and a < c? 
                    # 1/a = 1/b + 1/c > 1/b => a < b.
                    # 1/a = 1/b + 1/c > 1/c => a < c.
                    # So a is always the smallest.
                    # My loop has a < b. c can be anything > a.
                    # If c < b, the triple is {a, c, b}.
                    # If c > b, the triple is {a, b, c}.
                    # The check `pair_conflict[a][b] == x` works regardless of order of b and c,
                    # as long as we check all pairs in S.
    
    def is_valid_addition(S_set, x):
        """Check if adding x to S_set creates a forbidden triple."""
        # We need to check if there exist a, b in S_set such that pair_conflict[a][b] == x 
        # OR pair_conflict[b][a] == x (but we only store a<b, so we check min/max).
        # Actually, if {u, v, x} is a triple, then one of them is the smallest.
        # Let's just iterate through all pairs in S_set.
        # Optimization: Iterate through elements in S_set.
        # For each a in S_set:
        #   For each b in S_set where b > a:
        #      if pair_conflict[a][b] == x: return False
        #   Also check if x is the smallest?
        #   If x is smallest, then we need a, b in S_set such that pair_conflict[x][a] == b or pair_conflict[x][b] == a.
        #   Wait, pair_conflict is defined for (a, b) with a < b.
        #   If x is the smallest, then the triple is (x, a, b) with x < a < b or x < b < a.
        #   In our precomputed table, if x is smallest, it would be stored as pair_conflict[x][a] = b (if a < b) or pair_conflict[x][b] = a (if b < a).
        #   So we just need to check:
        #   1. Is there a pair (u, v) in S_set such that pair_conflict[min(u,v)][max(u,v)] == x?
        #   2. Is there a u in S_set such that pair_conflict[x][u] is in S_set? (Case x is smallest)
        
        # Let's combine:
        # Check case 1: x is the largest (or middle).
        # Iterate u, v in S_set.
        # This is O(K^2).
        
        # Check case 2: x is the smallest.
        # Iterate u in S_set.
        # Check if pair_conflict[x][u] is in S_set.
        # This is O(K).
        
        # Total O(K^2).
        
        S_list = list(S_set)
        K = len(S_list)
        
        # Case 2: x is smallest
        for u in S_list:
            if u == x: continue
            # We need v such that {x, u, v} is a triple.
            # If x < u, check pair_conflict[x][u].
            # If u < x, check pair_conflict[u][x].
            if x < u:
                v = pair_conflict[x][u]
                if v != 0 and v in S_set and v != x and v != u:
                    return False
            else: # u < x
                v = pair_conflict[u][x]
                if v != 0 and v in S_set and v != x and v != u:
                    return False
        
        # Case 1: x is not smallest. So x is larger than at least one element in the triple.
        # The triple is {a, b, c} with a < b, a < c.
        # If x is b or c, then a < x.
        # So there exists a in S_set with a < x, and another element y in S_set (y != a, y != x)
        # such that {a, x, y} is a triple.
        # This means pair_conflict[min(a, x)][max(a, x)] == the third one.
        
        # Simpler: Just check all pairs in S_set.
        # If pair_conflict[u][v] == x, invalid.
        for i in range(K):
            u = S_list[i]
            for j in range(i + 1, K):
                v = S_list[j]
                if u < v:
                    if pair_conflict[u][v] == x:
                        return False
                else:
                    if pair_conflict[v][u] == x:
                        return False
        
        return True

    best_set = set()
    best_size = 0
    
    # Run multiple random restarts
    # Budget: 80 seconds
    end_time = start_time + 80.0
    
    while time.time() < end_time:
        elements = list(range(1, N_val + 1))
        random.shuffle(elements)
        
        current_set = set()
        
        for x in elements:
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # If we found a very good set, we might want to try local search on it
            # But for now, just keep restarting.
            
    # One final check to ensure validity (paranoia)
    # Actually, is_valid_addition guarantees validity during construction.
    
    return list(best_set)

# EVOLVE-BLOCK-END
