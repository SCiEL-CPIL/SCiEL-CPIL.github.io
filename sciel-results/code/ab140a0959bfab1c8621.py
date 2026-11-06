import time
import random
import math
import heapq
from functools import lru_cache

def construct():
    start_time = time.time()
    limit = 85.0 # Safety margin for 90s limit
    
    n = 256
    
    # Weights: 1/i
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Precompute conflicts: conflict[i] is a bitmask of numbers that conflict with i
    # Two numbers a, b conflict if there exists c such that a,b,c form a 3-AP.
    # The third term c can be:
    # 1. 2a - b (if a is middle)
    # 2. 2b - a (if b is middle)
    # 3. (a+b)/2 (if a,b are ends, must be integer)
    # Actually, if we are building a set S, we cannot add x if it creates a 3-AP with any pair in S.
    # It is easier to define edges in a conflict graph: (a,b) is an edge if they cannot both be in S.
    # a, b cannot be in S if there exists c such that {a,b,c} is a 3-AP.
    # This happens if:
    # - 2a - b is in [1, n] (a is middle, b is end, c=2a-b) -> wait, if c is in S, then a,b,c is AP.
    #   So if we have a and b, and c=2a-b is also in S, it's bad.
    #   But in MWIS formulation, we just need to know if a and b are "incompatible".
    #   Two vertices a, b are incompatible if they form part of a 3-AP with *any* integer c.
    #   Wait, the constraint is "no 3-term AP".
    #   If we pick a and b, do they *force* a conflict? No.
    #   They conflict if there is a c such that a,b,c form a 3-AP.
    #   If we pick a and b, and later pick c=2a-b, we have a conflict.
    #   So a and b are not inherently incompatible unless we consider the third vertex.
    #   
    #   CORRECTION:
    #   A set S is 3-AP free if for any a,b in S, the third term c of the AP containing a,b is NOT in S.
    #   This is NOT a simple graph independent set problem on the vertices 1..n directly with static edges.
    #   Because whether a and b can coexist depends on whether the third term is present.
    #   
    #   HOWEVER, we can model this as an Independent Set in a graph where edges represent "forbidden pairs"?
    #   No. If a and b are in S, c=2a-b is forbidden.
    #   This is a hypergraph constraint (size 3).
    #   
    #   BUT, we can transform this.
    #   If we fix the set S, it is 3-AP free.
    #   Consider the graph G where vertices are 1..n.
    #   Edge (a,b) exists if 2a-b is in [1,n] OR 2b-a is in [1,n] OR (a+b)/2 is in [1,n] (integer).
    #   This is not quite right either.
    #   
    #   Let's stick to the definition:
    #   S is valid if for all distinct a,b in S, the third term c is not in S.
    #   
    #   Branch and Bound approach:
    #   We build S incrementally.
    #   State: (current_set_S, candidate_set_C)
    #   C contains elements that can potentially be added to S without violating constraints with S.
    #   When we add x to S, we must remove from C any y such that x and y form a partial AP that would be completed by some z in C?
    #   No, simpler:
    #   If we add x to S, any y in C such that (x,y) implies a third term z must be checked.
    #   Actually, if we add x, we must ensure that no y in S forms a 3-AP with x and some z in S.
    #   This requires checking against ALL existing elements in S.
    #   
    #   Alternative: Backtracking with bitmasks.
    #   n=256 is too large for 2^n.
    #   But the density of 3-APs is low?
    #   Number of 3-APs in [1, 256] is roughly sum_{d=1}^{128} (256 - 2d) ~ 128^2/2 ~ 8000.
    #   This is a sparse constraint set.
    #   
    #   Let's use a standard MWIS solver structure adapted for this specific constraint.
    #   We can use a recursive search where we decide for each number 1..n (in some order) whether to include it.
    #   Order: Greedy by weight (1, 2, 3...) is bad for pruning.
    #   Better order: High weight first (1, 2, 3...) is actually good for finding good solutions early.
    #   
    #   Pruning:
    #   Upper bound = current_weight + sum(weights of candidates that can still be added).
    #   To compute "can still be added", we need to know which candidates are compatible with the current S.
    #   If we add x, we must remove from candidates any y such that x, y, and some z in S form a 3-AP?
    #   No, if x and y are in S, we just need to ensure z is NOT in S.
    #   If z is already in S, then x and y cannot BOTH be in S.
    #   So, if we have S, and we consider adding x, we must check if x conflicts with any pair in S.
    #   Conflict check: For each y in S, check if z=2x-y is in S. If yes, x cannot be added.
    #   Also check if z=2y-x is in S.
    #   Also check if z=(x+y)/2 is in S.
    #   
    #   This check is O(|S|).
    #   
    #   To prune the search tree:
    #   We can use a "greedy coloring" or "clique cover" bound on the remaining candidates?
    #   The remaining candidates form a graph where edges connect pairs that *cannot* both be added given the current S?
    #   If we add x, it forbids y if... wait.
    #   Adding x does not forbid y immediately. It only forbids y if there is a z in S such that x,y,z is AP.
    #   So the "conflict" between x and y is dynamic based on S.
    #   
    #   This makes standard MWIS graph algorithms hard to apply directly.
    #   
    #   However, we can use a simple branch and bound with strong ordering and a tight upper bound.
    #   Upper bound: sum of weights of all candidates that are individually compatible with S.
    #   This is a loose bound but easy to compute.
    #   
    #   Let's try a randomized restart with local search first, but with a better local search?
    #   The current local search is 1-swap.
    #   2-swap (remove 2, add 2) or 1-remove 2-add (remove 1, add 2) might be needed.
    #   
    #   Let's try implementing a simple Branch and Bound.
    #   State: (index i, current_set S, current_weight w)
    #   We iterate i from 1 to n.
    #   Option 1: Include i.
    #       Check if i is compatible with S.
    #       If yes, recurse(i+1, S U {i}, w + weights[i]).
    #   Option 2: Exclude i.
    #       Recurse(i+1, S, w).
    #   
    #   Pruning:
    #   If w + sum(weights[j] for j in i..n) <= best_found, prune.
    #   This bound is very loose.
    #   
    #   Improved Pruning:
    #   Maintain a list of "active" candidates.
    #   If we include i, we must remove from active candidates any j such that i, j, and some k in S form a 3-AP.
    #   Actually, if i is added, it doesn't remove j from the pool of *potential* additions unless j conflicts with i *given S*.
    #   But j might conflict with i in the future if another element is added?
    #   No, the constraint is local to the triple.
    #   If i and j are both in S, they are valid iff no k in S completes the AP.
    #   
    #   This structure is difficult for simple B&B.
    #   
    #   Let's go back to improving the Local Search.
    #   The current search does 1-add and 1-swap.
    #   Let's add 2-swap (remove 2, add 1) and 2-swap (remove 1, add 2).
    #   Removing 2 and adding 1 is bad for weight (1/a + 1/b > 1/c usually).
    #   Removing 1 and adding 2: 1/c + 1/d > 1/a? Possible if a is large and c,d are small.
    #   
    #   Also, the current `can_add` is O(|S|).
    #   We can speed up `can_add` using bitmasks.
    #   Precompute for each x a bitmask `conflict_mask[x]` of all y such that x and y are "dangerous"?
    #   No, as established, conflict depends on the third element.
    #   
    #   However, we can precompute for each x and each y in S, the third term z.
    #   If we maintain S as a bitmask, checking if z is in S is O(1).
    #   Checking compatibility of x with S:
    #   For each y in S:
    #     z1 = 2x - y. If z1 in S, fail.
    #     z2 = 2y - x. If z2 in S, fail.
    #     z3 = (x+y)/2. If z3 in S, fail.
    #   
    #   This is O(|S|). |S| is at most ~100-200.
    #   So `can_add` is fast.
    #   
    #   The bottleneck is the search space.
    #   
    #   Let's try a more aggressive local search:
    #   1. Random restarts.
    #   2. In each restart, do greedy add.
    #   3. Then do extensive local search:
    #      - Try all 1-swaps.
    #      - Try all 2-swaps (remove 1, add 2).
    #      - Try all 3-swaps?
    #   
    #   Given the 90s limit, we can afford many iterations.
    #   
    #   Let's implement a "Simulated Annealing" style search or just a very deep local search with random restarts.
    #   
    #   Actually, the directive says: "Model the problem as a maximum weight independent set... solve this exactly using a branch-and-bound algorithm".
    #   I should try to implement a B&B that works.
    #   
    #   Key insight for B&B:
    #   The graph of conflicts is not static.
    #   But we can define a static graph where (a,b) is an edge if they *can* form a 3-AP with *any* c in [1,n].
    #   If we take an Independent Set in this static graph, is it 3-AP free?
    #   No. If a,b are not connected by an edge, it means they don't form a 3-AP with ANY c.
    #   Wait, if a,b don't form a 3-AP with any c, then they are safe together?
    #   Yes. If {a,b,c} is a 3-AP, then c is determined by a,b.
    #   So a,b are "conflicting" in the static sense if the third term c exists in [1,n].
    #   If c does not exist in [1,n], then a,b can never be part of a 3-AP within [1,n].
    #   So we can define a static conflict graph G_static where edge (a,b) exists if the third term c of the AP containing a,b is in [1,n].
    #   
    #   Is an Independent Set in G_static always 3-AP free?
    #   Let S be an Independent Set in G_static.
    #   Suppose S contains a 3-AP (a,b,c).
    #   Then a,b form a 3-AP with c.
    #   Since c is in [1,n], the edge (a,b) must exist in G_static.
    #   But S is an Independent Set, so (a,b) cannot be in S.
    #   Contradiction.
    #   
    #   YES! The problem is exactly Maximum Weight Independent Set in a specific static graph G.
    #   Vertices: 1..256.
    #   Edge (a,b) exists if the third term c of the 3-AP containing a and b is in [1, 256].
    #   Note: If a=b, not an edge.
    #   If a != b:
    #   c1 = 2a - b. If 1 <= c1 <= 256, edge exists.
    #   c2 = 2b - a. If 1 <= c2 <= 256, edge exists.
    #   c3 = (a+b)/2. If (a+b) even and 1 <= c3 <= 256, edge exists.
    #   
    #   So we can build this graph and solve MWIS.
    #   
    #   Now, how to solve MWIS for n=256?
    #   Standard algorithms (like Bron-Kerbosch or B&B) work well if the graph is dense or has high clique number.
    #   This graph is relatively sparse.
    #   However, we can use a B&B with a good upper bound.
    #   Upper bound: Sum of weights of all vertices in the remaining subgraph?
    #   Better: Greedy Coloring bound.
    #   
    #   Let's implement a B&B for MWIS.
    #   1. Build adjacency list and bitmasks.
    #   2. Order vertices by weight (descending) or degree?
    #      Weight descending is good for finding high weight solutions early.
    #   3. Recursive function `search(idx, current_mask, current_weight)`.
    #      `current_mask` is the set of selected vertices.
    #      `idx` is the index in the ordered list.
    #      We consider vertices `v` from `idx` to `n`.
    #      For each `v`:
    #         If `v` is compatible with `current_mask` (i.e., no edge between `v` and any `u` in `current_mask`):
    #            Recurse with `v` added.
    #         Recurse with `v` excluded.
    #   
    #   Pruning:
    #   `current_weight` + `upper_bound(remaining_vertices)` <= `best_weight` -> Prune.
    #   
    #   `upper_bound` for remaining vertices (which form a subgraph induced by vertices compatible with `current_mask`):
    #   We can use a simple sum of weights, but it's loose.
    #   We can use a greedy coloring of the remaining graph to get a tighter bound.
    #   Color classes are independent sets. The sum of max weight in each color class is an upper bound?
    #   No, the sum of weights of all vertices in the largest independent set?
    #   Standard bound: Color the remaining graph. The number of colors is an upper bound on the size of the independent set? No, that's for Max Clique.
    #   For Max Independent Set, we can use the fact that any independent set can take at most one vertex from each clique.
    #   Or use fractional coloring?
    #   
    #   Simpler bound:
    #   For the remaining candidates `C`, compute a greedy independent set `I` and its weight `W_I`.
    #   This is a lower bound, not upper.
    #   
    #   Let's use a standard B&B for MWIS.
    #   Given the time limit, I will implement a robust B&B with:
    #   - Vertex ordering by weight.
    #   - Bitmask representation.
    #   - Pruning based on `current_weight + sum(weights of candidates)` (loose but fast).
    #   - If time allows, switch to a tighter bound.
    
    # Build Conflict Graph
    # adj_mask[i] is a bitmask of all j such that (i,j) is an edge in G_static.
    adj_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            conflict = False
            # Check 3rd terms
            c1 = 2 * i - j
            if 1 <= c1 <= n:
                conflict = True
            c2 = 2 * j - i
            if 1 <= c2 <= n:
                conflict = True
            if (i + j) % 2 == 0:
                c3 = (i + j) // 2
                if 1 <= c3 <= n:
                    conflict = True
            
            if conflict:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Weights
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Order vertices by weight descending
    # We want to pick high weight vertices first.
    order = list(range(1, n + 1))
    order.sort(key=lambda x: -weights[x])
    
    # Map original index to position in order?
    # Or just iterate through `order`.
    # In B&B, we typically fix an order and decide include/exclude for each.
    
    best_weight = 0.0
    best_mask = 0
    
    # Precompute suffix sums of weights for pruning?
    # suffix_sum[i] = sum of weights of vertices from order[i] to order[n-1]
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
        
    # B&B Search
    def search(pos, current_mask, current_weight):
        nonlocal best_weight, best_mask
        
        if time.time() - start_time > limit - 2:
            return
            
        # Pruning
        if current_weight + suffix_sum[pos] <= best_weight:
            return
            
        if pos == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_mask = current_mask
            return
            
        v = order[pos]
        
        # Option 1: Exclude v
        search(pos + 1, current_mask, current_weight)
        
        # Option 2: Include v
        # Check if v is compatible with current_mask
        # v is compatible if no neighbor of v is in current_mask
        # neighbors = adj_mask[v]
        # conflict = current_mask & adj_mask[v]
        if not (current_mask & adj_mask[v]):
            search(pos + 1, current_mask | (1 << v), current_weight + weights[v])
            
    # Start search
    search(0, 0, 0.0)
    
    # Extract solution
    S_set = set()
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            S_set.add(i)
            
    # If B&B didn't find anything (shouldn't happen, at least {1} is valid), fallback to greedy
    if not S_set:
        S_set = {1}
        
    return list(S_set)

def can_add(x, S_bits):
    """
    Check if x can be added to S (represented by S_bits) without creating a 3-AP.
    x is 1-indexed.
    """
    # Iterate through all y in S
    temp = S_bits
    while temp:
        low_bit = temp & -temp
        y = low_bit.bit_length() - 1
        
        # Check 1: y, x, z. z = 2x - y
        z = 2 * x - y
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 2: x, y, z. z = 2y - x
        z = 2 * y - x
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 3: x, z, y. z = (x+y)/2
        if (x + y) % 2 == 0:
            z = (x + y) // 2
            if 1 <= z <= 256:
                if S_bits & (1 << z):
                    return False
        
        temp &= temp - 1
        
    return True
