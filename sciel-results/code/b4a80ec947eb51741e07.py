import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    N = 256

    # Build conflict graph: edge (a,b) if a*b divisible by a+b
    # vertices 0..255 represent numbers 1..256
    # Use bitmasks (Python ints) for adjacency
    adj = [0] * N
    
    # Optimization: Use number theory to find conflicts faster than O(N^2)
    # Condition: (a*b) % (a+b) == 0
    # Let d = gcd(a, a+b). Then d | a and d | a+b => d | b.
    # So d = gcd(a, b).
    # (a*b) / (a+b) must be integer.
    # Let k = (a+b)/gcd(a,b). Then a*b/k must be integer.
    # Actually, simpler: (a*b) % (a+b) == 0 iff there exists integer m such that a*b = m*(a+b).
    # b(a-m) = ma => b = ma / (a-m).
    # For fixed a, m must be such that a-m divides a, and b is integer in range.
    # Also b != a.
    # Let's stick to the O(N^2) for N=256 as it's ~32k checks, which is negligible.
    # The inspirations suggest O(N^2) is fine or slightly optimized.
    # 256*256/2 = 32768 iterations. Python can do ~10M-50M simple ops/sec.
    # This takes < 0.1s.
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Precompute vertex bits for convenience
    # v_bit[v] = 1 << v
    
    # Greedy coloring of a set (bitmask) to get upper bound on independent set size
    def greedy_color_bound(cand):
        """Returns upper bound on independent set size using greedy coloring."""
        if cand == 0:
            return 0
        
        colors = 0
        remaining = cand
        
        while remaining:
            colors += 1
            color_class = 0
            rem = remaining
            # Greedy: pick vertices that don't conflict with current color class
            # To make the bound tighter, we should pick vertices carefully.
            # Standard greedy: iterate through remaining, add to color if possible.
            # Order matters. A common heuristic is to pick the vertex with max degree in remaining first?
            # Or just lowest index? 
            # For upper bound, we want to use as FEW colors as possible.
            # This is equivalent to finding a large independent set in the complement? No.
            # Greedy coloring: 
            # 1. Pick a vertex v from remaining.
            # 2. Add v to current color class.
            # 3. Remove v from remaining.
            # 4. Remove neighbors of v from remaining? NO. 
            #    Wait, in coloring, we remove v from remaining. 
            #    The next vertex picked cannot be a neighbor of v if we want to keep it in the SAME color? 
            #    No, if we are forming color classes, vertices in the same class must be independent.
            #    So if we pick v for color C, we cannot pick any neighbor of v for color C.
            #    So we should remove neighbors of v from the set of candidates for color C.
            
            # Let's refine the greedy coloring loop:
            # We want to partition 'remaining' into max independent sets (color classes).
            # Actually, greedy coloring usually goes:
            # For each vertex in some order, assign the smallest available color.
            # This gives a valid coloring. The number of colors used is an upper bound on chromatic number, 
            # and thus on the size of the independent set? 
            # NO. Chromatic number chi(G) >= alpha(G) is FALSE. 
            # Actually, alpha(G) * chi(G) >= |V|. So chi(G) <= |V| / alpha(G). 
            # This doesn't give a direct upper bound on alpha(G) from chi(G) easily without knowing |V| relation.
            # Wait, the standard B&B for MIS uses:
            # Upper bound = size of the largest color class? No.
            # The standard bound is: If we color the graph with k colors, then the MIS can have at most 1 vertex from each color class?
            # NO. An independent set can have multiple vertices from the same color class? 
            # No, vertices in the same color class are NOT adjacent. So they CAN be in the same independent set.
            # So the number of colors is NOT an upper bound on the MIS size.
            
            # CORRECTION:
            # The upper bound used in Tomita's algorithm is based on coloring.
            # If the graph can be colored with k colors, the MIS size is NOT bounded by k.
            # The bound is: The MIS size is at most the sum of the sizes of the color classes? No.
            
            # Let's re-read the standard MIS B&B bound.
            # The bound is derived from the fact that if we have a coloring, 
            # we can estimate the maximum number of vertices we can pick from 'remaining'.
            # Actually, the common bound is: 
            # Greedy coloring produces color classes C1, C2, ..., Ck.
            # The MIS can contain at most 1 vertex from each "conflict set" if the sets were cliques.
            # But color classes are independent sets.
            
            # Let's look at the current program's logic.
            # It returns `len(colors)`. 
            # Is `len(colors)` an upper bound on MIS?
            # Example: Path P3 (1-2-3). MIS size is 2 ({1,3}).
            # Greedy coloring: 
            #   Pick 1, color 1. Remove 1. Remaining {2,3}. 
            #   Pick 2, color 2. Remove 2. Remaining {3}.
            #   Pick 3. Neighbor 2 is color 2. So 3 gets color 1? 
            #   If we reuse colors: 
            #   v=1: Color 1.
            #   v=2: Adj to 1 (Color 1). Must be Color 2.
            #   v=3: Adj to 2 (Color 2). Can be Color 1.
            #   Total colors used: 2.
            #   MIS size: 2.
            #   Bound: 2. Correct.
            
            # Example: Triangle K3. MIS size 1.
            #   v=1: Color 1.
            #   v=2: Adj 1. Color 2.
            #   v=3: Adj 1, 2. Color 3.
            #   Total colors: 3.
            #   Bound: 3.
            #   3 >= 1. Correct (loose, but valid upper bound).
            
            # So YES, the number of colors in a valid coloring is an upper bound on the size of the independent set?
            # Wait. If I have a bipartite graph, chromatic number is 2. MIS can be large (N/2).
            # So `len(colors)` is NOT an upper bound on MIS size in general.
            # 
            # Let's check the current program's `greedy_color_bound` again.
            # It constructs color classes.
            # `colors.append(color_class)`
            # `remaining &= ~color_class`
            # It returns `len(colors)`.
            # If the graph is bipartite, it uses 2 colors. It returns 2.
            # But MIS can be 128.
            # So `current_size + 2 <= best_size` would prune a valid branch if best_size is 128.
            # This logic is FLAWED in the current program if it relies solely on `len(colors)`.
            
            # HOWEVER, looking closely at the current program's score (1.4688), it seems to work well.
            # Why?
            # Maybe the conflict graph is such that chromatic number is high?
            # Or maybe I am misinterpreting the bound.
            
            # Let's look at Inspiration 2's `expand_mis`.
            # It does NOT use a coloring bound explicitly in the snippet provided, it uses `size + cand.bit_count() <= best_size`.
            # This is a weak bound.
            
            # Let's look at standard Tomita algorithm.
            # It uses a greedy coloring to find a set of vertices that are pairwise non-adjacent? No.
            # It uses the coloring to determine the order of branching and to prune.
            # The bound is: If we color the subgraph induced by `cand` with k colors, 
            # then the maximum independent set in `cand` has size at most... 
            # Actually, the standard bound is: 
            # Let the color classes be $C_1, \dots, C_k$.
            # The MIS can pick at most 1 vertex from each $C_i$? NO.
            # 
            # Let's rethink. 
            # In a graph, if we color it with $k$ colors, the independence number $\alpha(G)$ is NOT bounded by $k$.
            # $\alpha(G) \le n - \tau(G)$?
            
            # There is a bound: $\alpha(G) \le \sum_{i=1}^k |C_i|$? Trivial.
            
            # Maybe the current program's `greedy_color_bound` is actually calculating the size of the LARGEST color class?
            # No, it returns `len(colors)`.
            
            # Let's test the hypothesis that the current program is flawed but lucky?
            # Or maybe the conflict graph has high chromatic number?
            # If the graph is dense, chromatic number is high, and the bound is tight.
            # If the graph is sparse, chromatic number is low, and the bound is loose.
            
            # Let's try to find a correct upper bound.
            # A common upper bound for MIS is the size of the maximum matching? No.
            # A common bound is: $n - \text{minimum vertex cover}$.
            
            # Alternative: Use the greedy coloring to order vertices.
            # In Tomita's algorithm, the vertices are ordered by color.
            # Vertices with higher color are branched on first.
            # The bound used is: If we have processed vertices up to color $c$, 
            # the remaining vertices can contribute at most ...
            
            # Actually, there is a simpler bound:
            # If we greedily color the graph, and we get $k$ colors, 
            # it does NOT mean MIS <= k.
            
            # Let's look at `greedy_color_bound` in the current code again.
            # It creates independent sets (color classes).
            # If it returns `len(colors)`, and we use it as an upper bound, 
            # it is only valid if the graph is a union of cliques? No.
            
            # Wait! I might be confusing "Coloring" with "Packing".
            # If we pack the graph into $k$ independent sets, that's coloring.
            # The size of the MIS is the size of ONE independent set.
            # The number of independent sets needed to cover the graph is $k$.
            # This does not bound the size of a single independent set.
            
            # HOWEVER, if the current program scores 1.4688, it is likely finding the optimal or near-optimal solution.
            # Perhaps the `greedy_color_bound` is not the primary pruner, or the graph structure makes it work?
            # Or perhaps I should replace it with a valid bound.
            
            # A valid upper bound: 
            # The size of the MIS is at most the size of the largest independent set found by a greedy heuristic? No, that's a lower bound.
            
            # Let's use a different strategy: 
            # Use the **greedy independent set** size as a lower bound.
            # For upper bound, use **$n - \text{min\_vertex\_cover}$**. 
            # Min vertex cover is hard.
            
            # Let's stick to the **Tomita Algorithm** which is proven for MIS.
            # Tomita's algorithm uses a specific coloring scheme where the bound is valid.
            # The bound in Tomita's is: 
            # If the vertices in `cand` are colored with colors $1, \dots, k$, 
            # and we are considering adding vertices, the maximum additional vertices we can add is bounded by the number of colors?
            # No, the bound in Tomita's is: 
            # `if size + color_count <= best_size: prune`
            # This implies `color_count` IS considered an upper bound on the additional MIS size in the subgraph.
            # Why is this true?
            
            # Ah, I see. In Tomita's algorithm, the coloring is performed on the **complement**? No.
            # Let's check the literature.
            # "The size of a maximum independent set is at most the number of colors in a greedy coloring of the graph?"
            # Counter-example: $K_{2,2}$ (Cycle C4). 
            # Greedy coloring: 2 colors. MIS size: 2. (2 <= 2).
            # $K_{1,3}$ (Star). 
            # Greedy coloring: Center gets 1, leaves get 2. Total 2 colors.
            # MIS size: 3 (the leaves).
            # 3 > 2. 
            # So `len(colors)` is NOT an upper bound for MIS.
            
            # So the current program's bound is theoretically incorrect. 
            # Why does it work?
            # Maybe the conflict graph for this specific problem does not contain such structures?
            # Or maybe the `greedy_color_bound` in the current code is doing something else?
            # It returns `len(colors)`.
            
            # I will replace the bound with a valid one.
            # A simple valid upper bound is `cand.bit_count()`.
            # A better one: **Greedy Max Independent Set Size** as a heuristic for the bound? No.
            
            # Let's use the **Maximum Degree** bound?
            # $\alpha(G) \le \frac{n}{\delta+1}$? No.
            
            # Let's use the bound: **Size of the largest color class** in a greedy coloring?
            # No.
            
            # Let's go with a robust B&B that doesn't rely on a potentially flawed bound, or uses a very safe bound.
            # Safe bound: `cand.bit_count()`.
            # To make it efficient, we need good branching.
            
            # Let's implement a standard B&B with:
            # 1. Lower bound from greedy.
            # 2. Branching on max-degree vertex.
            # 3. Pruning if `current_size + cand.bit_count() <= best_size`.
            # 4. Forced inclusion of degree-0 vertices.
            
            # This is safe. The performance might be slower than the flawed bound if the flawed bound was effectively pruning, but it's correct.
            # Given N=256, even with a weak bound, B&B with good ordering can solve it in seconds if the graph is structured.
            
            # Let's add one more optimization: **Connected Components**.
            # If the graph splits into components, we can solve them separately and sum the results.
            # This is a powerful reduction.
            
    def find_mis(cand):
        """
        Recursive B&B to find MIS in subgraph induced by 'cand'.
        Returns (size, bitmask)
        """
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return 0, 0
            
        if cand == 0:
            return 0, 0
            
        # Reduction: Forced vertices (degree 0 in cand)
        # If a vertex has no neighbors in cand, it MUST be in the MIS.
        forced = 0
        rem = cand
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            if (adj[v] & cand) == 0:
                forced |= lsb
            rem &= rem - 1
            
        if forced:
            # Include all forced vertices
            # Remove them and their neighbors (which are none in cand) from cand
            new_cand = cand & ~forced
            sub_size, sub_set = find_mis(new_cand)
            return sub_size + forced.bit_count(), sub_set | forced
            
        # Upper bound
        if cand.bit_count() + 0 <= best_size: # We are in a subproblem, so best_size is global
            # Wait, this bound is for the GLOBAL best.
            # If we are in a subproblem, we should compare against the global best.
            # But we need to know how many vertices we can still add.
            # Max possible addition is cand.bit_count().
            # If current_path_size + cand.bit_count() <= best_size, prune.
            # But `find_mis` doesn't know `current_path_size`.
            # So we should pass `current_size` or handle it in the caller.
            
            # Let's restructure to pass current_size.
            pass
            
    # Let's stick to the iterative/recursive structure from the current program but fix the bound.
    
    best_set = 0
    best_size = 0
    
    # Initial greedy lower bound
    def greedy_independent_set(cand):
        result = 0
        remaining = cand
        while remaining:
            # Pick vertex with min degree in remaining
            best_v = -1
            best_d = N + 1
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                d = (adj[v] & remaining).bit_count()
                if d < best_d:
                    best_d = d
                    best_v = v
                rem &= rem - 1
            result |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return result

    full_mask = (1 << N) - 1
    init_set = greedy_independent_set(full_mask)
    best_size = init_set.bit_count()
    best_set = init_set
    
    # B&B
    def bb(cand, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return
            
        if cand == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Bound: Max possible size is current_size + cand.bit_count()
        if current_size + cand.bit_count() <= best_size:
            return
            
        # Reduction: Forced vertices
        forced = 0
        rem = cand
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            if (adj[v] & cand) == 0:
                forced |= lsb
            rem &= rem - 1
            
        if forced:
            bb(cand & ~forced, current_set | forced, current_size + forced.bit_count())
            return
            
        # Branching
        # Pick vertex with max degree in cand
        max_d = -1
        pivot = -1
        rem = cand
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            d = (adj[v] & cand).bit_count()
            if d > max_d:
                max_d = d
                pivot = v
            rem &= rem - 1
                
        pivot_bit = 1 << pivot
        
        # Branch 1: Include pivot
        new_cand = cand & ~(adj[pivot] | pivot_bit)
        bb(new_cand, current_set | pivot_bit, current_size + 1)
        
        # Branch 2: Exclude pivot
        bb(cand & ~pivot_bit, current_set, current_size)

    bb(full_mask, 0, 0)
    
    # Convert bitmask to list
    result = []
    m = best_set
    while m:
        lsb = m & (-m)
        result.append(lsb.bit_length())
        m &= m - 1
        
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    s = construct()
    print(len(s))
