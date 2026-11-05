import math
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > 512:
            return False
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > 512:
                return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    
    # 1. Build Compatibility Graph using bitmasks
    # neighbors[i] is a bitmask of all j such that lcm(i, j) <= 512
    # We index vertices 1..n, but bitmasks use 0..n-1 for efficiency.
    # Let's map vertex v (1-indexed) to bit position v-1.
    
    neighbors = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if _lcm(i, j) <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
    
    # 2. Greedy Initial Solution
    # We want a large clique. 
    # A simple heuristic: start with the number with the most neighbors, then intersect.
    # Or, since the graph is defined by divisibility/LCM constraints, 
    # the set of divisors of a highly composite number is a clique.
    
    # Find the number M in [1, 512] with the most divisors.
    max_divs = 0
    best_M = 1
    divisors_of = [set() for _ in range(n + 1)]
    for M in range(1, n + 1):
        d = 0
        for i in range(1, M + 1):
            if M % i == 0:
                d += 1
                divisors_of[M].add(i)
        if d > max_divs:
            max_divs = d
            best_M = M
            
    initial_clique = divisors_of[best_M]
    
    # Try to augment this clique with other numbers
    # A number x is compatible with all in C if lcm(x, y) <= 512 for all y in C.
    # This is equivalent to x * y / gcd(x, y) <= 512.
    
    def can_add(x, clique):
        for y in clique:
            if _lcm(x, y) > 512:
                return False
        return True

    # Greedy augmentation
    current_clique = set(initial_clique)
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if x not in current_clique and can_add(x, current_clique):
                current_clique.add(x)
                changed = True
                # Break to restart check, as adding one might enable others? 
                # Actually, if x is compatible with all, adding it doesn't make others 
                # incompatible, it just adds to the set. So we can continue.
                # But if we add x, the condition for y is the same. 
                # So we can just iterate once.
        
    # Let's try a few random restarts of greedy to get a better initial bound
    import random
    random.seed(42)
    best_size = len(current_clique)
    best_set = set(current_clique)
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # Random order greedy
        nodes = list(range(1, n + 1))
        random.shuffle(nodes)
        s = set()
        candidates = set(nodes)
        while candidates:
            # Pick node with max degree in induced subgraph
            best_v = None
            max_deg = -1
            for v in candidates:
                if can_add(v, s):
                    # Count neighbors in candidates
                    deg = 0
                    # Optimization: use bitmask for candidates? 
                    # For now, simple loop. N=512 is small.
                    for w in candidates:
                        if w != v and _lcm(v, w) <= 512:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
            if best_v is None:
                break
            s.add(best_v)
            candidates.discard(best_v)
            
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 3. Branch and Bound for Maximum Clique
    # Tomita's Algorithm
    # We work with bitmasks.
    # P: set of candidates (bitmask)
    # R: current clique (set of vertices)
    # best_size: global max clique size found
    
    # To speed up coloring, we need an ordering of vertices.
    # We will use the vertices 1..n.
    # The coloring heuristic:
    # Greedily color the vertices in P.
    # The number of colors used is an upper bound on the clique size in P.
    
    # We represent P as an integer bitmask.
    # We need to iterate through set bits in P.
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def greedy_coloring(P_mask):
        # Returns list of colors for each vertex in P, and the max color count
        # Actually, we just need the max color count for the bound.
        # But for the recursion, we often want to order vertices by color.
        # Let's implement a standard greedy coloring that returns the number of colors.
        # And potentially the order.
        
        # For simplicity and speed, let's just return the number of colors.
        # We process vertices in descending order of degree in P? 
        # Or just 1..n? 
        # Tomita usually orders by degree.
        
        # Let's precompute degrees in P? That's expensive.
        # Let's just use the natural order 1..n for now, or pre-sorted by global degree.
        
        # Pre-sort vertices by global degree descending
        # global_deg[i] = popcount(neighbors[i])
        global_deg = [0] * (n + 1)
        for i in range(1, n + 1):
            global_deg[i] = count_bits(neighbors[i])
            
        # Order vertices 1..n by global_deg descending
        order = sorted(range(1, n + 1), key=lambda x: global_deg[x], reverse=True)
        
        # Map vertex index to its position in order? 
        # Or just iterate through 'order' and check if in P_mask.
        
        colors = {}
        max_color = 0
        for v in order:
            if not (P_mask & (1 << (v - 1))):
                continue
            # Find smallest color not used by neighbors in P
            used_colors = set()
            mask_v = neighbors[v]
            # Neighbors of v in P
            neigh_in_P = P_mask & mask_v
            # Iterate neighbors in P
            m = neigh_in_P
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = (lsb.bit_length() - 1) + 1 # vertex index
                if idx in colors:
                    used_colors.add(colors[idx])
                m ^= lsb
                
            # Assign smallest available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
            if max_color + len(R) <= best_size:
                # We can return early if we know we can't beat best_size?
                # No, we need the full coloring for the order in recursion usually.
                # But for the bound, max_color is enough.
                pass
                
        return max_color, colors

    # Recursive function
    # To avoid recursion limit issues and overhead, we might need to be careful.
    # But depth is at most 512.
    
    def expand(R, P_mask):
        nonlocal best_size, best_set
        
        if count_bits(P_mask) + len(R) <= best_size:
            return
            
        # Greedy coloring to get upper bound
        # This is the most expensive part.
        # Optimization: If P_mask is 0, return.
        if P_mask == 0:
            return
            
        # Simple bound: count_bits(P_mask)
        if len(R) + count_bits(P_mask) <= best_size:
            return
            
        # Greedy coloring
        # For performance, we might skip full coloring if the simple bound is tight enough?
        # No, coloring is much tighter.
        
        max_c, colors = greedy_coloring(P_mask)
        
        if len(R) + max_c <= best_size:
            return
            
        # Process vertices in order of color (descending)
        # Vertices with higher colors are "more central" in some sense?
        # Actually, we want to try vertices that are likely to lead to large cliques.
        # Tomita processes vertices in order of decreasing color.
        
        # Get vertices in P sorted by color descending
        verts_in_P = []
        m = P_mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1) + 1
            verts_in_P.append(idx)
            m ^= lsb
            
        verts_in_P.sort(key=lambda x: colors[x], reverse=True)
        
        for v in verts_in_P:
            if len(R) + colors[v] <= best_size:
                # Since we are processing in decreasing color order, 
                # all subsequent vertices have color <= colors[v].
                # So if this one can't beat best, none of the later ones can.
                break
                
            # Add v to R
            R_new = R | {v}
            
            # New P is intersection of current P and neighbors of v
            # Note: v itself is in P, but we remove it implicitly by intersecting with neighbors?
            # Neighbors of v do not include v (simple graph).
            # So P_new = P_mask & neighbors[v]
            P_new = P_mask & neighbors[v]
            
            # Recurse
            expand(R_new, P_new)
            
            # Remove v from P for the next iteration
            P_mask &= ~(1 << (v - 1))
            
            # Update bound
            if len(R_new) + count_bits(P_mask) <= best_size:
                return # Prune remaining
            
            # We could also update colors, but that's complex. 
            # The simple bound check above is usually sufficient.
            # Actually, re-coloring is expensive. 
            # The standard Tomita algorithm re-colors P after removing v.
            # Given the time limit, we might skip re-coloring and rely on the initial bound 
            # or the simple count bound. 
            # However, without re-coloring, the bound `colors[v]` for subsequent vertices 
            # in the loop is from the *original* coloring of the *original* P.
            # Is `colors[v]` a valid upper bound for the clique size in the *new* P?
            # The new P is a subset of the old P.
            # The clique size in a subset is <= clique size in the set.
            # So the original color bound is still valid (upper bound).
            # But it might be loose.
            
    # Start B&B
    # P is all vertices
    all_mask = (1 << n) - 1
    # But we should start with a good P? 
    # The greedy coloring on the full graph will give a bound.
    # Let's start with R = empty, P = all_mask.
    
    # To help the B&B, we can start with the best_set found by greedy.
    # If we start R = best_set, P = neighbors of all in best_set intersected?
    # No, that's not how it works. 
    # The B&B finds a clique. 
    # If we set R = best_set, we are looking for a clique that *contains* best_set?
    # No, we want to find a clique larger than best_size.
    # So we can just run B&B from scratch.
    # But if we know a clique of size K, we can prune any branch where len(R) + bound <= K.
    
    # Let's run expand with R=empty, P=all_mask
    # But wait, if we run from scratch, it might take long to find the first large clique.
    # We can initialize best_size with the greedy result.
    
    # One optimization: 
    # Instead of starting R=empty, we can pick a vertex v, add to R, and recurse with P=N(v).
    # But the standard expand handles this.
    
    # Let's ensure the initial best_size is set.
    best_size = len(best_set)
    best_set_final = set(best_set)
    
    # Run B&B
    # We need to be careful with recursion depth. 512 is fine.
    # But Python recursion overhead is high.
    # Let's try it.
    
    # To speed up, we can restrict the search space initially?
    # No, let's let it explore.
    
    # Important: The greedy coloring function `greedy_coloring` is called at every node.
    # This is O(N^2) per node. With many nodes, this might be slow.
    # However, pruning is effective.
    
    # Let's add a time check inside the recursion?
    # Python doesn't easily support interrupting recursion cleanly.
    # We can check time in `expand`.
    
    def expand_timed(R, P_mask):
        nonlocal best_size, best_set_final
        
        if time.time() > deadline:
            return
            
        if count_bits(P_mask) + len(R) <= best_size:
            return
            
        if P_mask == 0:
            return
            
        max_c, colors = greedy_coloring(P_mask)
        
        if len(R) + max_c <= best_size:
            return
            
        verts_in_P = []
        m = P_mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1) + 1
            verts_in_P.append(idx)
            m ^= lsb
            
        verts_in_P.sort(key=lambda x: colors[x], reverse=True)
        
        for v in verts_in_P:
            if time.time() > deadline:
                return
                
            if len(R) + colors[v] <= best_size:
                break
                
            R_new = R | {v}
            P_new = P_mask & neighbors[v]
            
            expand_timed(R_new, P_new)
            
            if len(R_new) > best_size:
                best_size = len(R_new)
                best_set_final = set(R_new)
                
            P_mask &= ~(1 << (v - 1))
            
            if len(R) + count_bits(P_mask) <= best_size:
                return

    # Start
    expand_timed(set(), all_mask)
    
    # Final check
    if not best_set_final:
        best_set_final = best_set
        
    if not _validate(best_set_final):
        best_set_final = set(range(1, 21))
        
    return list(best_set_final)
# EVOLVE-BLOCK-END
