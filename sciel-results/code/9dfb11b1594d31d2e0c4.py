import time
import random

N = 512

# Precompute adjacency lists as bitmasks
# adj[x] is a bitmask where bit y is 1 if x and y are in a 4-AP together
# Note: The graph is undirected. If x and y are in a 4-AP, they conflict.
# We need to identify all pairs (u, v) such that there exists a 4-AP containing both.
# A 4-AP is {a, a+d, a+2d, a+3d}.
# Pairs in a 4-AP:
# (a, a+d), (a, a+2d), (a, a+3d)
# (a+d, a+2d), (a+d, a+3d)
# (a+2d, a+3d)
# So for each 4-AP, we add edges between all pairs of its elements.

adj = [0] * (N + 1)

def add_edge(u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

# Generate all 4-APs and add edges
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # Elements: a, b, c, e
        # Add edges for all pairs
        add_edge(a, b)
        add_edge(a, c)
        add_edge(a, e)
        add_edge(b, c)
        add_edge(b, e)
        add_edge(c, e)

# Global variables for the solver
best_size = 0
best_mask = 0
start_time = 0
time_limit = 55.0 # Leave 5s buffer

def get_coloring_bound(available):
    """
    Greedy coloring of the graph induced by 'available'.
    Returns the number of colors, which is a lower bound on the clique size,
    and thus an upper bound on the independent set size for that subgraph?
    No, for MIS, the coloring bound is:
    The maximum size of an independent set in a graph is at most the number of colors in any valid coloring?
    Actually, the standard MIS B&B uses coloring to bound the *size* of the independent set.
    If we color the graph such that no two adjacent vertices have the same color,
    then an independent set can contain at most one vertex of each color.
    So, the number of colors is an upper bound on the size of the MIS of the subgraph.
    """
    if available == 0:
        return 0
    
    colors = 0
    remaining = available
    
    while remaining:
        colors += 1
        # Greedily select a maximal independent set for this color class
        # A vertex v can be in this color class if it is in remaining and 
        # has no neighbors in the already selected vertices for this color class?
        # No, standard greedy coloring for MIS bound:
        # We want to partition 'available' into independent sets.
        # We pick a vertex v from 'remaining'.
        # We assign it the current color.
        # We remove v and its neighbors from 'remaining' (because neighbors cannot share the color).
        # Wait, if we remove neighbors, we are forming a maximal independent set for this color.
        # Then we repeat for the next color.
        
        # To make the bound tight, we should order vertices by degree or something.
        # Here we just pick any available bit.
        # For speed, let's pick the lowest set bit.
        # v = (remaining & -remaining).bit_length() - 1
        
        # Better heuristic: Pick vertex with highest degree in 'remaining'?
        # For simplicity and speed in Python, let's just peel off a maximal independent set.
        
        color_class = 0
        temp_remaining = remaining
        
        # We need to find a maximal independent set in 'remaining'.
        # We iterate through vertices.
        # To be fast, we can just pick one, remove it and its neighbors.
        
        # Actually, a simpler bound often used is just the number of vertices,
        # but coloring is better.
        
        # Let's implement a simple greedy coloring step:
        # 1. Pick a vertex v from 'remaining'.
        # 2. Mark v as colored with 'colors'.
        # 3. Remove v and all its neighbors from 'remaining'.
        # This forms one color class.
        
        # To improve the bound, we should pick v such that it "removes" many vertices?
        # Or just pick arbitrarily?
        # Arbitrary is fast.
        
        # Let's pick the vertex with the highest degree in 'remaining' to make the color class smaller?
        # If the color class is smaller, we need more colors, which is a *worse* (higher) bound.
        # We want a *tight* (lower) bound on the upper bound?
        # No, the number of colors is an upper bound on the MIS size.
        # We want the *smallest* possible number of colors to get the tightest upper bound.
        # To minimize the number of colors, we should make each color class as *large* as possible.
        # So we should pick vertices that have *few* neighbors in 'remaining'?
        # Or just greedy: pick v, remove v and N(v).
        
        # Let's try picking the vertex with the minimum degree in 'remaining'.
        # This tends to create larger independent sets for each color, reducing total colors.
        
        # Finding min degree is O(V^2) or O(E). With bitmasks, we can check degrees.
        # deg(v) = popcount(adj[v] & remaining)
        
        # For speed, let's just pick the first available bit. It's O(1) to find.
        # The bound might be loose, but the search might still be fast enough for N=512?
        # N=512 is small. The graph is sparse?
        # Number of 4-APs is ~ N^2/6 ~ 40k. Edges ~ 6 * 40k = 240k.
        # Density is low.
        
        # Let's stick to simple greedy coloring:
        # While remaining:
        #   Pick v (lowest bit)
        #   remaining = remaining & ~(adj[v] | (1<<v))
        #   colors++
        
        # This is fast.
        
        # Optimization: Precompute nothing, just use bit ops.
        
        pass # Placeholder, logic below

    return colors

# The logic inside get_coloring_bound needs to be implemented properly.

def construct():
    global best_size, best_mask, start_time
    
    start_time = time.time()
    best_size = 0
    best_mask = 0
    
    # Initial greedy solution to set a good lower bound
    # This helps prune the B&B early
    greedy_mask = 0
    greedy_size = 0
    available = (1 << (N + 1)) - 2  # Bits 1..N set
    
    # Greedy: pick vertices in order of increasing index? Or random?
    # Let's do a few random greedy runs to get a good initial bound.
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        g_mask = 0
        g_size = 0
        for v in order:
            # Check if v can be added
            # v is available if it's not in g_mask and no neighbor is in g_mask
            # Condition: (g_mask & adj[v]) == 0
            if (g_mask & adj[v]) == 0:
                g_mask |= (1 << v)
                g_size += 1
        if g_size > best_size:
            best_size = g_size
            best_mask = g_mask
            
    # Now run B&B
    # We need a function that explores the search tree.
    # To avoid recursion depth issues and overhead, we might use a stack,
    # but for N=512, recursion depth is at most 512, which is fine.
    # However, Python function call overhead is significant.
    
    # We will use a recursive function with a node limit or time check.
    
    def solve(available, current_mask, current_size):
        global best_size, best_mask, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Upper bound
        # Simple bound: current_size + popcount(available)
        # If this is <= best_size, prune
        # popcount in Python: bin(x).count('1') is slow.
        # Use x.bit_count() (Python 3.10+)
        
        if current_size + available.bit_count() <= best_size:
            return
            
        # Tighter bound: Coloring
        # If we use coloring, it might be too slow if done at every node.
        # Let's try without coloring first, or only if the simple bound is close.
        # Given N=512, the simple bound might be loose.
        # Let's add a simple coloring bound if the gap is small?
        # Or just use the simple bound and hope the search is guided well.
        
        # Heuristic: Pick a vertex to branch on.
        # Common heuristic: Pick vertex with highest degree in 'available'.
        # Or random.
        
        # Let's pick a vertex with high degree to prune faster?
        # Or low degree?
        # In MIS, branching on a vertex with high degree removes many neighbors,
        # reducing the 'available' set significantly in the "Include" branch.
        # This is usually good.
        
        # Find vertex with max degree in available
        # deg(v) = (adj[v] & available).bit_count()
        
        # To save time, let's just pick the first available bit.
        # v_bit = available & -available
        # v = v_bit.bit_length() - 1
        
        # Let's try picking a random vertex from available to diversify?
        # No, B&B usually benefits from deterministic or degree-based heuristics.
        
        # Let's pick the vertex with the highest degree in the available subgraph.
        # This is O(N) per node. 512 * 512 = 260k ops per level?
        # Depth 512. 260k * 512 = 130M ops. Might be too slow in Python.
        
        # Let's stick to the first available bit for speed.
        v_bit = available & -available
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New available: remove v and its neighbors
        new_avail_include = available & ~(adj[v] | v_bit)
        solve(new_avail_include, current_mask | v_bit, current_size + 1)
        
        # Check time
        if time.time() - start_time > time_limit:
            return
            
        # Branch 2: Exclude v
        # New available: remove v
        new_avail_exclude = available & ~v_bit
        solve(new_avail_exclude, current_mask, current_size)
        
    # Start with all vertices available
    initial_available = 0
    for i in range(1, N + 1):
        initial_available |= (1 << i)
        
    solve(initial_available, 0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return sorted(result)
