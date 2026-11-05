import time
from math import gcd
import sys

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # 1. Build the conflict graph
    # G[i] is a bitmask of vertices j such that lcm(i, j) > N
    # We use 0-based indexing for vertices, so vertex i corresponds to number i+1
    G = [0] * N
    for i in range(N):
        val_i = i + 1
        for j in range(i + 1, N):
            val_j = j + 1
            # lcm(i, j) = i * j // gcd(i, j)
            # Check if lcm > N
            # To avoid overflow or division issues, check (i // gcd) * j > N
            g = gcd(val_i, val_j)
            if (val_i // g) * val_j > N:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
    
    # 2. Precompute vertex degrees and order them
    # Heuristic: Order vertices by degree descending to improve pruning
    vertices = list(range(N))
    # We will reorder the graph to match this order for better cache locality and coloring
    # Actually, for Tomita, we often just use the original indices but process them in a specific order.
    # Let's stick to the original indices but use a priority queue or sorted list for branching?
    # Standard Tomita uses a fixed order. Let's order by degree.
    
    # Calculate degrees
    degrees = [bin(G[i]).count('1') for i in range(N)]
    
    # Sort vertices by degree descending
    # We need to remap the graph or just access via original indices?
    # It's easier to keep original indices but pick the next vertex based on a heuristic.
    # However, the coloring bound works best if the set of candidates is processed in a specific order.
    # Let's use the standard approach: 
    # P is the set of candidates.
    # We maintain a list of candidates.
    
    best_size = 0
    best_set = []
    
    # Global variables for the recursive search
    # Using a list for the current path
    current_set = []
    
    def greedy_coloring(candidates):
        """
        Assigns colors to vertices in 'candidates' such that adjacent vertices have different colors.
        Returns the maximum color used.
        candidates is a list of vertex indices.
        """
        if not candidates:
            return 0
        
        colors = {}
        max_color = 0
        uncolored = set(candidates)
        
        # To make it fast, we can iterate through candidates and assign the smallest available color
        # A more efficient version for bitmasks:
        # But here we pass a list.
        
        # Actually, for the bound, we just need the number of color classes.
        # Greedy coloring:
        # While uncolored is not empty:
        #   Pick a vertex v
        #   Assign color max_color + 1
        #   Remove v and its neighbors from uncolored for this color class? No, that's independent set.
        # Greedy Coloring for Bound:
        # We want to partition V into k independent sets.
        # Algorithm:
        #   color = 0
        #   U = candidates
        #   while U is not empty:
        #       color += 1
        #       I = {}
        #       for v in U:
        #           if v is not adjacent to any vertex in I:
        #               I.add(v)
        #       U = U \ I
        
        # This is O(N^2) per bound calculation. With N=512, it might be slow if called often.
        # Optimized Greedy Coloring (Tomita):
        # It iterates through vertices in a specific order.
        
        # Let's implement the standard Tomita coloring which is faster.
        # It processes vertices in the order they appear in the candidate list.
        
        color = 0
        U = candidates[:]  # List of vertices
        # We need to check adjacency to already colored vertices of the current color.
        # This implementation is O(N^2) worst case.
        # Given N=512, 512^2 = 262k ops. If we do this thousands of times, it's fine.
        
        while U:
            color += 1
            new_U = []
            # We pick vertices greedily.
            # To be consistent, we pick the first vertex in U that is compatible with the current color class.
            # Actually, the standard algorithm picks vertices in order.
            
            # Let's use a simpler, slightly less tight but faster bound?
            # No, the coloring bound is the key.
            
            # Efficient greedy coloring:
            # We maintain a list of vertices that can take the current color.
            # Initially, all candidates.
            # We pick one, mark it, remove its neighbors from the "available for this color" set.
            
            # Let's try a standard implementation.
            pass # Placeholder, will implement below
            
        return color

    # The above greedy coloring is a bit slow if not careful.
    # Let's use a bitmask-based coloring which is faster.
    
    def color_bound(candidates_mask):
        """
        Returns an upper bound on the size of the maximum independent set
        for the subgraph induced by vertices in candidates_mask.
        Uses greedy coloring.
        """
        if candidates_mask == 0:
            return 0
        
        colors = 0
        remaining = candidates_mask
        
        while remaining:
            colors += 1
            # Find a maximal independent set in the remaining graph
            # to assign this color.
            # We iterate through bits of remaining.
            # We want to pick vertices that are not adjacent to each other.
            
            # Greedy: Pick the first vertex, remove it and its neighbors.
            # To make it deterministic and fast, we pick the lowest set bit?
            # Or we can try to pack more.
            
            # A common heuristic for the coloring bound in MIS is to process vertices in a fixed order
            # and assign the smallest possible color.
            # However, for the bound, any valid coloring works.
            # The "smallest color" greedy on the subgraph induced by 'remaining' gives a valid coloring.
            
            # Let's do:
            # uncolored = remaining
            # while uncolored:
            #    v = pick a vertex from uncolored
            #    assign color 'colors' to v
            #    uncolored = uncolored & ~G[v] & ~(1<<v)  # Remove v and its neighbors? 
            #    Wait, this removes neighbors from being the SAME color? No.
            #    If v is color C, its neighbors cannot be color C.
            #    So, for the current color class, we can only pick vertices from 'remaining' that are not adjacent to any previously picked vertex in this color class.
            
            # Let's implement:
            # current_color_class_mask = 0
            # available = remaining
            # while available:
            #    v = lowest set bit in available
            #    current_color_class_mask |= (1<<v)
            #    # Remove v and all its neighbors from 'available' for this color
            #    # Because if u is a neighbor of v, u cannot have the same color as v.
            #    available = available & ~(G[v] | (1<<v))
            
            # This produces a valid independent set for this color.
            # We repeat for the next color.
            
            available = remaining
            while available:
                # Pick the lowest set bit
                v_bit = available & (-available)
                v = v_bit.bit_length() - 1
                # Remove v and its neighbors from available
                available = available & ~(G[v] | v_bit)
            
            # The vertices picked in this loop form an independent set.
            # We remove them from 'remaining' for the next color.
            # We need to track which vertices were picked.
            
            # Let's re-do this properly.
            
            pass # Re-implementing in main logic for clarity

    # Re-implementing color_bound correctly
    def get_color_bound(mask):
        if mask == 0:
            return 0
        bound = 0
        remaining = mask
        while remaining:
            bound += 1
            # Find a maximal independent set in the subgraph induced by 'remaining'
            # to assign this color.
            # We greedily pick vertices.
            # Let's pick the vertex with the lowest index for determinism.
            # Then remove it and its neighbors from 'remaining' (for this color assignment).
            # Wait, the standard coloring algorithm assigns colors one by one.
            # Color 1: Max Independent Set in G[remaining].
            # Remove MIS from remaining.
            # Color 2: Max Independent Set in G[remaining].
            # ...
            
            # Greedy MIS in G[remaining]:
            # I = 0
            # avail = remaining
            # while avail:
            #   v = lowest bit in avail
            #   I |= (1<<v)
            #   avail = avail & ~(G[v] | (1<<v))
            
            # This I is an independent set.
            # We assign color 'bound' to all vertices in I.
            # We remove I from 'remaining'.
            
            I = 0
            avail = remaining
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                I |= v_bit
                # Remove v and its neighbors from avail
                avail = avail & ~(G[v] | v_bit)
            
            remaining = remaining & ~I
            
        return bound

    # Branch and Bound
    # We need to be careful with recursion depth. N=512, depth up to 512.
    # Python default recursion limit is 1000. We might hit it if we don't prune well.
    # But MIS on 512 nodes should prune heavily.
    # To be safe, we can increase recursion limit or use iterative DFS.
    # Let's increase recursion limit.
    sys.setrecursionlimit(2000)

    def search(mask, current_size, current_path):
        nonlocal best_size, best_set
        
        if time.time() > DEADLINE:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_path[:]
            return
        
        # Upper bound
        bound = get_color_bound(mask)
        if current_size + bound <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: Pick the vertex with the highest degree in the subgraph?
        # Or just the lowest index?
        # Tomita suggests picking a vertex that is "critical".
        # Let's pick the lowest index for simplicity first.
        # To improve, we can pick the vertex with the max degree in the subgraph induced by mask.
        
        # Find vertex v in mask
        # Let's try to pick a vertex that maximizes the pruning.
        # A common choice is the vertex with the highest degree in the current subgraph.
        
        # Calculate degrees in subgraph? That's expensive.
        # Let's just pick the lowest set bit.
        v_bit = mask & (-mask)
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New mask: mask without v and without neighbors of v
        new_mask_include = mask & ~(G[v] | v_bit)
        current_path.append(v)
        search(new_mask_include, current_size + 1, current_path)
        current_path.pop()
        
        # Branch 2: Exclude v
        # New mask: mask without v
        new_mask_exclude = mask & ~v_bit
        search(new_mask_exclude, current_size, current_path)
        
    # Initial call
    initial_mask = (1 << N) - 1
    search(initial_mask, 0, [])
    
    # Convert back to values
    # v is 0-indexed, value is v+1
    result = [v + 1 for v in best_set]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# Run construct to test locally if needed
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
