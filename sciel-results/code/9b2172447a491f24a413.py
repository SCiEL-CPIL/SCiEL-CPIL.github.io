import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    # We use 0-indexed bits for 1..256, so bit i-1 corresponds to number i.
    conflict_mask = [0] * (n + 1)
    
    # Build conflict graph
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    # Helper to get size of bitmask
    def popcount(x):
        return x.bit_count()

    # Greedy coloring to get an upper bound on the independence number of a set of vertices
    # This is the core of the Tomita algorithm's pruning power.
    # vertices: bitmask of vertices to color
    # Returns: list of colors (bitmasks), where each color class is an independent set
    # The number of colors is the upper bound.
    def color_bound(vertices):
        colors = []
        while vertices:
            # Pick a vertex, typically the one with the lowest index for determinism 
            # or highest degree for better bounds. 
            # Standard Tomita often picks based on the order in the vertex list.
            # Let's just pick the lowest set bit for simplicity and speed, 
            # but sorting vertices by degree initially helps.
            # Actually, for the coloring heuristic in Tomita, we process vertices in a specific order.
            # We'll use a simple greedy coloring on the induced subgraph.
            
            # To make the bound tighter, we should process vertices in an order that 
            # likely leads to fewer colors. 
            # A common heuristic is to sort by degree in the subgraph.
            # However, doing full sorting at every node is expensive.
            # Let's stick to a simple greedy coloring on the bitmask.
            
            color_class = 0
            remaining = vertices
            # We want to assign vertices to the current color class if they don't conflict with each other.
            # Wait, the standard coloring for upper bound works like this:
            # We partition the vertices into independent sets (color classes).
            # The number of classes is an upper bound on the size of the MIS.
            
            # Efficient greedy coloring for upper bound:
            # 1. Initialize color = 0
            # 2. While vertices is not empty:
            #    - Create a new color class C (bitmask)
            #    - Iterate through vertices in some order.
            #    - If vertex v is not in C and v does not conflict with any vertex in C, add v to C.
            #    - Remove C from vertices.
            #    - Increment color count.
            
            # To speed this up, we can just greedily pick vertices.
            # Let's use a simple loop.
            while remaining:
                # Pick a vertex, e.g., the one with the smallest index
                # Or better, pick one that has fewest conflicts in 'remaining' to pack more into the color class?
                # Actually, for the *upper bound* via coloring, we want to create *large* independent sets to minimize the number of colors.
                # So we should pick vertices that are compatible with many others in the current color class.
                
                # Simple heuristic: pick the vertex with the smallest index to ensure determinism and speed.
                # A better heuristic is to pick the vertex with the highest degree in the *complement* graph 
                # (i.e., most compatible neighbors) to maximize the size of the current color class.
                
                # Let's just pick the lowest set bit for now.
                v_bit = remaining & -remaining
                v_idx = v_bit.bit_length() - 1 # 0-indexed
                
                # Check if v can be added to the current color class
                # It can if it doesn't conflict with any vertex already in color_class
                # conflict_mask[v_idx+1] & color_class == 0
                if (conflict_mask[v_idx + 1] & color_class) == 0:
                    color_class |= v_bit
                    remaining ^= v_bit
                
            if color_class:
                colors.append(color_class)
                vertices ^= color_class
        
        return len(colors)

    # Branch and Bound for Maximum Independent Set
    # best_size: global variable to track the maximum size found
    # best_set: global variable to track the set
    best_size = 0
    best_set_mask = 0
    
    # Initial greedy solution to set a good lower bound
    # Greedy: pick vertex with highest degree, add to set, remove neighbors, repeat?
    # Or just pick 1, 2, ... if compatible?
    # Let's try a simple greedy:
    current_mask = 0
    candidates = (1 << n) - 1 # All vertices 0..n-1
    while candidates:
        # Pick vertex with highest degree in remaining candidates?
        # For speed, just pick the first one?
        # Let's pick the one with the smallest index to be fast.
        v_bit = candidates & -candidates
        v_idx = v_bit.bit_length() - 1
        current_mask |= v_bit
        # Remove v and its neighbors from candidates
        neighbors = conflict_mask[v_idx + 1]
        candidates &= ~(v_bit | neighbors)
    
    best_size = popcount(current_mask)
    best_set_mask = current_mask

    # Tomita Branch and Bound
    # We maintain a list of vertices to consider.
    # The state is defined by the set of available vertices.
    
    def branch_and_bound(avail_mask):
        nonlocal best_size, best_set_mask
        
        # Time check
        if time.time() > deadline:
            return
        
        # Upper bound using coloring
        # If the upper bound is not better than best_size, prune
        ub = color_bound(avail_mask)
        if ub <= best_size:
            return
        
        # If avail is empty, we are done with this branch
        if avail_mask == 0:
            return
        
        # Heuristic: Pick a vertex to branch on.
        # Usually the one with the highest degree in the induced subgraph is good.
        # For speed, we can just pick the lowest set bit or the highest.
        # Let's pick the vertex with the highest index in the avail set? 
        # Or the one that appears first in a pre-sorted order?
        
        # Let's pick the vertex with the highest degree in the subgraph defined by avail_mask.
        # This requires calculating degrees, which is O(N^2) in worst case if done naively.
        # With bitmasks, degree is popcount(avail_mask & conflict_mask[v]).
        
        # To avoid O(N^2) at every node, we can just pick the lowest set bit.
        # This is often sufficient for N=256.
        v_bit = avail_mask & -avail_mask
        v_idx = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New avail = avail_mask & ~v_bit & ~neighbors(v)
        neighbors = conflict_mask[v_idx + 1]
        new_avail_include = avail_mask & ~v_bit & ~neighbors
        new_set_mask = best_set_mask | v_bit # Wait, we need to track the current set, not just best.
        
        # Correction: The standard recursive formulation tracks the current set size and mask.
        # Let's refactor to pass current_size and current_mask.
        
        # Actually, the `color_bound` gives the *additional* vertices we can add.
        # So if we are at a node with `avail_mask`, the max size we can get is `current_size + color_bound(avail_mask)`.
        # If `current_size + ub <= best_size`, prune.
        
        # Let's rewrite the function to accept current_size and current_mask.
        pass

    # Refactored B&B
    def solve(avail_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_size + color_bound(avail_mask) <= best_size:
            return
            
        if avail_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
        
        # Select vertex
        # Heuristic: Pick vertex with max degree in avail_mask
        # For N=256, iterating through bits is fast.
        # Let's just pick the lowest set bit for simplicity first.
        v_bit = avail_mask & -avail_mask
        v_idx = v_bit.bit_length() - 1
        
        # Include v
        neighbors = conflict_mask[v_idx + 1]
        new_avail_include = avail_mask & ~v_bit & ~neighbors
        solve(new_avail_include, current_size + 1, current_mask | v_bit)
        
        # Exclude v
        new_avail_exclude = avail_mask & ~v_bit
        solve(new_avail_exclude, current_size, current_mask)

    # Start solving
    # We need to pass the full set of vertices initially.
    # The vertices are 0..255.
    initial_avail = (1 << n) - 1
    
    # To improve performance, we can order the vertices.
    # The order in which we pick vertices affects the branching.
    # A good order is by descending degree in the original graph.
    # However, the `color_bound` and `solve` functions assume a fixed order or pick dynamically.
    # If we pick dynamically (lowest bit), the order is fixed by bit position.
    # We can remap the vertices to improve the order.
    
    # Let's calculate degrees and sort vertices by degree descending.
    # Then create a mapping from original index to new index.
    # This might be overkill for N=256 if the simple bit-picking works well enough.
    # Let's try the simple version first. If it's too slow, we'll optimize.
    
    # One optimization: If the graph is sparse, the simple B&B might explode.
    # The coloring bound is crucial.
    
    solve(initial_avail, 0, 0)
    
    # If time remains, we could try a second pass with a different vertex ordering
    # or a local search polish on the result.
    
    # Convert best_set_mask to list of numbers
    result = []
    for i in range(n):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
                print(f"Conflict: {a}, {b}")
    print(f"Valid: {valid}")
