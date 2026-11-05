"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _conflicts(n=256):
    """Precompute conflict pairs as bitsets."""
    # neighbors[i] is a bitset where bit j is set if i conflicts with j
    neighbors = [0] * (n + 1)
    for i in range(1, n + 1):
        mask_i = 0
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                mask_i |= (1 << j)
                neighbors[j] |= (1 << i)
        neighbors[i] = mask_i
    return neighbors

def _solve_mis(neighbors, n, time_limit=80.0):
    """
    Solve Maximum Independent Set using Branch and Bound with bitsets.
    """
    start_time = time.time()
    
    # Initial ordering: sort vertices by degree (descending) to find large sets early
    # We don't actually need to reorder the bitset indices, but we can pick vertices 
    # in a specific order during branching.
    
    # Precompute bit masks for vertices 1..n
    # We will maintain 'available' as a bitmask of vertices that can still be included.
    
    best_size = 0
    best_set = []
    
    # Initial available set: all vertices 1..n
    # Bit i corresponds to vertex i.
    all_vertices = 0
    for i in range(1, n + 1):
        all_vertices |= (1 << i)
    
    # To speed up vertex selection, we can precompute a list of vertices sorted by degree.
    # However, the degree changes as we remove vertices. 
    # A common heuristic is to pick the vertex with the highest degree in the *available* subgraph.
    # Calculating dynamic degree is expensive. 
    # Alternative: Pick the vertex with the highest *static* degree among available.
    # Or simply pick the lowest index available? No, that's bad.
    # Let's use a greedy coloring bound if possible, or just bit count.
    # For N=256, bit count bound is weak if the graph is sparse. 
    # But the divisibility graph is not that sparse.
    
    # Let's implement a simple recursive B&B.
    # To avoid deep recursion limits, we can use an explicit stack or increase recursion limit.
    # N=256 depth is fine for Python if we are careful, but let's use a stack-based approach 
    # or just set recursion limit.
    import sys
    sys.setrecursionlimit(5000)
    
    def popcount(x):
        return x.bit_count()
    
    def branch_and_bound(available, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: upper bound is current_size + number of bits in available
        if current_size + popcount(available) <= best_size:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Select a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in available subgraph?
        # Approximation: Pick vertex with highest static degree among available.
        # Let's precompute static degrees.
        
        # Find a candidate vertex. 
        # To improve pruning, we should pick a vertex that conflicts with many available vertices.
        # We can iterate through available bits and pick one.
        # For better performance, we can sort available vertices by static degree.
        
        # Get list of available vertices
        # This is O(N) per node, which might be slow.
        # Optimization: We can just pick the first available vertex? 
        # No, ordering matters.
        
        # Let's try a greedy coloring bound for tighter pruning.
        # Greedy coloring of the available subgraph gives an upper bound on MIS size.
        # If the bound <= best_size, prune.
        
        # Implementing greedy coloring on bitsets:
        # 1. Copy available
        # 2. While copy is not empty:
        #    a. Pick a vertex v from copy (e.g., highest degree in copy)
        #    b. Remove v and its neighbors from copy (this forms an independent set in the complement? No, this is for coloring)
        #    Actually, greedy coloring assigns colors. The number of colors is an upper bound on MIS? No, it's a lower bound on Chromatic Number, which is an upper bound on MIS? 
        #    Wait. Chromatic Number >= Clique Number. MIS <= N / Clique Number? No.
        #    Theorem: Size of MIS <= Number of colors in a valid coloring? 
        #    Yes, because each color class is an independent set, so the largest independent set is at most the number of vertices in the largest color class? No.
        #    Actually, if you color a graph with k colors, the graph is k-partite. The MIS can take at most 1 vertex from each part? No, it can take all from one part.
        #    So k colors does NOT bound MIS size from above. 
        #    However, if we color the *complement* graph... 
        #    Let's stick to the simple bound: current_size + popcount(available).
        #    It is weak, but with N=256, it might still work if we prune well.
        
        # Better heuristic: 
        # Pick vertex v from available.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # To make it faster, we can use a specific vertex selection strategy.
        # Let's pick the vertex with the highest degree in the *original* graph among the available ones.
        # We precomputed neighbors. We can precompute static degrees.
        
        # Since we are in a function, accessing external scope for degrees is fast.
        
        # Find vertex with max static degree in available
        # To avoid iterating 1..256 every time, we can just pick the lowest index? 
        # No, let's try to pick a "hub" vertex.
        
        # Let's just pick the first available vertex for now to see if it's fast enough.
        # If not, we can optimize.
        
        # Extract lowest set bit
        # v_bit = available & -available
        # v = v_bit.bit_length() - 1
        
        # Let's try picking a vertex with high degree.
        # We'll iterate through available bits.
        # To speed this up, we can pre-sort vertices by degree and iterate that list, checking if bit is set.
        
        # Let's use a simpler, effective heuristic: 
        # Randomize the vertex selection slightly or use static degree.
        
        # For now, let's pick the vertex with the highest static degree among available.
        # We have a global list `vertices_by_degree`.
        
        # Actually, let's just use the lowest index available to ensure determinism and simplicity, 
        # but that is often bad for B&B.
        
        # Let's use the vertex with the highest number of conflicts *within the available set*.
        # This requires calculating degree for each available vertex, which is expensive.
        
        # Compromise: Use static degree.
        # We need access to static degrees.
        
        # Let's refactor to pass static degrees or compute them.
        
        pass # Placeholder

    # Let's implement the full B&B with static degree ordering.
    
    # Precompute static degrees
    static_degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        static_degrees[i] = neighbors[i].bit_count()
    
    # List of vertices sorted by static degree descending
    vertices_sorted = sorted(range(1, n + 1), key=lambda x: -static_degrees[x])
    
    # We will use a stack to avoid recursion depth issues and allow early exit.
    # State: (available, current_size, current_set_tuple)
    # Using tuple for set to make it hashable/copyable? Or just list.
    # Storing the set explicitly is memory heavy. 
    # Alternative: Store the path of decisions? 
    # For N=256, storing the set (list of ints) is fine.
    
    # Initial state
    stack = [(all_vertices, 0, [])]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        available, current_size, current_set = stack.pop()
        
        # Prune
        if current_size + available.bit_count() <= best_size:
            continue
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            continue
            
        # Select vertex
        # Pick the vertex with highest static degree from available
        v = -1
        for u in vertices_sorted:
            if (available >> u) & 1:
                v = u
                break
        
        if v == -1:
            continue
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New available: remove v and its neighbors
        new_available_include = available & ~(neighbors[v] | v_bit)
        # We only push if it's promising
        if current_size + 1 + new_available_include.bit_count() > best_size:
            # To improve search, we push the "Include" branch first (LIFO means it's processed next)
            # Wait, stack is LIFO. If we push Exclude then Include, Include is popped first.
            # We want to explore Include first to find large solutions quickly.
            stack.append((new_available_include, current_size + 1, current_set + [v]))
            
        # Branch 2: Exclude v
        new_available_exclude = available & ~v_bit
        if current_size + new_available_exclude.bit_count() > best_size:
            stack.append((new_available_exclude, current_size, current_set))
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 256
    neighbors = _conflicts(n)
    
    # Run B&B
    best_set = _solve_mis(neighbors, n, time_limit=deadline - start_time - 5)
    
    # If B&B didn't find anything (shouldn't happen), fallback to greedy
    if not best_set:
        # Simple greedy
        selected = set()
        for i in range(1, n + 1):
            if all((neighbors[i] >> j) & 1 == 0 for j in selected):
                selected.add(i)
        best_set = sorted(selected)
    else:
        best_set = sorted(best_set)
        
    return best_set

# EVOLVE-BLOCK-END
