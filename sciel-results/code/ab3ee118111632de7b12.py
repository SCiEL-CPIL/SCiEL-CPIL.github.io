"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph using bitsets (Python ints as bitsets)
    # conflict_bits[i] is a bitset of all nodes conflicting with node i
    conflict_bits = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_bits[i] |= (1 << j)
    
    # Tomita's algorithm for Maximum Independent Set (MIS)
    # We find MIS in the conflict graph = Maximum Clique in complement,
    # but it's easier to work directly with independent set formulation.
    # 
    # Actually, let's use the standard approach:
    # Maximum Independent Set in G = Maximum Clique in complement of G
    # But for sparse graphs, working with independent set directly is better.
    #
    # Tomita's color sort algorithm works on cliques. Let's reformulate:
    # We want max independent set in conflict graph.
    # This equals max clique in the "non-conflict" graph (complement).
    # 
    # However, a cleaner approach for MIS directly:
    # Use branch and bound where at each node we have a set of candidates R
    # and a current independent set S. We bound |S| + |R| and use coloring
    # on the induced subgraph of R to get tighter bounds.
    #
    # For independent set: the bound is |S| + coloring_number_of_R_in_conflict_graph
    # because in a conflict graph, a proper coloring gives an upper bound on
    # the size of an independent set (each color class is an independent set,
    # so we can pick at most one from each color... wait, no.
    #
    # Actually for independent set: we want to pick nodes with NO edges between them.
    # In the conflict graph, edges represent conflicts. An independent set is a set
    # of vertices with no edges between them.
    # 
    # The standard Tomita algorithm finds maximum clique. For maximum independent set,
    # we can apply it to the complement graph. But the complement of a sparse graph
    # is dense, which makes coloring expensive.
    #
    # Alternative: Use the fact that MIS in G = MCQ in complement(G).
    # But let's think differently.
    #
    # Better approach: Direct MIS branch-and-bound with bitset operations.
    # At each recursive call, we have:
    #   - current independent set size (count)
    #   - set of candidate vertices R (bitset)
    #   - upper bound: count + popcount(R)
    # 
    # For tighter bound, we can use a greedy coloring of the conflict graph
    # induced by R. In a proper coloring of a graph, each color class is an
    # independent set in that graph. Wait, that's the opposite of what we want.
    #
    # Let me reconsider. For MAX CLIQUE (Tomita):
    #   - We have candidate set R
    #   - We color R greedily (proper coloring of the graph induced by R)
    #   - The color number of vertex v gives an upper bound on clique size from v onward
    #   - This works because in a clique, all vertices must have different colors
    #
    # For MAX INDEPENDENT SET:
    #   - We want a set of vertices with NO edges between them
    #   - In the conflict graph, this means no two selected vertices are adjacent
    #   - Equivalent to: maximum clique in the complement graph
    #
    # Since our conflict graph is sparse (each node has maybe 5-15 conflicts out of ~200),
    # the complement is very dense (~190 edges per node). Coloring a dense graph
    # is expensive but the clique in a dense graph might be found quickly.
    #
    # Actually, let me just use a direct approach:
    # Branch and bound for MIS where:
    #   - bound = |current_set| + |R| (simple)
    #   - Or use a greedy partition of R into cliques in the conflict graph
    #     (since independent set in G = clique in complement, and clique in complement
    #      means no edges in G... this is getting circular)
    #
    # Let me use the simplest effective approach:
    # 1. Start with a good greedy solution as lower bound
    # 2. Use branch and bound with simple |R| bound
    # 3. Use bitset operations for speed
    # 4. Order vertices by degree to prune early
    
    # Get a good initial solution with greedy
    def greedy_solution():
        remaining = (1 << num_valid) - 1  # all vertices
        selected = []
        while remaining:
            # Find vertex with minimum degree in remaining
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (conflict_bits[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                v ^= lowbit
            
            selected.append(best_v)
            # Remove best_v and all its neighbors
            remaining &= ~(conflict_bits[best_v] | (1 << best_v))
        
        return selected, len(selected)
    
    greedy_sel, greedy_size = greedy_solution()
    best_size = greedy_size
    best_set = list(greedy_sel)
    
    # Check if we're still within time
    if time.time() > deadline - 5:
        return sorted([valid_numbers[i] for i in best_set])
    
    # Branch and bound for MIS
    # State: (R_bitset, current_size, current_set_list)
    # We'll use a recursive approach with global best tracking
    
    # Precompute vertex ordering by degree (higher degree first for pruning)
    degrees = [conflict_bits[i].bit_count() for i in range(num_valid)]
    order = sorted(range(num_valid), key=lambda x: -degrees[x])
    
    # For the B&B, we process vertices in a fixed order
    # At each step, we decide to include or exclude the next candidate
    # 
    # More efficient: recursive branching on the set R
    # Pick a vertex v from R, branch on: include v (remove v and neighbors) 
    # or exclude v (remove v only)
    
    # Use iterative deepening or just standard B&B
    # Let's use a stack-based approach to avoid recursion depth issues
    
    # Actually, for ~200 nodes, recursion should be fine with proper pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For better pruning, use a greedy coloring bound on the conflict graph induced by R
    # In a proper coloring of conflict graph induced by R, each color class is a set
    # of mutually non-adjacent vertices (independent set in conflict graph).
    # Wait, that's exactly what we want! If we color the conflict graph,
    # each color class is an independent set. So the maximum independent set
    # is at most... no, that's not right either.
    #
    # Let me think again. We want the maximum independent set in the conflict graph.
    # A proper coloring partitions vertices into independent sets (color classes).
    # The size of the maximum independent set is at least the size of the largest
    # color class, but that's a lower bound, not upper.
    #
    # For an UPPER bound on independent set size:
    # If we can partition R into k cliques in the conflict graph, then
    # the independent set can contain at most one vertex from each clique,
    # so |IS| <= k.
    #
    # Greedy clique cover: repeatedly find a maximal clique, remove it, count.
    # This gives an upper bound on the independent set size.
    #
    # Alternatively, use the complement: MIS in G = MCQ in complement(G).
    # Apply Tomita's color sort on the complement graph.
    # In the complement graph, an edge means "no conflict" (can be in same IS).
    # A clique in complement = independent set in original.
    # Tomita's coloring: proper coloring of complement graph induced by R.
    # In a proper coloring of complement, adjacent vertices (in complement) 
    # have different colors. A clique in complement has all vertices mutually
    # adjacent in complement, so all must have different colors.
    # Thus color_number gives upper bound on clique size in complement = IS size in original.
    
    # Build complement adjacency bitsets
    # complement[i] = all vertices j != i that are NOT in conflict_bits[i]
    all_mask = (1 << num_valid) - 1
    complement_bits = [0] * num_valid
    for i in range(num_valid):
        complement_bits[i] = all_mask & ~(conflict_bits[i] | (1 << i))
    
    # Tomita's color sort for maximum clique in complement graph
    # This finds maximum independent set in original graph
    
    def color_sort(R):
        """Greedy coloring of complement graph induced by R.
        Returns list of (vertex, color) sorted by color (ascending),
        where color is the upper bound on clique size from that vertex onward."""
        uncolored = R
        colors = []
        color_num = 0
        
        while uncolored:
            color_num += 1
            # Greedily assign this color to non-adjacent vertices in complement
            # (i.e., vertices that ARE adjacent in original conflict graph... no)
            # In complement graph, we want to color so that adjacent (in complement)
            # get different colors. Adjacent in complement = no edge in original.
            # So in a color class, no two vertices can be adjacent in complement,
            # meaning all pairs in a color class MUST be adjacent in original (conflict).
            # So a color class in complement coloring = a clique in original conflict graph.
            
            # Greedy: pick vertices that form a clique in original (all mutually conflicting)
            # and assign them the same color
            available = uncolored
            color_class = 0
            while available:
                # Pick first available vertex
                lowbit = available & (-available)
                v = lowbit.bit_length() - 1
                color_class |= lowbit
                # Remove v and all vertices NOT conflicting with v (in original)
                # i.e., remove v and vertices that have complement edge to v
                # Vertices that CAN be in same color class must conflict with v
                # So keep only vertices that conflict with v
                available &= conflict_bits[v]
            
            colors.append((color_class, color_num))
            uncolored &= ~color_class
        
        # Now we have color classes. For Tomita's algorithm, we need to assign
        # to each vertex its color number, and sort by color number.
        # The color number of a vertex is the color assigned to its color class.
        # Vertices with higher color numbers are processed first (they have higher upper bound).
        
        # Build vertex-to-color mapping
        vertex_color = {}
        for color_class, cnum in colors:
            cc = color_class
            while cc:
                lowbit = cc & (-cc)
                idx = lowbit.bit_length() - 1
                vertex_color[idx] = cnum
                cc ^= lowbit
        
        # Return list of (vertex, color) sorted by color descending
        result = []
        for v, c in vertex_color.items():
            result.append((v, c))
        result.sort(key=lambda x: -x[1])
        return result
    
    # Maximum clique in complement graph (= MIS in original)
    def max_clique_complement(R, current_size, current_clique):
        """Find maximum clique in complement graph induced by R,
        given we already have current_clique of size current_size."""
        nonlocal best_size, best_set, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if not R:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_clique)
            return
        
        # Upper bound: current_size + number of vertices in R
        if current_size + R.bit_count() <= best_size:
            return
        
        # Color sort R in complement graph
        sorted_vertices = color_sort(R)
        
        # Process vertices in order of decreasing color
        # For each vertex, the color gives an upper bound on how large
        # the clique can be if we include this vertex
        i = len(sorted_vertices) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            
            v, c = sorted_vertices[i]
            
            # Upper bound: current_size + c
            if current_size + c <= best_size:
                i -= 1
                continue
            
            # Include v in clique
            # In complement graph, neighbors of v are vertices in R that have
            # complement edge to v, i.e., complement_bits[v] & R
            # But we also need to only consider vertices that come AFTER v in our ordering
            # Actually in Tomita's algorithm, we just intersect with R
            new_R = R & complement_bits[v]
            
            # But we should only consider vertices with index <= i in sorted order
            # to avoid duplicates. Actually, standard Tomita processes in order.
            # Let's simplify: just use new_R = R & complement_bits[v]
            
            current_clique.append(v)
            max_clique_complement(new_R, current_size + 1, current_clique)
            current_clique.pop()
            
            # Remove v from R
            R &= ~(1 << v)
            i -= 1
    
    # Start the search
    initial_R = (1 << num_valid) - 1
    # Try to improve on greedy solution
    max_clique_complement(initial_R, 0, [])
    
    # Convert result back to original values
    result = sorted([valid_numbers[i] for i in best_set])
    
    # Verify the result
    if result:
        result_set = set(result)
        for a in result:
            if 2 * a in squares:
                # Invalid - fallback to greedy
                result = sorted([valid_numbers[i] for i in greedy_sel])
                break
            for b in result:
                if a <= b and (a + b) in squares:
                    result = sorted([valid_numbers[i] for i in greedy_sel])
                    break
            if result != sorted([valid_numbers[i] for i in best_set]):
                break
    
    return result
# EVOLVE-BLOCK-END
