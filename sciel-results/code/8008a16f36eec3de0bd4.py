"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Find a maximum cardinality subset of [1, 256] such that a+b is never a perfect square
    for any a <= b in the set (including a=b).
    
    This is equivalent to finding the Maximum Clique in the complement graph of the conflict graph.
    Conflict graph G: Edge (u,v) if u+v is a square.
    Complement graph H: Edge (u,v) if u+v is NOT a square.
    We want Max Clique in H.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Identify valid numbers (2*x is not a square)
    valid_nums = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            valid_nums.append(x)
    
    m = len(valid_nums)
    
    # Build adjacency bitsets for the COMPLEMENT graph (H)
    # adj_comp[i] has bit j set if valid_nums[i] + valid_nums[j] is NOT a square
    adj_comp = [0] * m
    for i in range(m):
        a = valid_nums[i]
        for j in range(i + 1, m):
            b = valid_nums[j]
            if a + b not in squares:
                adj_comp[i] |= (1 << j)
                adj_comp[j] |= (1 << i)
    
    # Global best solution
    best_size = 0
    best_clique = []
    
    # Greedy coloring of the subgraph induced by 'mask'
    # Returns a dictionary mapping vertex index to color, and the max color used.
    # This provides an upper bound on the clique size in that subgraph.
    def greedy_coloring(mask):
        if mask == 0:
            return {}, 0
        
        # Get vertices in mask
        verts = [v for v in range(m) if (mask >> v) & 1]
        
        # Sort by degree in the subgraph (descending) for better coloring
        # Degree in subgraph H
        degs = {}
        for v in verts:
            degs[v] = bin(mask & adj_comp[v]).count('1')
        verts.sort(key=lambda v: -degs[v])
        
        colors = {}
        max_color = 0
        for v in verts:
            # Find colors used by neighbors of v that have already been colored
            neighbor_mask = mask & adj_comp[v]
            used_colors = set()
            # Iterate through set bits in neighbor_mask
            nm = neighbor_mask
            while nm:
                # Get lowest set bit
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                nm -= lsb
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        return colors, max_color

    def branch_and_bound(mask, current_size, current_clique):
        nonlocal best_size, best_clique
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique[:]
            return
        
        # Upper bound check
        colors, max_color = greedy_coloring(mask)
        if current_size + max_color <= best_size:
            return
        
        # Get vertices in mask, ordered by color (descending)
        # We process vertices with higher colors first, as they offer larger potential
        verts = [v for v in range(m) if (mask >> v) & 1]
        verts.sort(key=lambda v: -colors[v])
        
        for v in verts:
            # Pruning: if even with this vertex's color bound we can't beat best, stop
            # (since subsequent vertices have <= color)
            if current_size + colors[v] <= best_size:
                break
            
            # Branch 1: Include v
            # New candidates must be neighbors of v in H (to form a clique)
            new_mask = mask & adj_comp[v]
            current_clique.append(v)
            branch_and_bound(new_mask, current_size + 1, current_clique)
            current_clique.pop()
            
            # Branch 2: Exclude v
            # Remove v from mask for the next iterations in this loop
            mask &= ~(1 << v)
            
    # Start search
    full_mask = (1 << m) - 1
    branch_and_bound(full_mask, 0, [])
    
    # Convert indices back to numbers
    result = [valid_nums[i] for i in best_clique]
    
    # Ensure nonempty
    if not result:
        # Fallback: pick any valid number
        if valid_nums:
            return [valid_nums[0]]
        else:
            return [1]
            
    return result

# EVOLVE-BLOCK-END
