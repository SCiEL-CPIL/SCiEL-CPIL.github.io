import time
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge indices for all 4-cycles
# Edge (i, j) with i < j.
# Index mapping:
# For i in range(N):
#   for j in range(i+1, N):
#      idx = ...
# We can compute index on the fly or precompute a dictionary.
# Precomputing dictionary is fast for N=16.

def get_edge_index(i, j):
    if i > j:
        i, j = j, i
    # Number of edges before row i:
    # Row 0: N-1
    # Row 1: N-2
    # ...
    # Row i: N-1-i edges starting from index sum_{k=0}^{i-1} (N-1-k)
    # Sum = i*(N-1) - i*(i-1)/2
    # Position in row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of edge indices
def precompute_cycles():
    cycles = []
    # 4 vertices a,b,c,d
    # Cycle 1: a-b-c-d-a -> edges (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c), (c,b), (b,d), (d,a)
    
    verts = range(N)
    for combo in itertools.combinations(verts, 4):
        a, b, c, d = combo
        # Edges for cycle 1: ab, bc, cd, da
        c1 = (get_edge_index(a, b), get_edge_index(b, c), get_edge_index(c, d), get_edge_index(d, a))
        # Edges for cycle 2: ab, bd, dc, ca
        c2 = (get_edge_index(a, b), get_edge_index(b, d), get_edge_index(d, c), get_edge_index(c, a))
        # Edges for cycle 3: ac, cb, bd, da
        c3 = (get_edge_index(a, c), get_edge_index(c, b), get_edge_index(b, d), get_edge_index(d, a))
        
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
    return cycles

CYCLES = precompute_cycles()

def is_valid(colors):
    """Check if any 4-cycle has 4 distinct colors."""
    for c in CYCLES:
        # c is a tuple of 4 edge indices
        # Check if colors on these edges are all distinct
        # We can do this without a set for speed, or use set.
        # len(set) is fine for 5460 cycles.
        if len({colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]}) == 4:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds hard limit

    # Strategy: Try modular colorings
    # 1. (i-j) % k
    # 2. (i+j) % k
    # 3. (i*j) % k (maybe)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    for k in range(120, 0, -1):
        if time.time() > deadline:
            break
            
        # Formula 1: (i-j) % k
        colors1 = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = get_edge_index(i, j)
                colors1[idx] = (i - j) % k
        # Check validity
        if is_valid(colors1):
            # Count unique colors
            unique_count = len(set(colors1))
            if unique_count > best_count:
                best_count = unique_count
                best_colors = colors1
            # Since we iterate k downwards, the first valid k for this formula
            # gives the max colors for THIS formula. 
            # But maybe a smaller k in another formula is better?
            # No, we want max colors. 
            # If k=50 works, unique_count is ~50.
            # If k=4 works, unique_count is 4.
            # So we want the largest k that works.
            # We are iterating k from 119 down.
            # So the FIRST valid k we find for (i-j)%k is the best for that formula.
            # We should check if we should break or continue.
            # It's possible (i+j)%k works for a larger k than (i-j)%k?
            # Unlikely, but possible.
            # We continue to check other formulas for the same k?
            # Actually, if k=10 works for formula 1, we have 10 colors.
            # If k=20 works for formula 2, we have 20 colors.
            # So we should check all formulas for the current k?
            # Or iterate k down, and for each k, try all formulas.
            # If any formula works for k, we record it.
            # Since we go down, the first k that works for ANY formula is the global max?
            # Yes, because larger k means more potential colors.
            # However, for a fixed k, different formulas might use different numbers of colors?
            # (i-j)%k uses colors 0..k-1. Max k colors.
            # So if ANY formula with modulus k is valid, we can get up to k colors.
            # We should check all formulas for this k.
            
            # Let's check other formulas for this k.
            # If we find one, we update best.
            # But we can't stop the k-loop immediately unless we are sure no larger k works.
            # We are checking k=119, 118...
            # If k=119 works for formula 1, we have 119 colors. 
            # Nothing larger exists. So we can return immediately?
            # Yes! Max possible colors is 119.
            # If we find a valid coloring with k colors, and we are checking k in descending order,
            # the first valid k we encounter gives the maximum number of colors 
            # achievable by that specific family.
            # But does it guarantee global max?
            # No, a non-modular coloring could use 119 colors.
            # But we are restricted to direct formulas in this bootstrap.
            # Within the scope of "try modular formulas", the largest k is best.
            
            # However, to be safe, we'll just return the first one found 
            # from the largest k downwards.
            
            # Optimization: If we find a valid coloring for k, 
            # we can return it immediately because we are iterating k downwards.
            # Any subsequent k will be smaller, yielding fewer or equal colors.
            
            # BUT: We must check all formulas for the current k?
            # No, if Formula 1 works for k, we have a valid coloring with ~k colors.
            # Formula 2 might also work for k. It doesn't matter, we have a solution.
            # We don't need to find the "best" coloring for k, just A valid one.
            
            return colors1
        else:
            # Formula 1 failed. Try Formula 2.
            colors2 = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    idx = get_edge_index(i, j)
                    colors2[idx] = (i + j) % k
            
            if is_valid(colors2):
                return colors2
            
            # Optional: Try (i*j) % k
            if k > 1: # avoid div by 0 or trivial
                 colors3 = [0] * NUM_EDGES
                 for i in range(N):
                     for j in range(i + 1, N):
                         idx = get_edge_index(i, j)
                         colors3[idx] = (i * j) % k
                 if is_valid(colors3):
                     return colors3

    # Fallback: 1 color
    return best_colors
# EVOLVE-BLOCK-END
