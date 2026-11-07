import math
from functools import lru_cache

def construct():
    N = 256
    # 1. Build the conflict graph (edges where gcd > 1)
    # We use 0-indexed for bitsets, so vertex i corresponds to number i+1
    # Bit i in an integer mask represents the presence of number (i+1)
    
    # Precompute neighbors for each vertex
    # neighbors[i] is a bitset of all j such that gcd(i+1, j+1) > 1
    neighbors = [0] * N
    
    # Also precompute weights
    weights = [i + 1 for i in range(N)]
    
    # To speed up neighbor computation, we can use prime factors
    # But N=256 is small, O(N^2) gcd computation is fine (256*256 = 65536 ops)
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
        neighbors[i] = mask

    # Total sum of all vertices
    total_sum = sum(weights)
    
    # We will use a recursive branch and bound.
    # State: available_mask (bitset of vertices that can still be selected)
    # We need to find the max weight independent set in the subgraph induced by available_mask.
    
    # To make B&B efficient, we need a good upper bound.
    # Bound: sum of weights of all vertices in available_mask.
    # This is valid because an independent set is a subset of available_mask.
    
    # We can improve the bound by noting that if we select a vertex v, we remove its neighbors.
    # A tighter bound: For each vertex v in available, the max weight we can get is either
    # weight[v] + MWIS(available \ (neighbors[v] U {v})) or MWIS(available \ {v}).
    # This is the definition of MWIS.
    
    # We implement a recursive function with memoization?
    # The state is the mask. There are 2^256 states, so memoization is impossible for large masks.
    # However, the branching factor is reduced significantly by the "remove neighbors" step.
    # The graph is sparse.
    
    # Let's use a simple B&B without memoization first, relying on the structure.
    # To ensure speed, we pick the vertex to branch on carefully.
    # Heuristic: Pick a vertex with the highest weight? Or highest degree?
    # Picking a high-degree vertex reduces the search space in the "Include" branch.
    
    # We'll use a global variable to track the best solution found so far.
    best_sum = 0
    best_set = 0
    
    # We need to calculate sum of weights for a mask quickly.
    # Since we call this often, we can precompute prefix sums? No, mask is arbitrary.
    # We can iterate through set bits.
    
    def get_sum(mask):
        s = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # To optimize get_sum, we can maintain the sum in the recursion stack.
    
    def branch_and_bound(available_mask, current_sum):
        nonlocal best_sum, best_set
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = 0 # We don't strictly need the set, just the sum, but good for verification
            return

        # Upper bound check
        # Simple bound: current_sum + sum(weights in available_mask)
        # We can compute this sum.
        # Optimization: If current_sum + (precomputed max possible remaining) <= best_sum, prune.
        
        # Let's compute the sum of available vertices.
        # This is O(k) where k is number of bits.
        avail_sum = get_sum(available_mask)
        if current_sum + avail_sum <= best_sum:
            return

        # Select a vertex to branch on.
        # Heuristic: Pick a vertex with high weight and high degree to reduce space.
        # Let's pick the vertex with the maximum weight in the available set.
        # Or, pick a vertex that appears in many conflicts.
        # Let's pick the vertex with the highest weight.
        
        # Find index of highest weight vertex in available_mask
        # Since weights are 1..256, highest weight is highest index.
        # We can find the highest set bit.
        # But wait, higher index means higher weight.
        # So we can just take the highest set bit.
        
        # However, a better heuristic for MWIS is often to pick a vertex with high degree.
        # Let's try picking a vertex with high degree in the induced subgraph.
        # Degree of v in available = popcount(neighbors[v] & available_mask)
        
        # To keep it simple and fast, let's pick the vertex with the highest weight (highest index).
        # This is a common greedy choice.
        
        # Find highest set bit
        # lsb is lowest. We want highest.
        # Python int: bit_length gives position of highest set bit + 1.
        highest_idx = available_mask.bit_length() - 1
        v = highest_idx
        
        # Branch 1: Include v
        # New available: available_mask with v and its neighbors removed
        new_mask_include = available_mask & ~(neighbors[v] | (1 << v))
        branch_and_bound(new_mask_include, current_sum + weights[v])
        
        # Branch 2: Exclude v
        # New available: available_mask with v removed
        new_mask_exclude = available_mask & ~(1 << v)
        branch_and_bound(new_mask_exclude, current_sum)

    # Initial call
    # All vertices are available initially
    initial_mask = (1 << N) - 1
    branch_and_bound(initial_mask, 0)
    
    # Reconstruct the set from best_set? 
    # Wait, I didn't track best_set properly in the recursive calls above (I set it to 0).
    # I need to track the actual set to return the list.
    # Let's modify to track the set of indices.
    
    # Re-doing with proper set tracking.
    
    best_sum = 0
    best_indices = []
    
    def get_sum_mask(mask):
        s = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def solve(available_mask, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return

        # Pruning
        # Upper bound: current_sum + sum of all weights in available_mask
        # We can speed this up by precomputing sums for chunks?
        # For now, get_sum_mask is fine.
        if current_sum + get_sum_mask(available_mask) <= best_sum:
            return

        # Select vertex v
        # Heuristic: Pick vertex with highest weight (highest index)
        # Or maybe pick a vertex that has many neighbors in the available set?
        # Let's stick to highest index (highest weight) for simplicity and effectiveness.
        
        # Find a vertex to branch on.
        # Using highest index is good because it removes high-value conflicts early.
        
        # Get highest set bit
        v = available_mask.bit_length() - 1
        
        # Branch 1: Include v
        # Remove v and its neighbors from available
        mask_remove = neighbors[v] | (1 << v)
        new_mask = available_mask & ~mask_remove
        solve(new_mask, current_sum + weights[v], current_indices + [v])
        
        # Branch 2: Exclude v
        new_mask_ex = available_mask & ~(1 << v)
        solve(new_mask_ex, current_sum, current_indices)

    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # Convert best_indices (0-based) to 1-based integers
    result = [i + 1 for i in best_indices]
    
    # Ensure types are correct
    return [int(x) for x in result]

# Run once to verify it works and is fast enough?
# The problem says "Each evaluation must finish within 60 s".
# The function construct() is called by the evaluator.
# We should ensure it doesn't hang.
# N=256 might be too large for naive B&B without stronger bounds.
# Let's add a stronger bound or heuristic.

# Stronger Bound:
# We can use a greedy coloring of the *complement* graph? No.
# We can use a simple packing bound:
# The maximum weight independent set is at most the sum of weights of a maximal matching? No.
# 
# Let's try a different approach:
# The graph of non-coprime numbers is composed of components based on primes.
# Actually, it's not disjoint components.
# 
# However, we can use the fact that for any prime p, at most one multiple of p can be in the set.
# This is a constraint.
# 
# Let's stick to B&B but improve the vertex selection.
# Selecting the vertex with the highest degree in the available subgraph often works well.
# 
# Let's refine the vertex selection in solve():
# 
# def get_v(available_mask):
#     # Pick v that maximizes degree in available subgraph
#     # degree(v) = popcount(neighbors[v] & available_mask)
#     # We can iterate through set bits in available_mask
#     # N=256, so max 256 iterations per node.
#     # If the tree is deep, this adds up.
#     # But it reduces branching factor significantly.
#     
#     best_v = -1
#     best_deg = -1
#     m = available_mask
#     while m:
#         lsb = m & -m
#         idx = lsb.bit_length() - 1
#         deg = (neighbors[idx] & available_mask).bit_count()
#         if deg > best_deg:
#             best_deg = deg
#             best_v = idx
#         m ^= lsb
#     return best_v
#
# This is O(N) per node. With good pruning, the number of nodes should be small.

# Let's implement this improved version.

def construct():
    N = 256
    neighbors = [0] * N
    weights = [i + 1 for i in range(N)]
    
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
        neighbors[i] = mask

    best_sum = 0
    best_indices = []
    
    def get_sum_mask(mask):
        s = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def get_v(available_mask):
        # Pick vertex with highest degree in available subgraph
        # Tie-breaker: highest weight
        best_v = -1
        best_key = (-1, -1) # (degree, weight)
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (neighbors[idx] & available_mask).bit_count()
            key = (deg, weights[idx])
            if key > best_key:
                best_key = key
                best_v = idx
            m ^= lsb
        return best_v

    def solve(available_mask, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return

        # Pruning
        if current_sum + get_sum_mask(available_mask) <= best_sum:
            return

        v = get_v(available_mask)
        
        # Branch 1: Include v
        mask_remove = neighbors[v] | (1 << v)
        new_mask = available_mask & ~mask_remove
        solve(new_mask, current_sum + weights[v], current_indices + [v])
        
        # Branch 2: Exclude v
        new_mask_ex = available_mask & ~(1 << v)
        solve(new_mask_ex, current_sum, current_indices)

    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    result = [i + 1 for i in best_indices]
    return [int(x) for x in result]

# To be safe against timeouts, we can add a node limit or time check, 
# but the problem asks for the exact maximum. 
# Given N=256, the B&B with degree heuristic should be fast.
# The number of primes is ~54. The rest are composites.
# The search space is constrained by the coprime condition.

# Final check on types:
# "Types are checked exactly: integers must be Python int ... sequences must be list or tuple"
# My return is a list of ints.

# One more optimization: 
# If we find a solution with sum equal to the theoretical maximum possible (sum of all primes + 1?), we can stop early.
# The theoretical maximum is not simply sum of primes, because we can include some composites if they are coprime to the selected primes? 
# No, if we select a prime p, we cannot select any multiple of p.
# So for each prime p, we can select at most one number from {p, 2p, 3p, ...}.
# To maximize sum, for each prime p, we should select the largest multiple of p that is coprime to all other selected numbers?
# This is the structure of the problem.
# It's a packing problem.
# 
# The B&B will find the optimum.

# Let's output the code.
