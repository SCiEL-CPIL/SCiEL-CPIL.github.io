"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    N = 256
    # Build conflict graph: edge between i and j (i<=j, 0-indexed for i,j in 0..N-1 representing values 1..N)
    # if (i+1) + (j+1) is a perfect square.
    # We need a subset S of {1,...,256} such that for all a,b in S with a<=b, a+b is NOT a perfect square.
    # This includes a=b, so 2a must not be a perfect square for any a in S.
    
    # Let's use 0-indexed: vertex i represents value i+1.
    # Conflict between i and j (i <= j) if (i+1)+(j+1) is a perfect square.
    # Note: when i==j, we check 2*(i+1).
    
    # Build adjacency as bitsets for efficiency.
    # adj[i] is a bitset where bit j is set if i and j conflict (i != j).
    # We also need to handle self-conflicts: if 2*(i+1) is a perfect square, then i cannot be in any valid set.
    
    # First, identify vertices that are self-conflicting (cannot be in any valid independent set)
    self_conflict = [False] * N
    for i in range(N):
        val = i + 1
        if _is_square(2 * val):
            self_conflict[i] = True
    
    # Build adjacency bitsets for non-self-conflicting vertices
    # Use Python integers as bitsets
    adj = [0] * N
    for i in range(N):
        if self_conflict[i]:
            continue
        val_i = i + 1
        for j in range(i + 1, N):
            if self_conflict[j]:
                continue
            val_j = j + 1
            if _is_square(val_i + val_j):
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Now solve maximum independent set using branch and bound with bitset pruning.
    # We work with the set of vertices that are not self-conflicting.
    # Let's reindex: create a list of valid vertices (not self-conflicting)
    valid_vertices = [i for i in range(N) if not self_conflict[i]]
    M = len(valid_vertices)
    
    # Remap to 0..M-1
    vertex_to_idx = {}
    for idx, v in enumerate(valid_vertices):
        vertex_to_idx[v] = idx
    
    # Build new adjacency in the remapped space
    adj2 = [0] * M
    for idx_i, vi in enumerate(valid_vertices):
        for idx_j, vj in enumerate(valid_vertices):
            if idx_i == idx_j:
                continue
            if (adj[vi] >> vj) & 1:
                adj2[idx_i] |= (1 << idx_j)
    
    # Branch and bound for maximum independent set
    # State: (candidates_bitset, current_set_size)
    # Upper bound: current_set_size + greedy_independent_set_size(candidates)
    
    # Precompute: for each vertex, its neighbors within the valid set
    # adj2[i] is the bitset of neighbors of i
    
    # Greedy independent set heuristic for upper bound
    def greedy_is_size(candidates):
        """Estimate size of a maximal independent set in candidates using greedy."""
        if candidates == 0:
            return 0
        # Pick vertices greedily: repeatedly pick the vertex with minimum degree in the induced subgraph
        # For simplicity and speed, use a simple greedy: pick vertex 0 from candidates, remove it and its neighbors
        # This gives a lower bound on MIS, but we want an upper bound... 
        # Actually for upper bound we can use: size of candidates / (avg degree + 1) or similar
        # A better upper bound: the size of a maximal independent set is at most the clique cover number, 
        # but computing that is hard. 
        # Let's use: upper bound = number of vertices in candidates (trivial) or 
        # a greedy packing: the minimum number of cliques needed to cover... 
        # Simpler: use the fact that in any independent set, no two vertices are adjacent.
        # Upper bound via greedy coloring of the complement? Too complex.
        # Let's use a simple bound: count vertices, and if we can quickly find a large independent set, use that as lower bound for pruning.
        # For upper bound, we'll use: len(bin(candidates).count('1')) which is trivial but valid.
        # Better: use a greedy maximal independent set size as a LOWER bound, and for upper bound use the total count.
        # Actually, let's implement a proper upper bound using a greedy vertex cover approximation.
        # The size of a minimum vertex cover >= size of maximum matching, and MIS = n - vertex_cover.
        # So upper bound on MIS = n - lower_bound_on_vertex_cover.
        # For a quick bound, let's just use the count of candidates as the upper bound initially.
        
        # Let's do a simple greedy to get a decent lower bound, and use count as upper bound.
        # For pruning, we need a tight upper bound. Let's use:
        # Upper bound = current_size + (number of candidates)
        # This is weak. Let's try a better heuristic.
        
        # Greedy maximal independent set (for lower bound):
        remaining = candidates
        count = 0
        while remaining:
            # Pick the vertex with lowest degree in the remaining subgraph
            # For speed, just pick the lowest-index vertex
            # Find lowest set bit
            lsb = remaining & (-remaining)
            v = lsb.bit_length() - 1
            count += 1
            # Remove v and its neighbors from remaining
            remaining &= ~(adj2[v] | lsb)
            remaining &= ~(1 << v)  # already removed by lsb, but be safe
        return count
    
    # Actually, for a better upper bound, let's use a simple greedy coloring of the conflict graph induced by candidates.
    # The chromatic number is an upper bound on the clique number, but we need the independence number.
    # Independence number <= n / (min_degree + 1) in the induced subgraph... not quite.
    # Let's just use a simple approach: the upper bound is the number of vertices in candidates.
    # We'll improve pruning with the greedy lower bound.
    
    best_size = 0
    best_set = set()
    
    # Recursive branch and bound
    # candidates: bitset of remaining candidates
    # current_set: set of vertices chosen so far
    # current_size: number of vertices in current_set
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + number of candidates
        num_candidates = candidates.bit_count()
        if current_size + num_candidates <= best_size:
            return
        
        # Try to pick a vertex to branch on
        # Heuristic: pick the vertex with the highest degree in the candidate subgraph
        # For simplicity, pick the first available vertex
        lsb = candidates & (-candidates)
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        # New candidates: candidates minus v and its neighbors
        new_candidates = candidates & ~(adj2[v] | (1 << v))
        current_set.add(v)
        branch_and_bound(new_candidates, current_set, current_size + 1)
        current_set.remove(v)
        
        # Option 2: exclude v
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(new_candidates, current_set, current_size)
    
    branch_and_bound((1 << M) - 1, set(), 0)
    
    # Convert back to original values
    result = [valid_vertices[v] + 1 for v in best_set]
    result.sort()
    
    # Verify the result
    for i, a in enumerate(result):
        for b in result[i:]:
            if _is_square(a + b):
                # This should not happen
                raise ValueError(f"Invalid pair: {a} + {b} = {a+b} is a perfect square")
    
    if not result:
        # Fallback: return a single element that doesn't self-conflict
        for i in range(N):
            if not self_conflict[i]:
                return [i + 1]
        return [1]
    
    return result

# EVOLVE-BLOCK-END
