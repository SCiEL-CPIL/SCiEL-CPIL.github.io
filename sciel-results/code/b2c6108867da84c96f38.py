"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [[False]*n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    
    def is_clique(vertices):
        if not vertices:
            return True
        for i, u in enumerate(vertices):
            for w in vertices[i+1:]:
                if not adj[u][w]:
                    return False
        return True
    
    def is_independent(vertices):
        if not vertices:
            return True
        for i, u in enumerate(vertices):
            for w in vertices[i+1:]:
                if adj[u][w]:
                    return False
        return True
    
    def is_homogeneous(vertices):
        return is_clique(vertices) or is_independent(vertices)
    
    # Try to find partition with k parts using backtracking
    # We search for a partition into exactly k homogeneous sets
    def try_k_partition(k, deadline):
        # For small n, we can try all ways to assign each vertex to one of k "colors"
        # where each color class must be homogeneous (all clique or all independent)
        # This is equivalent to: partition into k sets, each inducing clique or independent set
        
        # Use backtracking: assign vertices 0..n-1 to k slots
        # Prune: for each slot, maintain the set of vertices assigned and check if it can still be homogeneous
        
        # Better approach: enumerate all possible homogeneous subsets, then find k of them that partition V
        # For n=14, number of homogeneous subsets can be large, but we can prune
        
        # Alternative: direct backtracking with vertex ordering
        # Sort vertices by degree (descending) for better pruning
        
        vertices = list(range(n))
        
        # Precompute for each subset whether it's clique, independent, or neither
        # For n=14, 2^14 = 16384 subsets, feasible
        
        # Use bitmasks
        all_homog = {}  # mask -> True if homogeneous
        
        def compute_homog(mask, base_mask):
            """Compute if mask is homogeneous, using precomputed info for base_mask"""
            if mask in all_homog:
                return all_homog[mask]
            
            # Find a vertex in mask
            if mask == 0:
                all_homog[0] = True
                return True
            
            v = (mask & (-mask)).bit_length() - 1  # lowest set bit
            rest = mask ^ (1 << v)
            
            # Check if all vertices in mask are mutually adjacent (clique)
            # or mutually non-adjacent (independent)
            
            # For clique: every pair must be adjacent
            # For independent: no pair must be adjacent
            
            # Try clique: check if v is adjacent to all in rest, and rest is clique
            is_clique_candidate = True
            is_ind_candidate = True
            
            for u_bit in range(n):
                if rest & (1 << u_bit):
                    if not adj[v][u_bit]:
                        is_clique_candidate = False
                    else:
                        is_ind_candidate = False
            
            if is_clique_candidate and (rest == 0 or compute_homog(rest, mask)):
                all_homog[mask] = True
                return True
            if is_ind_candidate and (rest == 0 or compute_homog(rest, mask)):
                all_homog[mask] = True
                return True
            
            # Not homogeneous by this decomposition
            # But wait - the above logic is flawed. A set is clique iff all pairs are adjacent.
            # Let me just check directly for small sets.
            
            # Direct check
            verts = [i for i in range(n) if mask & (1 << i)]
            if len(verts) <= 1:
                all_homog[mask] = True
                return True
            
            # Check clique
            clique = True
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if not adj[verts[i]][verts[j]]:
                        clique = False
                        break
                if not clique:
                    break
            
            if clique:
                all_homog[mask] = True
                return True
            
            # Check independent
            independent = True
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if adj[verts[i]][verts[j]]:
                        independent = False
                        break
                if not independent:
                    break
            
            if independent:
                all_homog[mask] = True
                return True
            
            all_homog[mask] = False
            return False
        
        # Precompute all homogeneous subsets
        for mask in range(1 << n):
            compute_homog(mask, mask)
        
        # Now find k homogeneous subsets that form a partition of all vertices
        # Full mask
        full_mask = (1 << n) - 1
        
        # DFS: find k disjoint homogeneous subsets whose union is full_mask
        # Order doesn't matter for the partition, but for search we can fix an ordering
        # to avoid permutations: require that the first vertex in each subsequent subset
        # is greater than the first vertex in the previous subset (canonical form)
        
        # Simpler: just find any k disjoint homogeneous subsets covering all vertices
        # Use backtracking with memoization or pruning
        
        best_result = [None]
        
        def backtrack(remaining, parts_found, current_parts, deadline):
            if time.time() > deadline:
                return False
            
            if len(parts_found) == k:
                if remaining == 0:
                    best_result[0] = current_parts
                    return True
                else:
                    return False
            
            if remaining == 0:
                return False
            
            # Pruning: remaining must be partitionable into (k - len(parts_found)) nonempty homogeneous sets
            # Minimum number of vertices per set is 1, so we need at least (k - len(parts_found)) vertices
            remaining_count = bin(remaining).count('1')
            if remaining_count < (k - len(parts_found)):
                return False
            
            # Also, the maximum number of vertices in a homogeneous set is limited
            # But more importantly, we need to find subsets of remaining that are homogeneous
            
            # To avoid duplicate partitions, enforce that we pick subsets in a canonical order
            # Specifically, the minimum vertex in each new subset must be greater than the minimum vertex in the previous subset
            # This ensures we don't count permutations of the same partition
            
            if current_parts:
                min_prev = min(min(part) for part in current_parts)
            else:
                min_prev = -1
            
            # Find all homogeneous subsets of remaining whose min element > min_prev
            # This is expensive; instead, iterate over possible subsets
            
            # Better: pick the smallest available vertex, and try all homogeneous subsets containing it
            # that are subsets of remaining and have min > min_prev (or if first part, min >= 0)
            
            # Find smallest vertex in remaining
            if remaining == 0:
                return False
            min_v = (remaining & (-remaining)).bit_length() - 1
            
            # All valid next subsets must contain min_v (to fix ordering)
            # And must be homogeneous, subset of remaining
            
            # Enumerate all homogeneous subsets of remaining that contain min_v
            # and have min element == min_v (which is automatic since min_v is smallest in remaining)
            
            # Get all subsets of remaining that contain min_v and are homogeneous
            sub = remaining
            candidates = []
            
            # Iterate over all subsets of 'sub' that contain min_v
            # Use submask enumeration
            s = sub
            while s:
                if s & (1 << min_v) and all_homog.get(s, False):
                    candidates.append(s)
                s = (s - 1) & sub
            
            # Sort candidates by size (larger first may be better for pruning, or smaller)
            # Try larger subsets first to reduce remaining quickly
            candidates.sort(key=lambda x: bin(x).count('1'), reverse=True)
            
            for cand in candidates:
                if time.time() > deadline:
                    return False
                new_remaining = remaining ^ cand
                new_parts = current_parts + [cand]
                if backtrack(new_remaining, parts_found + 1, new_parts, deadline):
                    return True
            
            return False
        
        # Try different k values starting from 2
        for k in range(2, n + 1):
            result = backtrack(full_mask, 0, [], deadline)
            if best_result[0] is not None:
                # Convert masks to lists of vertices
                parts = []
                for mask in best_result[0]:
                    verts = [i for i in range(n) if mask & (1 << i)]
                    parts.append(verts)
                return parts
        
        return None
    
    # Greedy baseline
    def greedy_partition():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                if is_homogeneous(trial):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    # Try k=2 first
    result = try_k_partition(2, deadline)
    if result is not None:
        return [list(part) for part in result]
    
    if time.time() > deadline:
        return greedy_partition()
    
    # Try k=3
    result = try_k_partition(3, deadline)
    if result is not None:
        return [list(part) for part in result]
    
    if time.time() > deadline:
        return greedy_partition()
    
    # Try k=4
    result = try_k_partition(4, deadline)
    if result is not None:
        return [list(part) for part in result]
    
    if time.time() > deadline:
        return greedy_partition()
    
    # Fall back to greedy
    return greedy_partition()

# EVOLVE-BLOCK-END
