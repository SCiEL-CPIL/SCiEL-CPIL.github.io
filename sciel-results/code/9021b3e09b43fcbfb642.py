import time
import sys

# Increase recursion limit if necessary, though n=14 is small
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # 1. Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 2. Precompute compatibility masks
    # For a vertex v to be added to a clique containing mask M:
    # v must be adjacent to all vertices in M.
    # (M & ~adj[v]) must be 0.
    
    # For a vertex v to be added to an independent set containing mask M:
    # v must NOT be adjacent to any vertex in M.
    # (M & adj[v]) must be 0.
    
    # Precompute sorted vertex order by degree (descending) to prune earlier
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # We also need a list of masks for each vertex for quick lookup? 
    # adj[v] is the mask of neighbors.
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # 3. Search for minimum k
    # k can range from 1 to n.
    # For each k, we try all 2^k combinations of part types (0=clique, 1=independent).
    # We use backtracking to assign vertices 0..n-1 to parts 0..k-1.
    
    # To avoid re-solving for permutations of identical type vectors, we can
    # exploit symmetry: parts of the same type are interchangeable.
    # However, generating unique type vectors is complex.
    # Instead, we rely on the fact that n=14 is small. 2^14 = 16384.
    # If k is small (e.g., 2-5), this is very fast.
    # If k is large, the search space for vertices is smaller (fewer parts to choose from? No, more parts).
    # Actually, for large k, finding a partition is easier (less constraint).
    # The hard case is finding a partition with SMALL k.
    
    def check_partition(k, types):
        """
        Returns a list of parts (masks) if a valid partition exists, else None.
        types: list of 0s (clique) and 1s (ind set) of length k.
        """
        part_masks = [0] * k
        # Track counts for symmetry breaking?
        # If two parts have the same type and the same mask, they are symmetric.
        # We can enforce an ordering on the masks of parts with the same type?
        # e.g., mask[i] < mask[j] if i < j and types[i] == types[j].
        # This is a strong symmetry breaker.
        
        def can_add(p_idx, v):
            m = part_masks[p_idx]
            if types[p_idx] == 0: # Clique
                # v must be adjacent to all in m
                return (m & ~adj[v]) == 0
            else: # Independent Set
                # v must not be adjacent to any in m
                return (m & adj[v]) == 0

        def backtrack(v_idx):
            if time.time() > deadline:
                return False
            if v_idx == n:
                return True
            
            v = order[v_idx]
            
            # Symmetry breaking for empty parts of the same type
            # We keep track of how many times we've seen a specific "state" of empty parts?
            # Simpler: If part i is empty, and there is a previous part j < i with the same type,
            # and part j is also empty, we skip i.
            # This works because if we can put v in an empty part i, we could have put it in j.
            # We enforce that we fill parts in order for empty ones.
            
            last_empty_type = -1
            for i in range(k):
                # Check if part i is empty and if we've already skipped an empty part of this type
                if part_masks[i] == 0:
                    if types[i] == last_empty_type:
                        continue
                    # We only allow the FIRST empty part of a given type to be considered?
                    # No, we allow the first, then skip subsequent ones of the same type.
                    # But we need to update last_empty_type ONLY if we are considering it?
                    # Actually, if we skip it, we don't assign v to it.
                    # If we assign v to it, it's no longer empty.
                    # The standard trick:
                    # If part i is empty, and there exists j < i such that part j is empty and types[j] == types[i],
                    # then assigning v to i is symmetric to assigning v to j.
                    # So we only try assigning v to the FIRST empty part of each type.
                    # Wait, if part j is empty, and we are at vertex v, we haven't assigned v yet.
                    # So part j is still empty.
                    # Thus, for any empty part i, if there is a smaller empty part j of the same type, skip i.
                    
                    is_redundant_empty = False
                    for j in range(i):
                        if part_masks[j] == 0 and types[j] == types[i]:
                            is_redundant_empty = True
                            break
                    if is_redundant_empty:
                        continue
            
            if can_add(i, v):
                part_masks[i] |= (1 << v)
                if backtrack(v_idx + 1):
                    return True
                part_masks[i] &= ~(1 << v)
            
            return False

        if backtrack(0):
            # Convert masks to lists of vertices
            parts = []
            for m in part_masks:
                if m:
                    verts = [v for v in range(n) if (m >> v) & 1]
                    parts.append(verts)
            return parts
        return None

    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Generate type combinations
        # Optimization: We can group by number of cliques?
        # No, just iterate 0 to 2^k - 1
        for mask_types in range(1 << k):
            if time.time() > deadline:
                break
                
            types = []
            for i in range(k):
                if (mask_types >> i) & 1:
                    types.append(1) # Independent
                else:
                    types.append(0) # Clique
            
            res = check_partition(k, types)
            if res is not None:
                # Ensure types are strictly Python lists of ints
                final_res = []
                for part in res:
                    # part is already a list of ints
                    # Ensure it's not numpy or other types
                    clean_part = [int(x) for x in part]
                    final_res.append(clean_part)
                
                # Double check validity to be absolutely sure
                valid = True
                seen_vertices = set()
                for part in final_res:
                    # Check disjoint
                    if not part:
                        valid = False
                        break
                    for v in part:
                        if v in seen_vertices:
                            valid = False
                            break
                        seen_vertices.add(v)
                    if not valid: break
                    
                    # Check clique/ind property
                    # We don't know the type here easily, but we can check both?
                    # No, the search ensured it. But let's verify to catch bugs.
                    # We need to know if it's clique or independent.
                    # Since we don't return types, we can just check if it's a clique OR independent.
                    # If it's neither, it's invalid.
                    is_clique = True
                    is_ind = True
                    for i in range(len(part)):
                        for j in range(i+1, len(part)):
                            u, v = part[i], part[j]
                            if (adj[u] >> v) & 1:
                                is_ind = False
                            else:
                                is_clique = False
                            if not is_clique and not is_ind:
                                break
                        if not is_clique and not is_ind:
                            break
                    if not (is_clique or is_ind):
                        valid = False
                        break
                    
                if valid and len(seen_vertices) == n:
                    return final_res
                else:
                    # If the internal check fails, something is wrong with the logic.
                    # We should not return an invalid partition.
                    pass

    # Fallback: Singletons are always valid (cliques of size 1 are cliques, ind sets of size 1 are ind sets)
    return [[i] for i in range(n)]
