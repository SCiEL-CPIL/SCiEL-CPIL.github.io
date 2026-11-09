import time

def construct():
    n = 14
    edges = P['edges']
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Order vertices by degree to potentially prune earlier?
    # Let's stick to natural order first, it's often sufficient for n=14.
    # If it's too slow, we can reorder.
    
    def is_clique(part, v):
        for u in part:
            if not adj[u][v]:
                return False
        return True

    def is_independent(part, v):
        for u in part:
            if adj[u][v]:
                return False
        return True

    def search(v, parts, types, k):
        if v == n:
            return True
        
        # Try assigning v to each part j in 0..k-1
        for j in range(k):
            if not parts[j]:
                # Symmetry breaking: only use the first empty part
                if j > 0 and not parts[j-1]:
                    continue
                
                # Check if we can add v to empty part j
                # New part is [v], type is flexible (3)
                parts[j] = [v]
                types[j] = 3
                
                if search(v + 1, parts, types, k):
                    return True
                
                parts[j] = []
                types[j] = 0
            else:
                t = types[j]
                if t == 3: # Flexible (size 1)
                    u = parts[j][0]
                    # Can become clique or independent
                    # Since (u,v) is either edge or not, it's always valid
                    # But we need to update type
                    new_type = 1 if adj[u][v] else 2
                    parts[j].append(v)
                    types[j] = new_type
                    
                    if search(v + 1, parts, types, k):
                        return True
                    
                    parts[j].pop()
                    types[j] = 3
                elif t == 1: # Clique
                    if is_clique(parts[j], v):
                        parts[j].append(v)
                        # Type remains 1
                        if search(v + 1, parts, types, k):
                            return True
                        parts[j].pop()
                elif t == 2: # Independent
                    if is_independent(parts[j], v):
                        parts[j].append(v)
                        # Type remains 2
                        if search(v + 1, parts, types, k):
                            return True
                        parts[j].pop()
        
        return False

    # Try k from 1 to n
    for k in range(1, n + 1):
        parts = [[] for _ in range(k)]
        types = [0] * k
        if search(0, parts, types, k):
            # Filter out empty parts (though symmetry breaking should prevent this if k is minimal?
            # No, if k is larger than needed, some parts might remain empty.
            # But we iterate k from 1, so the first solution found uses exactly k parts?
            # Wait. The symmetry breaking "first empty part" means we fill parts 0, 1, ... sequentially.
            # If we have a solution with k parts, it means we used exactly k parts?
            # Yes, because we can't leave a gap. If part 0 is empty, part 1 must be empty too.
            # So if part j is non-empty, all i<j are non-empty.
            # Thus, the number of non-empty parts is exactly the index of the last non-empty part + 1.
            # Since we check v==n, and we might have used fewer than k parts?
            # No, if we used fewer, say m < k, then parts[m..k-1] are empty.
            # But the symmetry breaking prevents assigning to part j if part j-1 is empty.
            # So if we assign to part m, parts 0..m-1 must be non-empty.
            # If we never assign to part m+1, it stays empty.
            # So the number of used parts is the max index used + 1.
            # If the search succeeds, it means we assigned all vertices to some subset of 0..k-1.
            # Due to symmetry breaking, they must be 0..m-1 for some m <= k.
            # So we might have a solution with m < k parts.
            # But we are iterating k from 1 upwards.
            # If a solution with m parts exists, it would have been found when k=m.
            # So when we reach k, any solution found must use exactly k parts?
            # Not necessarily. It could use m < k parts.
            # But since we iterate k=1,2,..., the first k that succeeds gives the minimum number of parts.
            # However, the solution returned might have empty parts at the end.
            # We should filter them out.
            
            final_parts = [p for p in parts if p]
            # Ensure types are correct lists
            return [list(p) for p in final_parts]
    
    # Fallback (should not happen)
    return [[i] for i in range(n)]

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
