import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # We search for the minimum k
    for k in range(1, n + 1):
        if time.time() > deadline:
            # Fallback to greedy if timeout (unlikely for N=14)
            break
            
        # Backtracking search for partition into k sets
        # Buckets: list of dicts or tuples
        # Each bucket: (vertices_list, is_clique, is_indep)
        # Initially all empty
        
        # To break symmetry, we enforce that if we put a vertex in bucket j,
        # and bucket j was empty, then all buckets j+1..k-1 must also be empty.
        # This means we fill buckets in order 0, 1, ...
        
        buckets = []
        for i in range(k):
            buckets.append([])
            
        # is_clique[i], is_indep[i]
        # Empty bucket: True, True
        # We track state implicitly by checking vertices each time? 
        # No, better to track flags to avoid re-checking all pairs every time.
        # But N is small. Re-checking is fine.
        # Let's just re-check validity of the bucket when we add a vertex.
        
        def is_valid_bucket(vertices, is_c, is_i):
            # This function is not used if we do incremental updates,
            # but let's stick to incremental updates in the recursion.
            pass

        # Recursive search
        # idx: index of vertex to assign (0 to n-1)
        # buckets: list of lists
        # c_flags: list of bools (True if still potentially clique)
        # i_flags: list of bools (True if still potentially indep)
        
        def dfs(idx, buckets, c_flags, i_flags):
            if time.time() > deadline:
                return None
                
            if idx == n:
                # Check if all buckets are non-empty? 
                # The problem says "partition of vertices ... into nonempty lists".
                # So all k buckets must be non-empty.
                for i in range(k):
                    if not buckets[i]:
                        return None
                # Return a copy of the buckets
                return [list(b) for b in buckets]
                
            v = idx
            # Try assigning v to bucket 0..k-1
            # Symmetry breaking:
            # If we assign v to bucket j, and buckets[j] is empty,
            # then buckets[j+1]..buckets[k-1] must be empty.
            
            for j in range(k):
                # Check symmetry constraint
                if not buckets[j]:
                    # If bucket j is empty, we can only use it if all subsequent buckets are also empty
                    # This ensures we fill buckets in order 0, 1, ...
                    # If bucket j is empty, but bucket j+1 is not, we skip j.
                    # Wait, if bucket j is empty, it means we haven't put anything in it yet.
                    # If we put v in j, it becomes non-empty.
                    # If we skip j, we put v in j+1?
                    # If bucket j is empty, and we put v in j+1 (which might be non-empty or empty),
                    # we are leaving a hole.
                    # Standard symmetry break:
                    # We can only put v in bucket j if:
                    # 1. buckets[j] is non-empty, OR
                    # 2. buckets[j] is empty AND all buckets i > j are empty.
                    # This ensures that the sequence of "first vertices" in buckets is increasing?
                    # No, it ensures that we don't have an empty bucket followed by a non-empty one.
                    # i.e., the non-empty buckets form a prefix 0..m-1.
                    
                    if not all(not buckets[i] for i in range(j + 1, k)):
                        continue
                        
                # Check if v can be added to bucket j
                # Current state: buckets[j], c_flags[j], i_flags[j]
                
                # Check adjacency with all u in buckets[j]
                new_c = c_flags[j]
                new_i = i_flags[j]
                
                valid = True
                for u in buckets[j]:
                    if adj[v][u]:
                        # Edge exists
                        new_i = False
                    else:
                        # No edge
                        new_c = False
                        
                    if not new_c and not new_i:
                        valid = False
                        break
                        
                if valid:
                    # Assign v to bucket j
                    buckets[j].append(v)
                    c_flags[j] = new_c
                    i_flags[j] = new_i
                    
                    result = dfs(idx + 1, buckets, c_flags, i_flags)
                    if result is not None:
                        return result
                        
                    # Backtrack
                    buckets[j].pop()
                    c_flags[j] = c_flags[j] # Restore? No, we passed old values? 
                    # Wait, I modified c_flags[j] in place.
                    # I need to restore the old values.
                    # Let's store old values before modification.
                    # Actually, I set c_flags[j] = new_c.
                    # I need to set it back to the value before this iteration.
                    # Let's refactor to pass copies or store old state.
                    
            return None

        # Refactor DFS to handle state restoration correctly
        def dfs_v2(idx, buckets, c_flags, i_flags):
            if time.time() > deadline:
                return None
                
            if idx == n:
                for i in range(k):
                    if not buckets[i]:
                        return None
                return [list(b) for b in buckets]
                
            v = idx
            
            for j in range(k):
                # Symmetry: if bucket j is empty, all j+1..k-1 must be empty
                if not buckets[j]:
                    # Check if any subsequent bucket is non-empty
                    skip = False
                    for i in range(j + 1, k):
                        if buckets[i]:
                            skip = True
                            break
                    if skip:
                        continue
                        
                # Save old state
                old_c = c_flags[j]
                old_i = i_flags[j]
                
                new_c = old_c
                new_i = old_i
                valid = True
                
                for u in buckets[j]:
                    if adj[v][u]:
                        new_i = False
                    else:
                        new_c = False
                    if not new_c and not new_i:
                        valid = False
                        break
                        
                if valid:
                    buckets[j].append(v)
                    c_flags[j] = new_c
                    i_flags[j] = new_i
                    
                    result = dfs_v2(idx + 1, buckets, c_flags, i_flags)
                    if result is not None:
                        return result
                        
                    buckets[j].pop()
                    c_flags[j] = old_c
                    i_flags[j] = old_i
                    
            return None

        # Initialize state
        init_buckets = [[] for _ in range(k)]
        init_c = [True] * k
        init_i = [True] * k
        
        sol = dfs_v2(0, init_buckets, init_c, init_i)
        
        if sol is not None:
            # Ensure types are int
            final_sol = [[int(x) for x in part] for part in sol]
            return final_sol
            
    # If no solution found (should not happen for k=n), return greedy fallback
    # Greedy fallback from original code
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
