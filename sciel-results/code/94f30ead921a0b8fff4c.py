import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    # Adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    
    # Order vertices by degree (highest first) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Build reverse mapping: position in order -> vertex
    # We'll assign in the order of vertex_order
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    def backtrack(idx, k, parts, part_types):
        """
        idx: next vertex index in vertex_order to assign
        k: number of parts
        parts: list of lists (parts[i] contains vertices assigned to part i)
        part_types: list of 0 (empty), 1 (clique), 2 (independent), 3 (flexible/size 1)
        """
        if time.time() > deadline:
            return None
        
        if idx == n:
            # All vertices assigned, return the partition
            result = []
            for p in parts:
                if p:
                    result.append(list(p))
            return result
        
        v = vertex_order[idx]
        
        # Try to place v in each part
        # Symmetry breaking: skip duplicate part states
        seen = set()
        
        for i in range(k):
            # Skip if this part state was already tried
            # Use frozenset of part content and type as key
            if len(parts[i]) <= 1:
                key = (frozenset(parts[i]), part_types[i])
            else:
                key = (frozenset(parts[i]), part_types[i])
            
            if key in seen:
                continue
            seen.add(key)
            
            # Check if v can be added to part i
            valid = True
            new_type = part_types[i]
            
            if part_types[i] == 1:  # Clique
                for u in parts[i]:
                    if u not in adj[v]:
                        valid = False
                        break
                if valid:
                    new_type = 1
            elif part_types[i] == 2:  # Independent
                for u in parts[i]:
                    if u in adj[v]:
                        valid = False
                        break
                if valid:
                    new_type = 2
            elif part_types[i] == 3:  # Flexible (size 1)
                if len(parts[i]) == 1:
                    u = parts[i][0]
                    if u in adj[v]:
                        new_type = 1  # Becomes clique
                    else:
                        new_type = 2  # Becomes independent
                else:
                    valid = False
            else:  # Empty part (type 0)
                new_type = 3  # Becomes flexible (size 1)
            
            if not valid:
                continue
            
            # Assign v to part i
            parts[i].append(v)
            old_type = part_types[i]
            part_types[i] = new_type
            
            result = backtrack(idx + 1, k, parts, part_types)
            if result is not None:
                return result
            
            # Unassign
            parts[i].pop()
            part_types[i] = old_type
            
            # If part became empty or size 1, reset type
            if len(parts[i]) == 0:
                part_types[i] = 0
            elif len(parts[i]) == 1:
                part_types[i] = 3
        
        return None
    
    # Try increasing k from 1
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        parts = [[] for _ in range(k)]
        part_types = [0] * k
        
        result = backtrack(0, k, parts, part_types)
        
        if result is not None:
            # Verify the result
            for part in result:
                is_clique = True
                is_indep = True
                for i in range(len(part)):
                    for j in range(i + 1, len(part)):
                        if (part[i], part[j]) in edge_set:
                            is_indep = False
                        else:
                            is_clique = False
                if not (is_clique or is_indep):
                    raise ValueError("Invalid partition found")
            
            # Convert to proper Python types
            return [list(map(int, part)) for part in result]
    
    # Fallback: should not reach here
    return [[int(i)] for i in range(n)]
