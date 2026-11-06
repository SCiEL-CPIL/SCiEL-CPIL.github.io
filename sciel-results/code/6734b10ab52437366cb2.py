def _find_q3(edges_set, adj):
    n = 16
    vertices = list(range(n))
    
    for subset in combinations(vertices, 8):
        # 1. Min degree check
        min_deg = 8
        for u in subset:
            deg = sum(1 for v in subset if v != u and (min(u,v), max(u,v)) in edges_set)
            if deg < min_deg:
                min_deg = deg
            if min_deg < 3:
                break
        if min_deg < 3:
            continue
        
        subset_list = list(subset)
        
        # 2. Check bipartitions
        for A in combinations(subset_list, 4):
            B = tuple(v for v in subset_list if v not in A)
            
            # Quick check: does any vertex in A connect to all 4 in B?
            # If yes, no non-edge matching possible.
            valid_bipartition = True
            non_edge_matrix = [[False]*4 for _ in range(4)]
            
            for i, a in enumerate(A):
                deg_a = 0
                for j, b in enumerate(B):
                    if (min(a,b), max(a,b)) in edges_set:
                        deg_a += 1
                    else:
                        non_edge_matrix[i][j] = True
                
                if deg_a == 4:
                    valid_bipartition = False
                    break
            
            if not valid_bipartition:
                continue
                
            # Also check B side? If b connects to all 4 in A, then column is all False in non_edge_matrix.
            # The matching check will handle it, but we can prune.
            for j, b in enumerate(B):
                deg_b = 0
                for i, a in enumerate(A):
                    if (min(a,b), max(a,b)) in edges_set:
                        deg_b += 1
                if deg_b == 4:
                    valid_bipartition = False
                    break
                    
            if not valid_bipartition:
                continue
            
            # Check for perfect matching in non_edge_matrix
            # Brute force 24 perms
            for perm in permutations(range(4)):
                if all(non_edge_matrix[i][perm[i]] for i in range(4)):
                    return list(subset)
                    
    return None
