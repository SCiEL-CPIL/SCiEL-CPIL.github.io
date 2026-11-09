import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    # Start with complete 4-partite graph T4(16): parts of size 4
    # Part i: vertices {4*i, 4*i+1, 4*i+2, 4*i+3}
    # Edge exists iff vertices are in different parts
    part = [i // 4 for i in range(16)]
    
    edges = set()
    adj_mask = [0] * 16
    for i in range(16):
        for j in range(i + 1, 16):
            if part[i] != part[j]:
                edges.add((i, j))
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute all 8-sets for each pair (u,v) that could form Q3
    # For adding edge (u,v): u and v must be in opposite partitions A,B
    # So we need 3 more in A from V\{u,v} and 3 in B from V\{u,v}
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # u and v are in opposite partitions
        # Case 1: u in A, v in B
        # Case 2: v in A, u in B
        # By symmetry, just check one case (swap u,v if needed)
        
        ov = [x for x in range(16) if x != u and x != v]
        
        # For each choice of 3 vertices to join u in A,
        # and 3 vertices to join v in B, check if missing edges form partial matching
        
        # Optimization: pre-filter. The 4x4 grid between A and B must have
        # missing edges forming a partial matching (at most 4 missing, no two share a row or col)
        
        for s_a in itertools.combinations(ov, 3):
            A = (u, s_a[0], s_a[1], s_a[2])
            s_a_set = set(s_a)
            remaining = [x for x in ov if x not in s_a_set]
            for s_b in itertools.combinations(remaining, 3):
                B = (v, s_b[0], s_b[1], s_b[2])
                
                # Check partial matching condition on missing edges
                rows_used = 0  # bitmask of rows (A indices) with missing edge
                cols_used = 0  # bitmask of cols (B indices) with missing edge
                valid = True
                
                for ia in range(4):
                    a = A[ia]
                    for ib in range(4):
                        b = B[ib]
                        if not (adj_mask[a] & (1 << b)):
                            # Missing edge between a and b
                            if (rows_used >> ia) & 1 or (cols_used >> ib) & 1:
                                valid = False
                                break
                            rows_used |= (1 << ia)
                            cols_used |= (1 << ib)
                    if not valid:
                        break
                
                if valid:
                    return True
        
        return False
    
    # Try to add all non-edges in systematic order
    # Order: by sum of parts (edges within same part first, then cross-part)
    # Actually, start from current state and try to add missing edges
    
    non_edges = [(i, j) for i in range(16) for j in range(i + 1, 16) if (i, j) not in edges]
    
    # Sort by some heuristic: try edges that connect vertices with high "potential"
    # For now, just try in order
    for (u, v) in non_edges:
        if time.time() > deadline - 5:
            break
        
        if creates_q3(u, v):
            continue
        
        # Safe to add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    return [list(e) for e in edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
