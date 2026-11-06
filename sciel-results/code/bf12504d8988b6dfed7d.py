import random
import time
import sys

def construct():
    n = 20
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bitmask representation for adjacency
    # M[i] is a bitmask of neighbors of vertex i
    
    def can_add(M, u, v):
        """Check if adding edge (u,v) creates a K3,3 subgraph.
        
        A K3,3 exists if there are 3 vertices on one side and 3 on the other
        with all 9 cross-edges present. When adding (u,v), we need to check
        if u and v can be part of a K3,3.
        
        Specifically: if we add (u,v), a K3,3 could form where {u, a, b} is on one side
        and {v, c, d} is on the other, or {v, a, b} and {u, c, d}.
        
        More precisely: adding (u,v) creates K3,3 iff there exist vertices a,b,c,d
        such that:
        - u connects to v, a, b (3 neighbors including v)
        - c,d connect to u, v, a, b... 
        
        Actually, the standard check: adding (u,v) creates K3,3 iff there exist
        two other vertices a,b such that |N(a) ∩ N(b) ∩ N(u) ∩ N(v) \ {u,v}| >= 2
        OR there exist a,b,c where a,b are common neighbors of u and v, and 
        there's a K2,2 between {a,b} and some pair... 
        
        Let me think more carefully. K3,3 has partitions of size 3. After adding (u,v):
        Case 1: u and v are on the same side. Then we need one more vertex w on u's side
                 and 3 vertices x,y,z on v's side (opposite side) such that all of
                 u,w connect to x,y,z and v connects to x,y,z.
                 So: exists w in N(u)\\{v}, and exists x,y,z in N(v)\\{u} such that
                 x,y,z are all in N(u)∩N(w)∩N(v)\\{u,v}.
                 i.e., |N(u) ∩ N(w) ∩ N(v) \\ {u,v}| >= 3 for some w in N(u)\\{v}
                 
        Case 2: u and v are on opposite sides. Then we need 2 more vertices a,b on u's side
                 and 2 more vertices c,d on v's side such that all cross edges exist.
                 So: exists a,b in N(u)\\{v} and c,d in N(v)\\{u} such that
                 a,b connect to c,d and u connects to c,d, v connects to a,b.
                 i.e., a,b in N(u)∩N(v)\\{u,v} (wait, no)
                 
                 Actually: a,b must be neighbors of u (same side as u), and c,d neighbors of v.
                 Also a,b must connect to c,d, and u must connect to c,d, v must connect to a,b.
                 So: c,d in N(u)∩N(v)\\{u,v} (neighbors of both u and v)
                 a,b in N(v) such that a,b connect to c,d. But a,b are on u's side, so a,b in N(u)\\{v}.
                 And a,b must be neighbors of c and d.
                 
                 So: exists c,d in N(u)∩N(v)\\{u,v} and a,b in N(u)\\{v} such that
                 a,b are both in N(c)∩N(d).
                 i.e., |N(c) ∩ N(d) ∩ N(u) \\ {v, c, d}| >= 2 for some c,d in N(u)∩N(v)\\{u,v}
        """
        # Get neighbor sets as bitmasks
        Nu = M[u] & ~(1 << v)  # N(u) without v
        Nv = M[v] & ~(1 << u)  # N(v) without u
        
        # Case 1: u and v on same side
        # Need w in N(u)\{v} such that |N(u) ∩ N(w) ∩ N(v) \ {u,v}| >= 3
        # N(u) ∩ N(v) \ {u,v} is the common neighbors
        common_uv = Nu & Nv  # common neighbors of u and v (excluding each other)
        
        # For each w in N(u)\{v}, check if w shares at least 3 neighbors with u that are also in N(v)
        # That is: |N(w) ∩ N(u) ∩ N(v) \ {u,v,w}| >= 3
        # N(w) ∩ N(u) ∩ N(v) = N(w) ∩ common_uv
        # But we also need to exclude w itself (it's not a neighbor of itself)
        
        temp = Nu
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # Check |N(w) ∩ common_uv| >= 3
            shared = M[w] & common_uv
            # Count bits
            cnt = 0
            s = shared
            while s:
                s &= s - 1
                cnt += 1
                if cnt >= 3:
                    break
            if cnt >= 3:
                return False
            temp ^= lsb
        
        # Case 2: u and v on opposite sides
        # Need c,d in common_uv and a,b in N(u)\{v} such that a,b in N(c)∩N(d)
        # i.e., for some pair c,d in common_uv, |N(c) ∩ N(d) ∩ N(u) \ {v, c, d}| >= 2
        
        # Get list of vertices in common_uv
        cu_list = []
        temp = common_uv
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cu_list.append(idx)
            temp ^= lsb
        
        if len(cu_list) >= 2:
            # For each pair (c,d) in cu_list, check if there are at least 2 vertices in N(u)\{v} 
            # that are neighbors of both c and d
            for i in range(len(cu_list)):
                for j in range(i+1, len(cu_list)):
                    c = cu_list[i]
                    d = cu_list[j]
                    # Vertices in N(c) ∩ N(d) ∩ N(u) \ {v, c, d}
                    candidates = M[c] & M[d] & Nu
                    candidates &= ~(1 << c)
                    candidates &= ~(1 << d)
                    candidates &= ~(1 << v)
                    # Count bits
                    cnt = 0
                    s = candidates
                    while s:
                        s &= s - 1
                        cnt += 1
                        if cnt >= 2:
                            break
                    if cnt >= 2:
                        return False
        
        return True
    
    def count_edges(M):
        return sum(bin(M[i]).count("1") for i in range(n)) // 2
    
    def get_edges(M):
        result = []
        for i in range(n):
            temp = M[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if i < j:
                    result.append([i, j])
                temp ^= lsb
        return result
    
    def greedy_build(order):
        """Build graph by adding edges in given order if K3,3-free."""
        M = [0] * n
        for u, v in order:
            if can_add(M, u, v):
                M[u] |= (1 << v)
                M[v] |= (1 << u)
        return M
    
    def local_search(M, max_iter=500):
        """Try to add edges that were previously rejected, and swap edges."""
        M = M[:]
        edges_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if M[i] & (1 << j):
                    edges_set.add((i, j))
        
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges_set]
        
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try adding a non-edge
            random.shuffle(non_edges)
            for u, v in non_edges:
                if can_add(M, u, v):
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
                    non_edges.remove((u, v))
                    improved = True
                    break
            
            if not improved:
                # Try 1-swap: remove an edge, then add a non-edge
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                for rem_u, rem_v in edge_list[:20]:  # Limit swaps tried
                    # Temporarily remove
                    M[rem_u] &= ~(1 << rem_v)
                    M[rem_v] &= ~(1 << rem_u)
                    
                    # Try to add some non-edges
                    random.shuffle(non_edges)
                    added_any = False
                    for add_u, add_v in non_edges[:30]:
                        if can_add(M, add_u, add_v):
                            M[add_u] |= (1 << add_v)
                            M[add_v] |= (1 << add_u)
                            non_edges.remove((add_u, add_v))
                            added_any = True
                            break
                    
                    if not added_any:
                        # Restore
                        M[rem_u] |= (1 << rem_v)
                        M[rem_v] |= (1 << rem_u)
                    
                    # Rebuild edges_set and non_edges
                    edges_set = set()
                    for i in range(n):
                        for j in range(i+1, n):
                            if M[i] & (1 << j):
                                edges_set.add((i, j))
                    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges_set]
                    
                    if added_any:
                        improved = True
                        break
        
        return M
    
    best_M = [0] * n
    best_count = 0
    
    start_time = time.time()
    
    # Multiple restarts with different strategies
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Strategy: random greedy with random edge ordering
        order = all_edges[:]
        random.shuffle(order)
        
        M = greedy_build(order)
        cnt = count_edges(M)
        
        # Local search to improve
        M = local_search(M, max_iter=100)
        cnt = count_edges(M)
        
        if cnt > best_count:
            best_count = cnt
            best_M = M[:]
        
        # If we've found a good solution, try more intensive local search
        if best_count >= 110:
            # More intensive local search on best
            M2 = local_search(best_M[:], max_iter=300)
            cnt2 = count_edges(M2)
            if cnt2 > best_count:
                best_count = cnt2
                best_M = M2[:]
    
    return get_edges(best_M)

# Quick self-test
if __name__ == "__main__":
    import time
    start = time.time()
    edges = construct()
    elapsed = time.time() - start
    print(f"Found {len(edges)} edges in {elapsed:.1f}s")
    
    # Verify K3,3-free
    n = 20
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all triples
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                common = adj[a] & adj[b] & adj[c]
                if len(common) >= 3:
                    print(f"K3,3 found! Vertices {a},{b},{c} share {len(common)} neighbors")
                    sys.exit(1)
    print("K3,3-free verified")
