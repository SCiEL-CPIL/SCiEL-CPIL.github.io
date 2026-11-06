"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 110
    n = 16

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)  # 120

    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    # Q3 neighbors:
    # 0: {1, 2, 4}
    # 1: {0, 3, 5}
    # 2: {0, 3, 6}
    # 3: {1, 2, 7}
    # 4: {0, 5, 6}
    # 5: {1, 4, 7}
    # 6: {2, 4, 7}
    # 7: {3, 5, 6}

    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists
            cands0 = [i for i in range(n) if mask0 & (1 << i)]
            cands1 = [i for i in range(n) if mask1 & (1 << i)]
            
            # Q3 vertex 2 must be neighbor of 0 (mapping[0])
            # Q3 vertex 4 must be neighbor of 0 (mapping[0])
            # Q3 vertex 3 must be neighbor of 1 (mapping[1])
            # Q3 vertex 5 must be neighbor of 1 (mapping[1])
            
            for c2 in cands0:
                for c4 in cands0:
                    if c4 == c2:
                        continue
                    for c3 in cands1:
                        if c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands1:
                            if c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c6 = adj_bitmasks[c2] & adj_bitmasks[c4]
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = [i for i in range(n) if mask_c6 & (1 << i)]
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c7 = adj_bitmasks[c3] & adj_bitmasks[c5]
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = [i for i in range(n) if mask_c7 & (1 << i)]
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check remaining Q3 edges:
                                    # (2,3): c2-c3
                                    if not (adj_bitmasks[c2] & (1 << c3)):
                                        continue
                                    # (2,6): c2-c6
                                    if not (adj_bitmasks[c2] & (1 << c6)):
                                        continue
                                    # (3,7): c3-c7
                                    if not (adj_bitmasks[c3] & (1 << c7)):
                                        continue
                                    # (4,5): c4-c5
                                    if not (adj_bitmasks[c4] & (1 << c5)):
                                        continue
                                    # (4,6): c4-c6
                                    if not (adj_bitmasks[c4] & (1 << c6)):
                                        continue
                                    # (5,7): c5-c7
                                    if not (adj_bitmasks[c5] & (1 << c7)):
                                        continue
                                    # (6,7): c6-c7
                                    if not (adj_bitmasks[c6] & (1 << c7)):
                                        continue
                                    
                                    return True
        return False

    # Build initial graph: K3 on {0,1,2} + K_{3,13}
    def build_initial():
        adj = [0] * n
        edges = set()
        # K3 on {0,1,2}
        for a in range(3):
            for b in range(a+1, 3):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        # K_{3,13}: connect {0,1,2} to {3..15}
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        return adj, edges

    best_adj, best_edges = build_initial()
    best_count = len(best_edges)
    
    # Vertices in B = {3..15}
    b_vertices = list(range(3, 16))
    b_edges = [(b_vertices[i], b_vertices[j]) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    
    # Greedy + local search with multiple restarts
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        
        adj, edges = build_initial()
        
        # Greedy: try to add B-B edges in random order
        missing = [e for e in b_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 20:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        # Local search: try edge swaps (remove one, add another)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            
            for e_remove in list(edges):
                if time.time() > deadline - 15:
                    break
                u, v = e_remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e_remove)
                
                # Try to add a different edge
                for ne in missing:
                    if time.time() > deadline - 15:
                        break
                    nu, nv = ne
                    if (nu, nv) in edges or (nv, nu) in edges:
                        continue
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break  # Only try one addition per removal
        
        # Also try pure addition (no removal) in case we can still add edges
        if time.time() < deadline - 15:
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            for e in missing:
                if time.time() > deadline - 15:
                    break
                u, v = e
                if has_q3_fast(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add(e)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
                    best_adj = adj[:]

    # Final polish: try all possible single-edge additions
    if time.time() < deadline - 5:
        adj = best_adj[:]
        edges = set(best_edges)
        missing = [e for e in all_edges if e not in edges]
        for e in missing:
            if time.time() > deadline - 5:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]

    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
