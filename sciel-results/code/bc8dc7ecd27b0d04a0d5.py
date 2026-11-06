"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 145
    n = 16

    # Q3: 8 vertices, 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:
                q3_edges.append((u, v))
    assert len(q3_edges) == 12

    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120

    # Precompute all 24 permutations of 4 elements for Q3 check
    perms4 = list(permutations(range(4)))

    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            qv_to_kv = {0: mapping[0], 1: mapping[1]}
            used = {mapping[0], mapping[1]}
            
            # Q3 neighbors:
            # 0: {1, 2, 4}
            # 1: {0, 3, 5}
            # 2: {0, 3, 6}
            # 3: {1, 2, 7}
            # 4: {0, 5, 6}
            # 5: {1, 4, 7}
            # 6: {2, 4}
            # 7: {3, 5}
            
            # Vertex 2 must be a neighbor of 0 (i.e., mapping[0]) in K16
            # Vertex 4 must be a neighbor of 0 in K16
            # Vertex 3 must be a neighbor of 1 (i.e., mapping[1]) in K16
            # Vertex 5 must be a neighbor of 1 in K16
            
            # Candidates for Q3 vertex 2: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 4: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 3: neighbors of mapping[1] not in used
            # Candidates for Q3 vertex 5: neighbors of mapping[1] not in used
            
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists using bit operations for speed
            cands2 = []
            cands4 = []
            cands3 = []
            cands5 = []
            
            for i in range(n):
                if i in used:
                    continue
                if mask0 & (1 << i):
                    cands2.append(i)
                    cands4.append(i)
                if mask1 & (1 << i):
                    cands3.append(i)
                    cands5.append(i)
            
            # Try all combinations
            for c2 in cands2:
                for c4 in cands4:
                    if c4 == c2:
                        continue
                    for c3 in cands3:
                        if c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands5:
                            if c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                            
                            # Now we have 6 vertices assigned: 0,1,2,3,4,5
                            # Need to assign 6 and 7
                            # Q3 vertex 6 connects to 2 and 4
                            # Q3 vertex 7 connects to 3 and 5
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c2 = adj_bitmasks[c2]
                            mask_c4 = adj_bitmasks[c4]
                            mask_c6 = mask_c2 & mask_c4
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = []
                            for i in range(n):
                                if i not in (mapping[0], mapping[1], c2, c3, c4, c5) and (mask_c6 & (1 << i)):
                                    cands6.append(i)
                            
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c3 = adj_bitmasks[c3]
                            mask_c5 = adj_bitmasks[c5]
                            mask_c7 = mask_c3 & mask_c5
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = []
                            for i in range(n):
                                if i not in (mapping[0], mapping[1], c2, c3, c4, c5) and (mask_c7 & (1 << i)):
                                    cands7.append(i)
                            
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check all Q3 edges
                                    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    # Already checked: (0,1) is the new edge, (0,2),(0,4),(1,3),(1,5) by construction
                                    # Need to check: (2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    
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

    def prune_q3s(adj, edges):
        """Greedy removal of edges to eliminate all Q3s."""
        changed = True
        while changed:
            changed = False
            # Find a Q3
            # We can iterate over all subsets of 8 vertices? Too slow.
            # Instead, iterate over all pairs of edges?
            # Or just check if the graph is Q3-free. If not, find one and remove an edge.
            # To find a Q3, we can use the has_q3_fast logic but on the existing graph.
            # Actually, has_q3_fast checks if ADDING an edge creates a Q3.
            # We need a function that finds an EXISTING Q3.
            
            # Let's write a simple finder:
            # Pick a random edge (u,v). Check if it is part of a Q3.
            # If yes, remove (u,v) or another edge in that Q3.
            
            # For efficiency, we'll just try to remove random edges until Q3-free.
            # This is crude but works if we start dense.
            
            # Better: Use a backtracking search to find one Q3.
            # Since we have time, we can do a few iterations.
            
            # Let's just check all edges. If an edge is in a Q3, remove it.
            # To check if edge (u,v) is in a Q3, we can temporarily remove it and see if has_q3_fast(u,v) returns True?
            # No, has_q3_fast checks if adding (u,v) to a graph WITHOUT (u,v) creates a Q3.
            # So if we remove (u,v) from adj, and then call has_q3_fast(adj_without_uv, u, v), it will tell us if (u,v) completes a Q3 in the rest of the graph.
            
            for u in range(n):
                if time.time() > deadline - 5:
                    return
                for v in range(u+1, n):
                    if time.time() > deadline - 5:
                        return
                    if not (adj[u] & (1 << v)):
                        continue
                    
                    # Temporarily remove (u,v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Check if (u,v) is part of a Q3
                    if has_q3_fast(adj, u, v):
                        # (u,v) is in a Q3. Remove it.
                        edges.discard((u, v))
                        changed = True
                        # We can break early to re-evaluate, or continue
                        # Continuing might remove multiple edges from the same Q3, which is fine.
                    else:
                        # Put it back
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
    random.seed(42)
    
    best_adj = [0] * n
    best_edges = set()
    best_count = 0
    
    # Strategy: Start with dense random graphs, prune, then local search
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        # Generate random graph with p=0.8
        p = 0.8
        adj = [0] * n
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        
        # Prune Q3s
        prune_q3s(adj, edges)
        
        # Local Search: Try to add missing edges
        # We use the same local search as before but starting from a denser state
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            # Try to add a missing edge
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline - 20:
                    break
                u, v = e
                if has_q3_fast(adj, u, v):
                    continue
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add(e)
                improved = True
        
        # Local Search: Swap (remove one, add one)
        if time.time() < deadline - 20:
            swap_improved = True
            while swap_improved and time.time() < deadline - 20:
                swap_improved = False
                # Try removing an edge
                for e_rem in list(edges):
                    if time.time() > deadline - 20:
                        break
                    u, v = e_rem
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.discard(e_rem)
                    
                    # Try to add a missing edge
                    missing = [e for e in all_edges if e not in edges]
                    random.shuffle(missing)
                    
                    for e_add in missing:
                        if time.time() > deadline - 20:
                            break
                        ua, va = e_add
                        if has_q3_fast(adj, ua, va):
                            continue
                        # Add edge
                        adj[ua] |= (1 << va)
                        adj[va] |= (1 << ua)
                        edges.add(e_add)
                        swap_improved = True
                        break # Found a good swap, restart scan
                    
                    # If no good swap found, put e_rem back? 
                    # Or keep it removed if it didn't help?
                    # If we didn't add anything, we effectively removed an edge.
                    # We should only accept if we added something.
                    # So if we didn't break out of inner loop, we should revert?
                    # Actually, the loop breaks if it finds one.
                    # If it doesn't find one, it finishes the loop.
                    # Then we are left with e_rem removed and no new edge added.
                    # This reduces edge count. We should avoid this.
                    # So, if no swap found, revert e_rem.
                    if not swap_improved:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add(e_rem)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
