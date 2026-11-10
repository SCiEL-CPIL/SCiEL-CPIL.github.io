"""Q3-free dense graph on 16 vertices: greedy edge addition with on-the-fly cube check."""
import time
import itertools

N = 16
DEADLINE = 135.0

# Q3 edges on canonical vertices 0..7 (bipartition: even {0,2,4,6}, odd {1,3,5,7})
_Q3_CANON = [(0,1),(0,3),(0,5),(2,1),(2,3),(2,7),(4,5),(4,7),(4,1),(6,7),(6,5),(6,3)]
# Let me verify: Q3 is the cube. Vertices 0..7, edge if Hamming distance 1.
# 0=000,1=001,2=010,3=011,4=100,5=101,6=110,7=111
# Edges (dist 1): 0-1,0-2,0-4, 1-3,1-5, 2-3,2-6, 3-7, 4-5,4-6, 5-7, 6-7
_Q3_CANON = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]


def _creates_cube(adj, u, v):
    """Return True if adding edge (u,v) creates a Q3 subgraph.
    u and v must be in opposite bipartition classes of the cube.
    We try both: u in even class, v in odd; and u in odd, v in even.
    """
    return _check_orient(adj, u, v) or _check_orient(adj, v, u)


def _check_orient(adj, even_v, odd_v):
    """even_v in even class, odd_v in odd class. Check if a Q3 containing edge (even_v,odd_v) exists."""
    even_adj = adj[even_v]
    odd_adj = adj[odd_v]
    
    # even_v's other 2 cube-neighbors are in odd class (the odd class has 4 vertices: odd_v + 3 others)
    # even_v connects to odd_v (new) + 2 of the 3 others
    # So among the 3 others in odd class, exactly 2 are current neighbors of even_v.
    
    others = [i for i in range(N) if i != odd_v]
    
    for c1 in others:
        if not (even_adj >> c1 & 1):
            continue
        for c2 in others:
            if c2 <= c1:
                continue
            if not (even_adj >> c2 & 1):
                continue
            for c3 in others:
                if c3 <= c2:
                    continue
                # odd class = {odd_v, c1, c2, c3}
                # even_v connects to c1, c2 (2) + odd_v (new) = 3. 
                # even_v must NOT connect to c3 (else degree 4 in this subgraph... 
                # but actually we just need the 12 cube edges to be present, 
                # and even_v has degree 3 in cube, so it connects to exactly 3 of the 4 odd vertices.
                # It connects to odd_v, c1, c2. So it must NOT connect to c3.
                if even_adj >> c3 & 1:
                    continue
                
                # Now even class = {even_v} + 3 others (e1,e2,e3) from [0..15]\{even_v}\odd_class
                # odd_v connects to even_v (new) + 2 of {e1,e2,e3}
                allowed = [i for i in range(N) if i != even_v and i != odd_v and i != c1 and i != c2 and i != c3]
                
                for e1 in allowed:
                    if not (odd_adj >> e1 & 1):
                        continue
                    for e2 in allowed:
                        if e2 <= e1:
                            continue
                        if not (odd_adj >> e2 & 1):
                            continue
                        for e3 in allowed:
                            if e3 <= e2:
                                continue
                            # even class = {even_v, e1, e2, e3}
                            # odd_v connects to e1, e2 (2) + even_v (new) = 3.
                            # odd_v must NOT connect to e3.
                            if odd_adj >> e3 & 1:
                                continue
                            
                            # Now verify the remaining 8 cube edges (we have even_v-odd_v, even_v-c1, even_v-c2, odd_v-e1, odd_v-e2 = 5 edges accounted for as "present or new")
                            # Actually let's just verify all 12 cube edges are present in adj (treating (even_v,odd_v) as present).
                            # Cube edges in terms of (even_class, odd_class):
                            # even_class = (even_v, e1, e2, e3), odd_class = (odd_v, c1, c2, c3)
                            # In canonical Q3, even class = {0,2,4,6}, odd class = {1,3,5,7}
                            # Canonical edges: 0-1,0-2,0-4, 1-3,1-5, 2-3,2-6, 3-7, 4-5,4-6, 5-7, 6-7
                            # Map: 0->even_v, 2->e1, 4->e2, 6->e3, 1->odd_v, 3->c1, 5->c2, 7->c3
                            # But this mapping must be a valid cube embedding, i.e., the edge structure must match.
                            # The issue: not every assignment of e1,e2,e3 to {2,4,6} and c1,c2,c3 to {3,5,7} gives a valid cube.
                            # The cube structure constrains which even vertex connects to which odd vertices.
                            #
                            # In canonical: 
                            #   0 connects to 1,2,4  (odd_v, c1, c2) -- matches even_v connects to odd_v,c1,c2 ✓
                            #   2 connects to 1,3,6  (odd_v, c1, e3)
                            #   4 connects to 5,6,1  (c2, e3, odd_v)
                            #   6 connects to 7,5,3  (c3, c2, c1)
                            #   1 connects to 0,3,5  (even_v, c1, c2) -- matches odd_v connects to even_v,c1... wait odd_v should connect to even_v,e1,e2
                            #
                            # Hmm, my mapping is inconsistent. Let me redo.
                            # odd_v connects to even_v, e1, e2 (by construction).
                            # In canonical, vertex 1 (odd) connects to 0,3,5.
                            # So map 1->odd_v, 0->even_v, 3->e1, 5->e2.
                            # Then 2 (even) connects to 1,3,6 -> even_v? No, 2 is even, connects to odd 1,3 and even 6? No, Q3 is bipartite, 2 connects to 1,3,6 where 1,3 are odd, 6 is even? That's wrong.
                            #
                            # Let me recheck Q3 edges. 2=010. Neighbors (dist 1): 0=000, 3=011, 6=110. So 2 connects to 0,3,6. 0 is even, 3 is odd, 6 is even. That's not bipartite! 
                            #
                            # I mislabeled. Q3 bipartition: by parity of number of 1s.
                            # 0=000 (0 ones, even), 1=001 (1, odd), 2=010 (1, odd), 3=011 (2, even), 4=100 (1, odd), 5=101 (2, even), 6=110 (2, even), 7=111 (3, odd)
                            # Even class (0 or 2 ones): {0,3,5,6}
                            # Odd class (1 or 3 ones): {1,2,4,7}
                            # Edges (dist 1, so parity flips): 0-1,0-2,0-4, 3-1,3-2,3-7, 5-4,5-1,5-7, 6-4,6-2,6-7
                            # Let me verify: 0(000)-1(001)✓, 0-2(010)✓, 0-4(100)✓, 3(011)-1(001)✓, 3-2(010)✓, 3-7(111)✓, 5(101)-4(100)✓, 5-1(001)✓, 5-7(111)✓, 6(110)-4(100)✓, 6-2(010)✓, 6-7(111)✓. 
                            # So 12 edges. Even class {0,3,5,6}, odd class {1,2,4,7}.
                            
                            # This is getting too complicated to do a fixed mapping. Let me just check all 12 edges directly.
                            # I'll define the cube by: even class E=(even_v,e1,e2,e3), odd class O=(odd_v,c1,c2,c3).
                            # The cube requires a specific 3-regular bipartite graph between E and O.
                            # But there are multiple 3-regular bipartite graphs on 4+4 vertices (the number of 3-regular bipartite graphs on 4+4 is the number of 4x4 0-1 matrices with row/col sum 3 = number of ways to remove a permutation matrix from all-1s = 4! = 24).
                            # So I need to check if the bipartite graph between E and O is one of these 24 (specifically, isomorphic to Q3's bipartite graph, which is the 3-regular one).
                            # Actually ALL 3-regular bipartite graphs on 4+4 vertices are isomorphic to the cube's bipartite graph? No. The cube bipartite graph is a specific one. But any 3-regular bipartite graph on 4+4 has 12 edges. Are they all isomorphic? The number of 3-regular bipartite graphs on 4+4 labeled vertices: it's the number of 4x4 binary matrices with all row/col sums = 3. That equals the number of 4x4 permutation matrices (by complementing) = 4! = 24. And they're all isomorphic as bipartite graphs (since any two permutation matrices are related by row/col permutations). So YES, any 3-regular bipartite graph on 4+4 is isomorphic to Q3's bipartite graph, which means it IS a Q3 (since Q3 is exactly the 3-regular bipartite graph on 4+4... wait, is Q3 the only 3-regular bipartite graph on 4+4? No, but they're all isomorphic, so any one is a Q3).
                            #
                            # Wait, is that true? A 3-regular bipartite graph on 4+4 vertices has 12 edges. Q3 has 12 edges and is 3-regular bipartite on 4+4. Are all such graphs isomorphic to Q3? Q3 is the unique 3-regular bipartite graph on 4+4 up to isomorphism? Let me think: the bipartite adjacency matrix is 4x4 with row/col sum 3, i.e., J - P where P is a permutation matrix. All permutation matrices are permutation-equivalent, so all such graphs are isomorphic. And Q3's bipartite graph is J - P for some P. So YES, any 3-regular bipartite graph on 4+4 is isomorphic to Q3.
                            #
                            # GREAT. So I just need to check: is the bipartite graph between E and O 3-regular (each vertex degree 3)? If so, it's a Q3.
                            
                            # Check degrees:
                            # even_v: connects to odd_v (new), c1, c2 = 3 ✓ (by construction, not c3)
                            # e1: must connect to exactly 3 of {odd_v, c1, c2, c3}
                            # e2: same
                            # e3: same
                            # odd_v: connects to even_v (new), e1, e2 = 3 ✓ (not e3)
                            # c1: must connect to exactly 3 of {even_v, e1, e2, e3}
                            # c2: same
                            # c3: same
                            
                            ok = True
                            for e in (e1, e2, e3):
                                cnt = 0
                                for o in (odd_v, c1, c2, c3):
                                    if adj[e] >> o & 1:
                                        cnt += 1
                                if cnt != 3:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            for o in (c1, c2, c3):
                                cnt = 0
                                for e in (even_v, e1, e2, e3):
                                    if adj[o] >> e & 1:
                                        cnt += 1
                                if cnt != 3:
                                    ok = False
                                    break
                            if ok:
                                return True
    return False


def construct():
    start_time = time.time()
    best_edges = []
    
    # Strategy 1: greedy natural order
    edges = []
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if time.time() - start_time > DEADLINE:
                break
            if _creates_cube(adj, u, v):
                continue
            edges.append([u, v])
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        if time.time() - start_time > DEADLINE:
            break
    if len(edges) > len(best_edges):
        best_edges = edges[:]
    
    # Strategy 2: start from K_{3,13}, add intra-part edges
    if time.time() - start_time < DEADLINE:
        edges = [[a, b] for a in range(3) for b in range(3, 16)]
        adj = [0] * N
        for e in edges:
            adj[e[0]] |= (1 << e[1])
            adj[e[1]] |= (1 << e[0])
        for u in range(3):
            for v in range(u + 1, 3):
                if time.time() - start_time > DEADLINE:
                    break
                if not _creates_cube(adj, u, v):
                    edges.append([u, v])
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        for u in range(3, 16):
            for v in range(u + 1, 16):
                if time.time() - start_time > DEADLINE:
                    break
                if not _creates_cube(adj, u, v):
                    edges.append([u, v])
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        if len(edges) > len(best_edges):
            best_edges = edges[:]
    
    return best_edges
