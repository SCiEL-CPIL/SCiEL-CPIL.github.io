   def check_q3_in_subset(vs, edge_set):
       # vs is a tuple of 8 vertices
       # Map vs to 0..7
       idx_map = {v: i for i, v in enumerate(vs)}
       
       # Build adjacency as bitmasks or sets for fast lookup
       adj = [0]*8
       for i in range(8):
           for j in range(i+1, 8):
               u, w = vs[i], vs[j]
               if (min(u, w), max(u, w)) in edge_set:
                   adj[i] |= (1 << j)
                   adj[j] |= (1 << i)
       
       # Quick prune: degree < 3
       for i in range(8):
           if bin(adj[i]).count('1') < 3:
               return False
               
       # Q3 structure
       # Even vertices: 0, 2, 4, 6
       # Odd vertices: 1, 3, 5, 7
       # Edges from even u to odd v if u^v has popcount 1
       
       # Iterate over all ways to assign 4 vertices from vs to Even(Q3)
       # and the remaining 4 to Odd(Q3)
       from itertools import combinations, permutations
       
       # Precompute Q3 edges between Even and Odd
       q3_even = [0, 2, 4, 6]
       q3_odd = [1, 3, 5, 7]
       q3_edges_eo = []
       for e in q3_even:
           for o in q3_odd:
               if bin(e ^ o).count('1') == 1:
                   q3_edges_eo.append((e, o))
       
       # For each subset of 4 from 8 to be the image of q3_even
       for A in combinations(range(8), 4):
           B = tuple(x for x in range(8) if x not in A)
           
           # Check if there is a permutation of A and B that satisfies the edges
           # We can fix the mapping of the first even vertex to reduce symmetry?
           # Or just try all 4! * 4! = 576 permutations.
           # 70 subsets * 576 perms * ~12 edge checks = ~480k ops.
           # This might be slightly slow if called 12870 times (all 8-sets).
           # 12870 * 480k = 6e9 ops. Too slow.
           
           # Optimization:
           # The degrees in Q3 are all 3.
           # So in the induced bipartite graph between A and B,
           # every vertex in A must have degree >= 3 within B?
           # And every vertex in B must have degree >= 3 within A?
           # If not, this partition A,B cannot support a Q3 subgraph.
           
           # Check degrees within the bipartite cut (A, B)
           valid_partition = True
           for u in A:
               # Count neighbors in B
               deg = 0
               for v in B:
                   if adj[u] & (1 << v):
                       deg += 1
               if deg < 3:
                   valid_partition = False
                   break
           if not valid_partition:
               continue
           
           for v in B:
               deg = 0
               for u in A:
                   if adj[v] & (1 << u):
                       deg += 1
               if deg < 3:
                   valid_partition = False
                   break
           if not valid_partition:
               continue
           
           # Now try permutations
           # We can prune permutations by checking edges early.
           # Let perm_A be a permutation of A, perm_B be a permutation of B.
           # Map q3_even[i] -> perm_A[i], q3_odd[j] -> perm_B[j]
           
           # To speed up, we can pick a specific edge in Q3, say (0, 1).
           # 0 is in q3_even, 1 is in q3_odd.
           # They must be mapped to an edge in G.
           
           # Actually, let's just use the 576 permutations but break early.
           for perm_A in permutations(A):
               for perm_B in permutations(B):
                   ok = True
                   for e, o in q3_edges_eo:
                       # e is index in q3_even (0..3), o is index in q3_odd (0..3)
                       # Wait, q3_even is a list [0,2,4,6].
                       # We need to map the *values* 0,2,4,6 to perm_A.
                       # Let's map value 0 to perm_A[0], 2 to perm_A[1], etc.
                       # So index in perm_A for value v is q3_even.index(v).
                       ua = perm_A[q3_even.index(e)]
                       ub = perm_B[q3_odd.index(o)]
                       if not (adj[ua] & (1 << ub)):
                           ok = False
                           break
                   if ok:
                       return True
       return False
   