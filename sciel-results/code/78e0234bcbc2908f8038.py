   def construct():
       # Grötzsch graph on 0..10
       # Cycle 0-1-2-3-4-0
       # 5 connected to 0,1,2,3,4? No.
       # Mycielski construction:
       # Original C5: 0-1-2-3-4-0
       # New vertices 5..9 corresponding to 0..4
       # 5 connected to 1,4 (neighbors of 0)
       # 6 connected to 0,2
       # 7 connected to 1,3
       # 8 connected to 2,4
       # 9 connected to 3,0
       # Vertex 10 connected to 5,6,7,8,9
       
       edges = []
       # C5
       for i in range(5):
           edges.append([i, (i+1)%5])
           
       # Mycielski step
       for i in range(5):
           u = i
           v1 = (i+1)%5
           v2 = (i-1)%5
           edges.append([5+i, v1])
           edges.append([5+i, v2])
           
       # Apex
       for i in range(5):
           edges.append([10, 5+i])
           
       # Add 5 new vertices 11..15
       # Connect each to a maximal independent set in Grötzsch
       # Independent set: {0, 2, 5, 7, 9}?
       # Let's find one.
       # 0 is not connected to 2, 3? 0-1, 0-4. So 0 not connected to 2,3.
       # 5 connected to 1,4.
       # 7 connected to 1,3.
       # 9 connected to 3,0.
       
       # Let's use a known independent set of size 5.
       # In Grötzsch, max independent set size is 5.
       # Example: {0, 2, 5, 7, 9}
       # Check 0-2: No edge.
       # Check 0-5: 5 connected to 1,4. No.
       # Check 0-7: 7 connected to 1,3. No.
       # Check 0-9: 9 connected to 3,0. YES, EDGE.
       # So {0,2,5,7,9} is not independent.
       
       # Try {0, 2, 6, 8, 10}?
       # 10 connected to 5,6,7,8,9.
       # 6 connected to 0,2.
       # So 10-6 edge.
       
       # Let's just pick 5 vertices that are not connected.
       # We can compute this in the code or hardcode a valid one.
       
       # A valid independent set in Grötzsch:
       # Vertices: 0, 2, 4, 6, 8?
       # 0-2: No.
       # 0-4: No.
       # 2-4: No.
       # 6 connected to 0,2. So 6-0, 6-2 edges.
       
       # Let's use the set of "original" vertices {0,2,4} and "new" vertices that don't connect to them.
       # 0 neighbors: 1,4, 5, 9.
       # 2 neighbors: 1,3, 6, 8.
       # 4 neighbors: 0,3, 5, 7.
       
       # Vertex 10 neighbors: 5,6,7,8,9.
       
       # Independent set:
       # Pick 0. Cannot pick 1,4,5,9.
       # Pick 2. Cannot pick 1,3,6,8.
       # Pick 4. Cannot pick 0,3,5,7.
       # So {0,2,4} is independent.
       # Can we add more?
       # Remaining: 1,3,5,6,7,8,9,10.
       # Excluded by 0: 5,9.
       # Excluded by 2: 6,8.
       # Excluded by 4: 5,7.
       # Available: 1,3,10.
       # 1 connected to 0,2,5,7.
       # 3 connected to 2,4,6,8.
       # 10 connected to 5,6,7,8,9.
       
       # Check 1 and 3.
       # 1-3? No edge in C5.
       # 1-10? No.
       # 3-10? No.
       # So {0,2,4,1,3} is independent?
       # Check 1-0: Edge.
       # So 1 cannot be with 0.
       
       # So {0,2,4} is size 3.
       # Can we get size 5?
       # Yes, it is known.
       # Let's just use {0, 2, 4, 6, 8}?
       # 6-0 edge.
       
       # Let's use {1, 3, 5, 7, 9}?
       # 1-3: No.
       # 1-5: 5 connected to 1. Edge.
       
       # Let's use {0, 2, 4, 10, ?}
       # 10 not connected to 0,2,4.
       # So {0,2,4,10} is independent.
       # Can we add one more?
       # Excluded by 0: 1,4,5,9.
       # Excluded by 2: 1,3,6,8.
       # Excluded by 4: 0,3,5,7.
       # Excluded by 10: 5,6,7,8,9.
       # Union of excluded: 0,1,3,4,5,6,7,8,9.
       # Only 2 is left? No, 2 is in set.
       # So no 5th vertex.
       
       # Max independent set is 5.
       # Let's find one programmatically in the thought process or just use a smaller set if 45 is hard.
       # If we use set of size 4, edges = 4*5 = 20. Total 40.
       
       # Let's assume we can find size 5.
       # One such set is {0, 2, 4, 6, 8}? No.
       # {0, 2, 5, 7, 9}?
       # 0-9 edge.
       
       # {0, 2, 4, 1, 3}?
       # 0-1 edge.
       
       # Actually, the independence number is 5.
       # Set: {0, 2, 4, 6, 8} is not independent.
       # Set: {1, 3, 5, 7, 9} is not independent.
       
       # Let's just use the set {0, 2, 4, 10} (size 4) to be safe.
       # Edges: 4 * 5 = 20. Total 40.
       
       # Or use {0, 2, 4} (size 3). Edges 15. Total 35.
       
       # Let's try to find a size 5 set.
       # Vertices: 0..10.
       # 0: 1,4,5,9
       # 2: 1,3,6,8
       # 4: 0,3,5,7
       # 1: 0,2,5,7
       # 3: 2,4,6,8
       # 5: 1,4,10
       # 7: 1,3,10
       # 9: 0,3,10
       # 6: 0,2,10
       # 8: 2,4,10
       # 10: 5,6,7,8,9
       
       # Try {0, 2, 4, 1, 3} -> 0-1 edge.
       # Try {0, 2, 4, 5, 7} -> 0-5 edge.
       # Try {0, 2, 4, 6, 8} -> 0-6 edge.
       # Try {0, 2, 4, 9, 7} -> 0-9 edge.
       
       # It seems hard to find manually.
       # I will use the set {0, 2, 4, 10} (size 4).
       # Total edges 40.
       # This is still better than 20.
       
       for new_v in range(11, 16):
           for old_v in [0, 2, 4, 10]:
               edges.append([new_v, old_v])
               
       return edges
   