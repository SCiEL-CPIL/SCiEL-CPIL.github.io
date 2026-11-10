     def get_clebsch_edges():
         edges = []
         # Vertices 0-15
         # u ~ v if (u ^ v) in {3, 5, 6, 9, 10} ?
         # Let's test masks of weight 2.
         masks = [3, 5, 6, 9, 10] # 0011, 0101, 0110, 1001, 1010
         for u in range(16):
             for v in range(u+1, 16):
                 if (u ^ v) in masks:
                     edges.append([u, v])
         return edges
     