def construct():
           n = 15
           # Bipartite graph K_{7,8}
           # Part L: 0..6
           # Part R: 7..14
           edges = []
           for u in range(7):
               for v in range(7, 15):
                   edges.append([u, v])
           return edges
