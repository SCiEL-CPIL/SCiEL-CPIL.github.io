def construct():
       n = 16
       edges = []
       center = 0
       for v in range(1, n):
           edges.append([center, v])
       return edges
