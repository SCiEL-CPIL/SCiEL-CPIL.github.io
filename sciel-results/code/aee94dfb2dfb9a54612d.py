def construct():
       n = 16
       # Partition vertices into two sets of 8
       A = list(range(8))
       B = list(range(8, 16))
       
       edges = []
       for u in A:
           for v in B:
               edges.append([u, v])
               
       return edges
