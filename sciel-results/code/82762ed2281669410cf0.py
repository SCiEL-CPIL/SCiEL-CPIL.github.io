   def construct(k, g):
       # Use Lazebnik-Ustimenko-Woldar graph CD(d, q)
       # Girth is 2d+2 for q >= d
       # We need 2d+2 >= g => d >= (g-2)/2
       # d must be integer.
       d = max(1, (g - 1) // 2)
       
       # If k < d, LUW girth guarantee might fail or be lower.
       # For k=4, g=10, d=4, k>=d holds.
       # If k < d, we might need to increase d? No, increasing d increases n.
       # If k < d, the graph might have shorter cycles.
       # For the specific task k=4, g=10, this is fine.
       
       q = k
       
       # GF(q) arithmetic
       if q == 4:
           # GF(4) representation
           # 0:00, 1:01, 2:11, 3:10? Or standard xor table.
           # Seed used:
           # table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
           # This is addition in GF(4)?
           # Let's check: 1+1=2, 1+2=3, 2+2=0?
           # table[1][1]=1? No, table[1][1] is 1.
           # Wait, the seed table:
           # row 0: 0,0,0,0
           # row 1: 0,1,2,3
           # row 2: 0,2,3,1
           # row 3: 0,3,1,2
           # This looks like addition mod 4? No, 2+2=0? table[2][2]=3.
           # This is a specific GF(4) table.
           # I will reuse the seed's GF(4) logic to be safe.
           table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
           add = lambda x, y: table[x][y] # Wait, seed used x^y for add?
           # Seed: add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
           # So add is XOR.
           # Is XOR addition in GF(4)?
           # In GF(4), characteristic is 2, so addition is XOR.
           # Yes.
           add = lambda x, y: x ^ y
           sub = lambda x, y: x ^ y
           mul = lambda x, y: table[x][y]
       else:
           # Assume prime q
           add = lambda x, y: (x + y) % q
           sub = lambda x, y: (x - y) % q
           mul = lambda x, y: (x * y) % q
           
       # Rule for LUW
       # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
       # The seed code had a specific rule generation.
       # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
       # for c in range(5, d + 1): ...
       
       # I will copy the seed's rule generation but ensure it works for d=4.
       rule = {}
       # Standard LUW rules:
       # For c=2: l[2] - p[2] = l[1] * p[1] ?
       # The seed used:
       # rule[2] = (1, 1)
       # rule[3] = (2, 1)
       # rule[4] = (1, 2)
       
       # Let's just use the seed's rule logic.
       
       rule[2] = (1, 1)
       if d >= 3: rule[3] = (2, 1)
       if d >= 4: rule[4] = (1, 2)
       for c in range(5, d + 1):
           r, c0 = (c - 5) % 4, c - (c - 5) % 4
           rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
           
       def neighbours(vertex):
           side, x = vertex
           for first in range(q):
               y = [None, first] + [0] * (d - 1)
               for c in range(2, d + 1):
                   a, b = rule[c]
                   if side == 0:
                       y[c] = add(x[c], mul(y[a], x[b]))
                   else:
                       y[c] = sub(x[c], mul(x[a], y[b]))
               yield 1 - side, tuple(y)
               
       start = (0, (None,) + (0,) * d)
       index, order, edges = {start: 0}, [start], []
       for vertex in order:
           for other in neighbours(vertex):
               if other not in index:
                   index[other] = len(order)
                   order.append(other)
               if vertex[0] == 0:
                   edges.append((index[vertex], index[other]))
                   
       return len(order), edges
   