   s = target
   while s > 0:
       prev_s, d = parent[s]
       chosen.append(d)
       s = prev_s
   