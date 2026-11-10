   def can_add(p, S, dist_sets):
       # Case 1 & 3: Existing points are apex
       for a in S:
           dx = p[0] - a[0]
           dy = p[1] - a[1]
           d2 = dx*dx + dy*dy
           if d2 in dist_sets[a]:
               return False
       
       # Case 2: P is apex
       # Check if any distance from p to a in S is repeated
       seen_dists = set()
       for a in S:
           dx = p[0] - a[0]
           dy = p[1] - a[1]
           d2 = dx*dx + dy*dy
           if d2 in seen_dists:
               return False
           seen_dists.add(d2)
           
       return True
   