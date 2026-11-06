import random
import time

def construct():
       start_time = time.time()
       deadline = start_time + 110  # Safety margin for 120s limit
       
       # Try to find a graph for n=46
       # If that fails quickly, try n=45, etc.
       # But n=46 is the target.
       
       for n in [46, 45, 44]:
           min_deg_req = n - 39
           
           # Try multiple random restarts
           for attempt in range(10000):
               if time.time() > deadline:
                   break
                   
               # Initialize graph
               adj = {i: set() for i in range(n)}
               
               # Generate random edge order
               edges = [(i, j) for i in range(n) for j in range(i+1, n)]
               random.shuffle(edges)
               
               # Greedy add edges
               for u, v in edges:
                   # Check if adding (u,v) creates a C4
                   # This happens if N(u) and N(v) share a common neighbor
                   # i.e., there exists w such that w in N(u) and w in N(v)
                   # Since we add edges one by one, N(u) and N(v) are current neighbors.
                   # If w is in N(u) and w is in N(v), then u-w-v-u is a triangle? No.
                   # u-w, w-v are edges. If we add u-v, we get triangle u-v-w.
                   # A C4 is u-v-x-y-u.
                   # Adding edge (u,v) creates a C4 if there is a path of length 3 between u and v in the current graph.
                   # i.e., u-w-x-v.
                   # This means w is in N(u) and x is in N(v) and w-x is an edge?
                   # No.
                   # A C4 containing (u,v) is u-v-w-x-u.
                   # This requires edges (v,w), (w,x), (x,u).
                   # So w must be in N(v), x must be in N(u), and (w,x) must be an edge.
                   # Also w != x.
                   
                   # Efficient check:
                   # Iterate over neighbors of u, say w.
                   # For each w in N(u), check if there is a neighbor x of w such that x is in N(v).
                   # If so, adding (u,v) creates a C4.
                   
                   creates_c4 = False
                   for w in adj[u]:
                       # If w is connected to any neighbor of v, then u-w-x-v-u is a C4?
                       # Path: u-w-x-v. Add u-v. Cycle: u-v-x-w-u.
                       # Yes.
                       # So check if N(w) intersects N(v).
                       for x in adj[w]:
                           if x in adj[v]:
                               creates_c4 = True
                               break
                       if creates_c4:
                           break
                   
                   if not creates_c4:
                       adj[u].add(v)
                       adj[v].add(u)
               
               # Check min degree
               min_deg = min(len(adj[i]) for i in range(n))
               if min_deg >= min_deg_req:
                   # Construct edge list
                   edge_list = []
                   for u in range(n):
                       for v in adj[u]:
                           if u < v:
                               edge_list.append([u, v])
                   return {'n': n, 'edges': edge_list}
               
               # If not successful, try again
           
       # Fallback: return the trivial solution if no better found
       return {'n': 39, 'edges': []}
