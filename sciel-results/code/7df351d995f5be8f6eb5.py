import time
import random

def construct():
       start_time = time.time()
       deadline = 110 # 10s margin

       # Try q=9
       try:
           n, edges = generate_polarity_graph(9)
           if time.time() < deadline:
               # Check if alpha < 11
               if not has_independent_set_11(n, edges):
                   return {'n': n, 'edges': edges}
       except Exception:
           pass

       # Try q=7
       try:
           n, edges = generate_polarity_graph(7)
           if time.time() < deadline:
               if not has_independent_set_11(n, edges):
                   return {'n': n, 'edges': edges}
       except Exception:
           pass

       # Try Hoffman-Singleton
       try:
           n, edges = generate_hoffman_singleton()
           if time.time() < deadline:
               if not has_independent_set_11(n, edges):
                   return {'n': n, 'edges': edges}
       except Exception:
           pass

       # Fallback to seed
       edges = []
       for t in range(10):
           a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
           edges += [[a, b], [a, c], [b, c]]
       return {'n': 30, 'edges': edges}
