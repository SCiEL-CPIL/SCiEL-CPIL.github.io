import time
import random
import sys

def construct():
       start_time = time.time()
       deadline = start_time + 110
       
       n = 20
       best_edges = []
       best_count = 0
       
       def try_add_edge(u, v, adj, cc):
           # Check if adding (u, v) is valid
           if adj[u][v]:
               return False
           
           # 1. cc(u, v) <= 2
           if cc[u][v] > 2:
               return False
               
           # 2. For all w in N(u), cc(v, w) <= 1
           for w in range(n):
               if adj[u][w]:
                   if cc[v][w] > 1:
                       return False
                   
           # 3. For all w in N(v), cc(u, w) <= 1
           for w in range(n):
               if adj[v][w]:
                   if cc[u][w] > 1:
                       return False
                   
           return True

       def add_edge(u, v, adj, cc):
           adj[u][v] = True
           adj[v][u] = True
           
           # Update cc
           # For w in N(u): cc(v, w) increases by 1
           for w in range(n):
               if adj[u][w]:
                   cc[v][w] += 1
                   cc[w][v] += 1
                   
           # For w in N(v): cc(u, w) increases by 1
           for w in range(n):
               if adj[v][w]:
                   cc[u][w] += 1
                   cc[w][u] += 1

       def remove_edge(u, v, adj, cc):
           adj[u][v] = False
           adj[v][u] = False
           
           # Reverse update
           for w in range(n):
               if adj[u][w]: # Note: adj[u][v] is now False, so w != v
                   cc[v][w] -= 1
                   cc[w][v] -= 1
                   
           for w in range(n):
               if adj[v][w]:
                   cc[u][w] -= 1
                   cc[w][u] -= 1

       for restart in range(100):
           if time.time() > deadline:
               break
               
           adj = [[False] * n for _ in range(n)]
           cc = [[0] * n for _ in range(n)]
           
           # Greedy addition
           edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
           random.shuffle(edges_to_try)
           
           current_count = 0
           for u, v in edges_to_try:
               if time.time() > deadline:
                   break
               if try_add_edge(u, v, adj, cc):
                   add_edge(u, v, adj, cc)
                   current_count += 1
                   
           if current_count > best_count:
               best_count = current_count
               best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
               
           # Optional: Local search to add more edges
           # Try adding random non-edges
           improvements = True
           while improvements and time.time() < deadline:
               improvements = False
               non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
               if not non_edges:
                   break
               random.shuffle(non_edges)
               for u, v in non_edges:
                   if time.time() > deadline:
                       break
                   if try_add_edge(u, v, adj, cc):
                       add_edge(u, v, adj, cc)
                       current_count += 1
                       improvements = True
           
           if current_count > best_count:
               best_count = current_count
               best_edges = [[u, v] for u in range(n) for j in range(u+1, n) if adj[u][v]]

       return best_edges
