   def _find_longest_path(adj, remaining_edges, time_limit):
       # adj is dict of sets
       # remaining_edges is set of frozensets
       
       best_path = []
       start_time = time.time()
       
       # Try starting from each vertex
       for start in list(adj.keys()):
           if time.time() - start_time > time_limit:
               break
           
           path = [start]
           visited_edges = set()
           
           def dfs(node, current_path, current_edges):
               nonlocal best_path
               if time.time() - start_time > time_limit:
                   return
               
               # Update best if longer
               if len(current_path) > len(best_path):
                   best_path = list(current_path)
                   
               # Try neighbors
               for nb in adj.get(node, []):
                   e = frozenset((node, nb))
                   if e in current_edges or e not in remaining_edges:
                       continue
                   
                   # Heuristic: Order neighbors by degree in remaining graph?
                   # For now, just iterate
                   new_path = current_path + [nb]
                   new_edges = current_edges | {e}
                   dfs(nb, new_path, new_edges)
                   
           dfs(start, path, set())
           
       return best_path
   