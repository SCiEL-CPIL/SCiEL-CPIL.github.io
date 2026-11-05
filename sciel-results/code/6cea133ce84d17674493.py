import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 110
       
       n = 16
       best_edges = []
       best_count = 0
       
       all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
       
       def is_3_colorable(adj):
           # adj is list of bitmasks
           # 1. Order vertices by degree descending
           degs = [bin(adj[i]).count('1') for i in range(n)]
           order = sorted(range(n), key=lambda i: degs[i], reverse=True)
           
           # colors[i] = -1, 0, 1, 2
           colors = [-1] * n
           # color_masks[c] = bitmask of vertices with color c
           color_masks = [0, 0, 0]
           
           # Map original index to position in order?
           # We process vertices in 'order'.
           # We need to check conflicts with already colored neighbors.
           # Since we process in fixed order, we only check neighbors that are earlier in 'order'.
           # Wait, if we process in order, a neighbor might be later.
           # But if a neighbor is later, it is uncolored, so no conflict.
           # So we only check neighbors that are ALREADY COLORED.
           
           def backtrack(idx_in_order):
               if idx_in_order == n:
                   return True
               
               v = order[idx_in_order]
               
               # Determine available colors
               # c is available if (adj[v] & color_masks[c]) == 0
               avail = []
               for c in range(3):
                   if (adj[v] & color_masks[c]) == 0:
                       avail.append(c)
                       
               if not avail:
                   return False
               
               # Try colors
               # Heuristic: try color that appears least? Or just 0,1,2
               for c in avail:
                   colors[v] = c
                   color_masks[c] |= (1 << v)
                   
                   if backtrack(idx_in_order + 1):
                       return True
                   
                   colors[v] = -1
                   color_masks[c] &= ~(1 << v)
               
               return False
               
           return backtrack(0)

       while time.time() < deadline:
           adj = [0] * n
           edge_count = 0
           
           # Shuffle
           random.shuffle(all_edges)
           
           for u, v in all_edges:
               if (adj[u] & adj[v]) == 0:
                   adj[u] |= (1 << v)
                   adj[v] |= (1 << u)
                   edge_count += 1
                   
           if not is_3_colorable(adj):
               if edge_count > best_count:
                   best_count = edge_count
                   best_edges = []
                   for u in range(n):
                       mask = adj[u]
                       while mask:
                           lsb = mask & -mask
                           v = lsb.bit_length() - 1
                           if u < v:
                               best_edges.append([u, v])
                           mask -= lsb
                   
       if best_count == 0:
           # Fallback Clebsch
           clebsch_edges = []
           for u in range(n):
               for bit_idx in range(4):
                   v = u ^ (1 << bit_idx)
                   if u < v:
                       clebsch_edges.append([u, v])
               v = u ^ 15
               if u < v:
                   clebsch_edges.append([u, v])
           best_edges = clebsch_edges
           
       return best_edges
