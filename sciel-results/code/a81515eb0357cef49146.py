import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 110
       
       n = 18
       # Init K_6,6,6
       adj = [0] * n
       parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
       for i in range(3):
           for j in range(i+1, 3):
               for u in parts[i]:
                   for v in parts[j]:
                       adj[u] |= (1 << v)
                       adj[v] |= (1 << u)
                       
       def calc_score(adj):
           # B&B
           best = 0
           # Order vertices by degree?
           # Sort by popcount(adj[v])
           # Vertices with high degree are more likely to form triangles.
           # We want to find large triangle-free set.
           # Greedy: pick vertex with min degree in remaining?
           
           # Let's use a simple recursive search with memoization?
           # State is (rem, curr_set) -> too big.
           
           # Just standard B&B.
           def rec(rem, curr_size, curr_set):
               nonlocal best
               if not rem:
                   if curr_size > best:
                       best = curr_size
                   return
                   
               # Prune
               if curr_size + rem.bit_count() <= best:
                   return
                   
               # Pick v
               # Heuristic: pick v that is "hard" to add?
               # Or pick v that is isolated in rem?
               
               # Let's pick the vertex with the most neighbors in curr_set?
               # If it has many neighbors in curr_set, it's likely to form triangles.
               
               # For simplicity, pick LSB.
               v = (rem & -rem).bit_length() - 1
               rem_without_v = rem ^ (1 << v)
               
               # Exclude v
               rec(rem_without_v, curr_size, curr_set)
               
               # Include v
               valid = True
               cs = curr_set
               while cs:
                   u = (cs & -cs).bit_length() - 1
                   cs ^= (1 << u)
                   if adj[u] & (1 << v):
                       if adj[u] & adj[v] & curr_set:
                           valid = False
                           break
                           
               if valid:
                   new_curr = curr_set | (1 << v)
                   rec(rem_without_v, curr_size + 1, new_curr)
                   
           rec((1<<18)-1, 0, 0)
           return best
           
       current_score = calc_score(adj)
       best_score = current_score
       best_adj = [x for x in adj]
       
       while time.time() < deadline:
           # Pick random edge
           u = random.randint(0, 17)
           v = random.randint(0, 17)
           if u == v:
               continue
               
           if adj[u] & (1 << v):
               # Remove edge
               new_adj = [x for x in adj]
               new_adj[u] &= ~(1 << v)
               new_adj[v] &= ~(1 << u)
               # Always K4-free
               new_score = calc_score(new_adj)
               if new_score < best_score:
                   best_score = new_score
                   best_adj = new_adj
                   adj = new_adj
                   current_score = new_score
           else:
               # Add edge
               # Check K4
               common = adj[u] & adj[v]
               # Check if common has an edge
               k4_found = False
               c = common
               while c:
                   w = (c & -c).bit_length() - 1
                   c ^= (1 << w)
                   if adj[w] & common:
                       k4_found = True
                       break
                       
               if not k4_found:
                   new_adj = [x for x in adj]
                   new_adj[u] |= (1 << v)
                   new_adj[v] |= (1 << u)
                   new_score = calc_score(new_adj)
                   if new_score < best_score:
                       best_score = new_score
                       best_adj = new_adj
                       adj = new_adj
                       current_score = new_score
                   
       # Return best_adj edges
       edges = []
       for u in range(18):
           for v in range(u+1, 18):
               if best_adj[u] & (1 << v):
                   edges.append([u, v])
       return edges
