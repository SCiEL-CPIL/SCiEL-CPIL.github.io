import time
import random

def construct():
       N = 20
       
       def popcount(x):
           return bin(x).count('1')

       def is_valid(neigh):
           for i in range(N):
               for j in range(i+1, N):
                   for k in range(j+1, N):
                       if popcount(neigh[i] & neigh[j] & neigh[k]) > 2:
                           return False
           return True

       def edge_count(neigh):
           return sum(popcount(neigh[i]) for i in range(N)) // 2

       def greedy_fill(neigh):
           improved = True
           # Randomize order to avoid bias?
           while improved:
               improved = False
               # Shuffle pairs for randomization in greedy
               pairs = [(u, v) for u in range(N) for v in range(u+1, N) if not (neigh[u] >> v) & 1]
               random.shuffle(pairs)
               for u, v in pairs:
                   if (neigh[u] >> v) & 1:
                       continue
                   neigh[u] |= (1 << v)
                   neigh[v] |= (1 << u)
                   if is_valid(neigh):
                       improved = True
                   else:
                       neigh[u] &= ~(1 << v)
                       neigh[v] &= ~(1 << u)
           return neigh

       start_time = time.time()
       deadline = start_time + 10.0
       
       best_count = 0
       best_neigh = [0]*N

       # 1. Try Seed
       neigh = [0]*N
       for a in range(2):
           for b in range(2, N):
               neigh[a] |= (1 << b)
               neigh[b] |= (1 << a)
       
       greedy_fill(neigh)
       c = edge_count(neigh)
       if c > best_count:
           best_count = c
           best_neigh = neigh[:]

       # 2. Random Restarts
       while time.time() < deadline - 1.0:
           neigh = [0]*N
           p = random.uniform(0.1, 0.5)
           for u in range(N):
               for v in range(u+1, N):
                   if random.random() < p:
                       neigh[u] |= (1 << v)
                       neigh[v] |= (1 << u)
           
           # Ensure valid start? Greedy fill will remove invalid edges? 
           # No, greedy_fill only ADDS edges. It doesn't remove.
           # So if start is invalid, result is invalid.
           # We need to start with a valid graph.
           # We can just start with empty and add randomly?
           # Or check validity and if invalid, remove random edges until valid?
           # Simpler: Start with empty, add random edges one by one if valid.
           
           neigh = [0]*N
           # Random Greedy Construction
           pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
           random.shuffle(pairs)
           for u, v in pairs:
               neigh[u] |= (1 << v)
               neigh[v] |= (1 << u)
               if not is_valid(neigh):
                   neigh[u] &= ~(1 << v)
                   neigh[v] &= ~(1 << u)
           
           # Now greedy fill to maximize
           greedy_fill(neigh)
           c = edge_count(neigh)
           if c > best_count:
               best_count = c
               best_neigh = neigh[:]
               
       # 3. Local Search on Best
       # If time remains, do local search on best_neigh
       if time.time() < deadline - 0.5:
           current_neigh = best_neigh[:]
           current_count = best_count
           no_improve_count = 0
           while time.time() < deadline - 0.1 and no_improve_count < 50:
               # Remove random edge
               edges = [(u, v) for u in range(N) for v in range(u+1, N) if (current_neigh[u] >> v) & 1]
               if not edges:
                   break
               u, v = random.choice(edges)
               old_neigh = current_neigh[:]
               
               # Remove
               current_neigh[u] &= ~(1 << v)
               current_neigh[v] &= ~(1 << u)
               
               # Greedy Fill
               greedy_fill(current_neigh)
               new_count = edge_count(current_neigh)
               
               if new_count > current_count:
                   current_count = new_count
                   no_improve_count = 0
                   if current_count > best_count:
                       best_count = current_count
                       best_neigh = current_neigh[:]
               else:
                   current_neigh = old_neigh
                   no_improve_count += 1

       # Return edges
       edges = []
       for u in range(N):
           for v in range(u+1, N):
               if (best_neigh[u] >> v) & 1:
                   edges.append([u, v])
       return edges
