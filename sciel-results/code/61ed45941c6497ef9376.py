       # Local Search
       current = set(best_set)
       all_nums = list(range(2, NMAX + 1))
       
       def is_valid(s):
           lst = list(s)
           n = len(lst)
           for i in range(n):
               a = lst[i]
               for j in range(n):
                   if i == j: continue
                   b = lst[j]
                   for k in range(n):
                       if i == k or j == k: continue
                       c = lst[k]
                       if (b * c) % a == 0:
                           return False
           return True
       
       improved = True
       while improved and time.time() < deadline:
           improved = False
           # Try adding
           for x in all_nums:
               if x in current: continue
               new_set = set(current)
               new_set.add(x)
               if is_valid(new_set):
                   current = new_set
                   improved = True
                   break # Restart search with new set
           
           # Try swapping (if no addition worked)
           if not improved:
               for x in all_nums:
                   if x in current: continue
                   for y in list(current):
                       new_set = set(current)
                       new_set.remove(y)
                       new_set.add(x)
                       if is_valid(new_set) and len(new_set) > len(current): # We don't want to decrease size
                           # Actually, swapping keeps size same. We want to increase size.
                           # So we only swap if it allows a future addition?
                           # For now, just stick to addition.
                           pass
       
       if len(current) > len(best_set):
           best_set = current
   