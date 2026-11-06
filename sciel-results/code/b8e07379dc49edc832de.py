def construct():
       max_val = 256
       best_seq = []
       
       def dfs(seq, sums_mask, last):
           nonlocal best_seq
           if len(seq) > len(best_seq):
               best_seq = list(seq)
           
           # Pruning
           # Max possible additional elements
           if last > 0:
               max_add = 0
               t = last
               while t * 2 <= max_val:
                   t *= 2
                   max_add += 1
           else:
               max_add = 8
               
           if len(seq) + max_add <= len(best_seq):
               return
               
           start = last + 1 if last > 0 else 1
           if start > max_val:
               return
               
           for x in range(start, max_val + 1):
               # Compute new sums mask
               # Sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
               new_mask = 0
               total = x
               # Check if any new sum is already in sums_mask
               # We need to check bits.
               # If (sums_mask >> total) & 1: collision
               
               # Iterate backwards
               # We need to check all new sums
               # new_sums = [x, x+a_k, x+a_k+a_{k-1}, ...]
               
               # Efficient check:
               # Generate new sums and check against mask
               # If any collision, skip
               
               # To avoid creating a list every time, we can check on the fly
               # But we need the new mask if valid.
               
               # Let's try to be smart:
               # If x is small, sums are small, likely to collide.
               # If x is large, sums are large, likely to be new.
               
               # We can try candidates in reverse order (largest first)?
               # No, largest first limits future steps.
               
               # Let's stick to increasing order.
               
               valid = True
               new_mask = sums_mask
               total = x
               
               # Check x
               if (new_mask >> x) & 1:
                   valid = False
               else:
                   new_mask |= (1 << x)
                   
               if valid:
                   for a in reversed(seq):
                       total += a
                       if (new_mask >> total) & 1:
                           valid = False
                           break
                       else:
                           new_mask |= (1 << total)
                           
               if not valid:
                   continue
                   
               dfs(seq + (x,), new_mask, x)
               
       dfs((), 0, 0)
       return best_seq
