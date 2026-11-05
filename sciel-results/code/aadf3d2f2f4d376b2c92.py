   while active_indices:
       t_idx_pos = random.randint(0, len(active_indices) - 1)
       t_idx = active_indices[t_idx_pos]
       
       if not active_triples[t_idx]:
           # Remove inactive triple from list
           active_indices[t_idx_pos] = active_indices[-1]
           active_indices.pop()
           continue
       
       # Process t_idx
       ...
   