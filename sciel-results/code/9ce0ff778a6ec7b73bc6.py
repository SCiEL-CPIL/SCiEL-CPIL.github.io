   p.pop(i) # Remove job
   for pos in range(len(p) + 1):
       p.insert(pos, job)
       local_ms = _makespan(p, processing_times)
       if local_ms < best_local_ms:
           best_local_ms = local_ms
           best_pos = pos
       # Move job to next position for next iteration
       if pos < len(p): # len(p) includes the job, so max index is len(p)-1
           # Move from pos to pos+1
           p.pop(pos)
           p.insert(pos+1, job)
   