def dispatch(ready_operations, time):
       if not ready_operations:
           return 0
       
       best_idx = 0
       best_score = -1
       
       for i, op in enumerate(ready_operations):
           remaining_work = op["remaining_work"]
           options = op["machine_options"]
           if not options:
               continue
           
           min_dur = min(options.values())
           if min_dur < 1e-9:
               min_dur = 1e-9
           
           # Primary: Urgency (Work / Time)
           # Secondary: Absolute Work (to break ties favoring larger jobs)
           score = remaining_work / min_dur
           
           # Add a tiny epsilon based on index to ensure deterministic behavior if perfect ties?
           # Or just use the raw score.
           
           if score > best_score:
               best_score = score
               best_idx = i
               
       return int(best_idx)
