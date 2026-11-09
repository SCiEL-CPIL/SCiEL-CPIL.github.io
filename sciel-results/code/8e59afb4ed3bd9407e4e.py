import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 50 # 10s margin
       
       best_set = None
       best_size = 0
       
       # Helper to check validity using bitset
       def can_add(current_reachable, x):
           # current_reachable is an int bitset
           # Check if adding x creates sum 200
           # new_reachable = current_reachable | (current_reachable << x)
           # We only care if bit 200 is set in new_reachable
           # Bit 200 is set in new_reachable if:
           # 1. It was already set in current_reachable (but we maintain validity, so it's 0)
           # 2. It is set in (current_reachable << x), which means bit (200-x) was set in current_reachable
           if x > 200:
               return True
           # Check if (200-x) is reachable
           if x <= 200:
               if (current_reachable >> (200 - x)) & 1:
                   return False
           return True
           
       def update_reachable(current_reachable, x):
           # Return new reachable bitset
           # Mask to 200 bits to keep integers small?
           # Actually, we only care up to 200.
           # Shifts beyond 200 are irrelevant for the target 200.
           # But they might be relevant for future additions?
           # No, if sum > 200, it can't be part of a sum = 200 (since all numbers are positive).
           # So we can mask to 200 bits.
           mask = (1 << 201) - 1
           return (current_reachable | (current_reachable << x)) & mask

       # Initial best: {66..100}
       init_set = list(range(66, 101))
       # Verify init_set
       reachable = 1 # bit 0 set
       valid = True
       for x in init_set:
           if not can_add(reachable, x):
               valid = False
               break
           reachable = update_reachable(reachable, x)
       if valid:
           best_set = init_set
           best_size = len(init_set)
       else:
           # Fallback to empty or simple
           best_set = [100]
           best_size = 1

       numbers = list(range(1, 101))
       
       # Randomized Greedy with Restarts
       trial = 0
       while time.time() < deadline:
           trial += 1
           random.shuffle(numbers)
           selected = []
           reachable = 1
           for x in numbers:
               if can_add(reachable, x):
                   selected.append(x)
                   reachable = update_reachable(reachable, x)
           
           if len(selected) > best_size:
               best_size = len(selected)
               best_set = selected[:]
               
               # If we found a very large set, we might want to try local search to improve it
               # But first, just keep restarting to see if we can beat it.
               # If best_size reaches 36 or more, we can stop or continue for a bit.
               
           # Optional: If we are stuck at a size, try local search on best_set
           # But randomized greedy is often enough for this size.
           
       return best_set
