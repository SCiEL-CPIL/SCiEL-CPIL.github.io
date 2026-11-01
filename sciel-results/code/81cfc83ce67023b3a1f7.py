import time
import random
import itertools

def construct(q, n, radius):
           start_time = time.time()
           deadline = start_time + 110.0 # 10s margin
           
           total_words = q ** n
           
           # 1. Precompute neighbors
           # Represent words as tuples or ints? Ints are better for hashing/sets.
           # Map tuple to int: sum(d * q**i)
           
           # Generate all words
           all_words = list(itertools.product(range(q), repeat=n))
           word_to_idx = {w: i for i, w in enumerate(all_words)}
           idx_to_word = list(range(total_words))
           
           # Precompute neighbors for each index
           neighbors = [[] for _ in range(total_words)]
           for i, w in enumerate(all_words):
               # Self
               neighbors[i].append(i)
               # Single position changes
               w_list = list(w)
               for pos in range(n):
                   orig_val = w_list[pos]
                   for new_val in range(q):
                       if new_val == orig_val: continue
                       w_list[pos] = new_val
                       new_w = tuple(w_list)
                       neighbors[i].append(word_to_idx[new_w])
                   w_list[pos] = orig_val
           
           # 2. Greedy Construction
           uncovered = set(range(total_words))
           code = []
           
           while uncovered:
               if time.time() > deadline:
                   break
               
               # Pick a random uncovered word
               u = random.choice(list(uncovered))
               
               best_c = None
               max_covered = 0
               
               # Candidates are neighbors of u
               for c in neighbors[u]:
                   # Count how many uncovered words c covers
                   # Optimization: we can just iterate neighbors[c] and check membership in uncovered
                   count = 0
                   for v in neighbors[c]:
                       if v in uncovered:
                           count += 1
                       # Early break if count exceeds max_covered? No, we need exact max.
                       
                   if count > max_covered:
                       max_covered = count
                       best_c = c
                       
               if best_c is None:
                   # Fallback: pick any
                   best_c = u
                   
               code.append(best_c)
               # Remove covered
               for v in neighbors[best_c]:
                   if v in uncovered:
                       uncovered.remove(v)
               
           # 3. Redundancy Removal
           # We want to remove any codeword that is not essential.
           # Essential if its removal leaves some words uncovered.
           
           improved = True
           while improved and time.time() < deadline:
               improved = False
               for i in range(len(code)):
                   c = code[i]
                   # Check coverage without c
                   covered = set()
                   for j in range(len(code)):
                       if j == i: continue
                       covered.update(neighbors[code[j]])
                   
                   if len(covered) == total_words:
                       # c is redundant
                       code.pop(i)
                       improved = True
                       break # Restart loop
           
           # 4. Local Search (Optional, if time remains)
           # Try to swap codewords to enable further removals?
           # Or just try to find a better greedy solution with different random seeds?
           
           # Since we have time, let's try a few more random greedy restarts if the first one is poor?
           # No, the first greedy is usually good.
           # Let's try to improve the current code by trying to replace a codeword with another
           # that covers more "private" words of others?
           
           # A simple improvement:
           # If we can't remove a codeword, can we replace it with a different one
           # such that the new code allows removal of ANOTHER codeword?
           # This is complex.
           
           # Let's just return the current code.
           # Convert indices back to lists of digits
           result = []
           for idx in code:
               w = all_words[idx]
               result.append(list(w))
               
           return result
