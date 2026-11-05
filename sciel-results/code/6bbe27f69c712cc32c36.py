import itertools
import random
import time

def construct(n):
       if n == 1:
           return "1"
       
       # Generate permutations for n-1
       # We need to construct the word recursively.
       # But we can optimize the ordering at each step.
       
       # Let's build up from 1 to n.
       # word_k is the superpermutation for 1..k.
       # But we need the *sequence* of permutations used to build it to optimize the next step?
       # No, the next step only depends on the *set* of permutations of 1..k and their *order* in the final word?
       # Actually, the recursive construction is:
       # W_k is built from W_{k-1} by replacing each perm P in W_{k-1} (in order of appearance) with P k P.
       # So the order of the "blocks" in W_k is determined by the order of the permutations in W_{k-1}.
       
       # So, to optimize W_7, we need to choose a good ordering for the 720 perms of 1..6.
       # This ordering is then used to generate W_7.
       # But W_6 itself was generated from an ordering of 120 perms of 1..5.
       # If we change the ordering for W_6, W_6 changes, and the set of "starting points" for W_7 changes?
       # No, W_7 is built by taking the *permutations* that appear in W_6.
       # W_6 contains all 720 permutations of 1..6.
       # The order in which they appear in W_6 is determined by the construction of W_6.
       # If we want to optimize W_7, we should ideally be able to choose the order of the 720 perms independently.
       # But the recursive structure ties them.
       
       # Alternative:
       # Just construct W_{n-1} using the standard method (or optimized).
       # Extract the sequence of (n-1)-permutations from W_{n-1}.
       # Then, instead of using that fixed order for the n-th step, we can REORDER them?
       # If we reorder the blocks $\pi k \pi$, do we still get a valid superpermutation?
       # As established, YES, any ordering of the $(n-1)!$ permutations works.
       # So we can:
       # 1. Generate a valid superpermutation $W_{n-1}$.
       # 2. Extract the list of $(n-1)$-permutations in the order they appear in $W_{n-1}$.
       # 3. REORDER this list to maximize overlap for the n-th step.
       # 4. Construct $W_n$ from this reordered list.
       
       # This decouples the optimization of step n from step n-1.
       # We can apply this at each step.
       
       word = [1]
       for k in range(2, n + 1):
           # Extract (k-1)-permutations from current word
           perms = []
           seen = set()
           for i in range(len(word) - (k - 1) + 1):
               window = tuple(word[i:i + k - 1])
               if len(set(window)) == k - 1 and window not in seen:
                   seen.add(window)
                   perms.append(window)
           
           # Optimize the order of these perms to maximize overlap when inserting k
           # We have a list of tuples.
           # We want an ordering P_0, ..., P_{m-1} such that sum overlap(P_i, P_{i+1}) is maximized.
           # m = (k-1)!
           
           # For k=7, m=720.
           # Greedy approach:
           # 1. Start with a random permutation of the list.
           # 2. Calculate total overlap.
           # 3. Try local swaps to improve.
           
           # Given time constraints, a simple greedy "nearest neighbor" might be enough to beat the seed.
           # The seed uses the order from the previous step.
           # A random order might be better?
           # Or a greedy order.
           
           # Let's implement a greedy construction of the order:
           # Start with a random perm.
           # At each step, pick the unused perm with max overlap with current.
           
           # To be safe and fast, we'll do a few random restarts of the greedy method and pick the best.
           
           def calc_overlap(p1, p2):
               # p1, p2 are tuples of length k-1
               max_k = 0
               # Max possible overlap is k-2 (since they are distinct)
               for j in range(1, len(p1)):
                   if p1[-j:] == p2[:j]:
                       max_k = j
               return max_k
               
           def greedy_order(perms_list, seed=0):
               random.seed(seed)
               remaining = list(perms_list)
               random.shuffle(remaining)
               current = remaining.pop()
               order = [current]
               
               while remaining:
                   best_next = None
                   best_ov = -1
                   # To speed up, we might not check all remaining if m is large, but 720 is small.
                   for p in remaining:
                       ov = calc_overlap(current, p)
                       if ov > best_ov:
                           best_ov = ov
                           best_next = p
                   # If multiple have same best_ov, pick random?
                   # For now, pick the first one found with best_ov
                   order.append(best_next)
                   remaining.remove(best_next)
                   current = best_next
                   
               return order
               
           def total_overlap(order):
               tot = 0
               for i in range(len(order) - 1):
                   tot += calc_overlap(order[i], order[i+1])
               return tot
               
           # Find a good order
           best_order = None
           best_score = -1
           # Try a few random seeds for the greedy start
           # 720 items, greedy is O(m^2) ~ 500k ops. Very fast.
           # We can do 100 restarts easily.
           
           start_time = time.time()
           limit = 10 # seconds per step? No, total 150s.
           # For n=7, this is the heavy step.
           # For n<7, it's trivial.
           
           for seed in range(50):
               if time.time() - start_time > 10: break
               cand = greedy_order(perms, seed)
               score = total_overlap(cand)
               if score > best_score:
                   best_score = score
                   best_order = cand
                   
           # If no order found (shouldn't happen), use original
           if best_order is None:
               best_order = perms
               
           # Construct new word
           new_word = []
           for p in best_order:
               block = list(p) + [k] + list(p)
               # Overlap with new_word
               # Find max overlap between suffix of new_word and prefix of block
               max_ov = 0
               # Block length is 2*(k-1)+1
               # Max overlap is min(len(new_word), len(block))
               # But we know the structure.
               # The prefix of block is p.
               # The suffix of new_word ends with the previous p_prev.
               # So overlap is between p_prev and p.
               # We already calculated this in the optimization?
               # Yes, but we need to append block[overlap:]
               
               # To be safe, calculate overlap directly
               # Optimization: we know the overlap is the same as calc_overlap(prev_p, current_p)
               # But new_word might have changed?
               # No, new_word ends with the previous block's suffix, which is the previous p.
               # So overlap is exactly calc_overlap(prev_p, current_p) if we track prev_p.
               
               if not new_word:
                   new_word = block
               else:
                   # Get previous p from the order
                   prev_p = best_order[best_order.index(p)-1] # This is O(N) lookup, bad.
                   # Better: iterate with index
                   pass
                   
           # Let's rewrite the construction loop to be efficient
           new_word = []
           for i, p in enumerate(best_order):
               block = list(p) + [k] + list(p)
               if i == 0:
                   new_word = block
               else:
                   prev_p = best_order[i-1]
                   # Calculate overlap
                   ov = 0
                   # Max overlap is k-2
                   for j in range(1, k-1):
                       if prev_p[-j:] == p[:j]:
                           ov = j
                   # The suffix of new_word is prev_p.
                   # The prefix of block is p.
                   # Overlap is ov.
                   # Append block[ov:]
                   new_word.extend(block[ov:])
                   
           word = new_word
           
       return "".join(map(str, word))
