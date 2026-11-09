import time

def construct():
      start_time = time.time()
      deadline = start_time + 115 # 5 sec margin
      
      best_max = 10**6
      best_set = [0]
      
      target_diffs = set(range(1, 31))
      
      def dfs(A, diffs, covered):
          nonlocal best_max, best_set
          
          if time.time() > deadline:
              return
              
          if covered == target_diffs:
              val = max(A)
              if val < best_max:
                  best_max = val
                  best_set = A[:]
              # We can return here if we assume smaller x leads to smaller max?
              # Not necessarily, but likely. To be safe, we can continue a bit
              # or just return if we want to find *a* good one fast.
              # Given the score is 1/value, finding the absolute minimum is best.
              # But DFS with small x first finds small max first.
              # Let's return to allow exploration of other branches?
              # No, if we found a solution with max M, any branch with current max >= M is useless.
              # But we are in a specific branch.
              # Let's just update and return to try other paths?
              # Actually, if we sort candidates by x, the first solution found is likely optimal or near-optimal.
              # Let's just return to save time for other branches?
              # If we return, we might miss a better solution in a different branch.
              # But different branches diverge at earlier steps.
              # Let's continue search if time permits, but prune if max(A) >= best_max.
              pass
          
          if len(A) >= 32:
              return
              
          current_max = max(A)
          
          # Generate candidates
          # We can limit x to best_max to prune
          # Start from current_max + 1
          # To avoid infinite loop if no solution found, we need a cap on x?
          # If we don't find a solution, we might search x=1000, 10000...
          # But if covered != target, we need to add elements.
          # If we can't cover target with reasonable x, we might fail.
          # However, the problem guarantees a solution exists (implied by "construct").
          
          candidates = []
          # We try x up to best_max (or a limit like 200 if best_max is inf)
          limit = best_max if best_max < 10**6 else 200 
          
          for x in range(current_max + 1, limit + 1):
              # Check validity
              valid = True
              new_diffs = []
              for a in A:
                  d = x - a
                  if d in diffs:
                      valid = False
                      break
                  new_diffs.append(d)
              
              if not valid:
                  continue
                  
              # Calculate gain
              new_covered = covered.union(d for d in new_diffs if d in target_diffs)
              gain = len(new_covered) - len(covered)
              
              candidates.append((x, gain, new_diffs, new_covered))
              
          # Sort by gain desc, then x asc
          candidates.sort(key=lambda item: (-item[1], item[0]))
          
          for x, gain, new_diffs, new_covered in candidates:
              # Prune if current max already >= best_max?
              # x is the new max. If x >= best_max, no improvement.
              if x >= best_max:
                  continue
                  
              A.append(x)
              diffs.update(new_diffs)
              # Note: covered is passed by value (set), so we don't need to revert if we pass new_covered
              # But diffs is a set, we need to revert.
              # Better to pass copies or revert.
              # Since diffs is a set, copying it at every step is expensive.
              # Let's revert manually.
              
              added_diffs = set(new_diffs)
              dfs(A, diffs, new_covered)
              
              # Revert
              for d in added_diffs:
                  diffs.discard(d)
              A.pop()
              
              if time.time() > deadline:
                  return

      dfs([0], set(), set())
      
      # If best_set is still just [0] or invalid, return a fallback
      # But the problem implies a solution exists.
      # Fallback: Greedy from seed program
      if best_max == 10**6:
          # Run the seed logic as fallback
          a = [0]
          def diffs(s):
              return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
          for d in range(1, 31):
              if d in diffs(sorted(a)):
                  continue
              for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                  trial = sorted(a + [x])
                  if len(set(diffs(trial))) == len(diffs(trial)):
                      a = trial
                      break
              else:
                  y = 2 * max(a) + 3 * d + 1
                  while True:
                      trial = sorted(a + [y, y + d])
                      if len(set(diffs(trial))) == len(diffs(trial)):
                          a = trial
                          break
                      y += 1
          return a
          
      return best_set
