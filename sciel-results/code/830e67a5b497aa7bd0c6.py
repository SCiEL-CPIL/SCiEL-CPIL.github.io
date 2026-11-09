import random
import time

def construct():
      start_time = time.time()
      deadline = 120  # Safety margin
      
      def get_diffs(S):
          return set(y - x for i, x in enumerate(S) for y in S[i+1:])
      
      def is_valid(S):
          diffs = get_diffs(S)
          return len(diffs) == len(S) * (len(S) - 1) // 2
      
      best_set = None
      best_max = float('inf')
      
      # Heuristic search
      for restart in range(1000):
          if time.time() - start_time > deadline:
              break
          
          S = [0]
          covered = set()
          
          # Try to build the set
          max_iter = 50
          for _ in range(max_iter):
              if covered.issuperset(set(range(1, 31))):
                  break
              
              # Try adding a new element
              # Candidate: existing element + d
              # We want to cover missing diffs
              missing = set(range(1, 31)) - covered
              
              # Pick a random missing diff
              d = random.choice(list(missing))
              
              # Try to realize d
              # Option 1: x + d is new
              # Option 2: y - d is new
              
              candidates = []
              for x in S:
                  candidates.append(x + d)
                  candidates.append(x - d)
              
              # Filter valid candidates
              valid_candidates = []
              for c in candidates:
                  if c < 0:
                      continue
                  if c in S:
                      continue
                  new_S = S + [c]
                  new_S.sort()
                  if is_valid(new_S):
                      valid_candidates.append(c)
              
              if not valid_candidates:
                  # Try larger jump?
                  # Or break
                  break
              
              # Pick best candidate (e.g., smallest max)
              c = min(valid_candidates)
              S.append(c)
              S.sort()
              covered = get_diffs(S)
          
          if covered.issuperset(set(range(1, 31))):
              max_elem = max(S)
              if max_elem < best_max:
                  best_max = max_elem
                  best_set = S[:]
      
      if best_set:
          return best_set
      else:
          # Fallback to seed-like greedy
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
