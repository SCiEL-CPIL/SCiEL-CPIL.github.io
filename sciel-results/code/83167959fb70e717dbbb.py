def construct():
             # ... deadline guard ...
             
             def check_sum_200(S):
                 # Returns True if S has subset summing to 200
                 # Use DP or set of reachable sums
                 reachable = {0}
                 for x in S:
                     new = set()
                     for s in reachable:
                         val = s + x
                         if val <= 200:
                             new.add(val)
                     reachable.update(new)
                 return 200 in reachable

             def find_conflict_subset(S, target):
                 # Returns a subset of S summing to target, or None
                 # Since target <= 199 and S elements >= 1, size is small
                 # We can use DFS or DP to reconstruct
                 # For speed, we can just check if target is reachable, 
                 # but we need the subset to remove an element.
                 # We can store parents in DP.
                 
                 # Given time limits, a simple DFS might be okay for small targets?
                 # Target is 200-x. x is small, so target is large (e.g. 135).
                 # S size is ~35.
                 # DFS branching factor is high.
                 # Better: DP with backtracking.
                 
                 dp = {0: []}
                 for x in S:
                     new_dp = {}
                     for s, subset in dp.items():
                         ns = s + x
                         if ns <= target:
                             if ns not in new_dp:
                                 new_dp[ns] = subset + [x]
                     # merge
                     for ns, subset in new_dp.items():
                         if ns not in dp:
                             dp[ns] = subset
                         # else keep existing (shorter?)
                 if target in dp:
                     return dp[target]
                 return None

             def add_and_repair(initial_S, order, removal_strategy):
                 S = set(initial_S)
                 for x in order:
                     if x in S: continue
                     target = 200 - x
                     if target <= 0: continue
                     C = find_conflict_subset(list(S), target)
                     if C is None:
                         S.add(x)
                     else:
                         # Remove one element from C
                         if removal_strategy == 'max':
                             to_remove = max(C)
                         elif removal_strategy == 'min':
                             to_remove = min(C)
                         else:
                             to_remove = C[0]
                         S.discard(to_remove)
                         S.add(x)
                 return S
             
             # Strategies
             base = list(range(66, 101))
             orders = [
                 list(range(65, 0, -1)),
                 list(range(1, 66)),
                 # Randomized
             ]
             strategies = ['max', 'min', 'first']
             
             best_set = set(base)
             
             for strat in strategies:
                 for order in orders:
                     S = add_and_repair(base, order, strat)
                     if len(S) > len(best_set):
                         best_set = S
                         
             # Also run original greedy with random shuffles
             # ...
             
             return list(best_set)
