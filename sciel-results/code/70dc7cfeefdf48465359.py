import time

def construct(n_rows, k):
         # 1. Get initial bound from greedy
         greedy_rows = []
         used = set()
         for _ in range(n_rows):
             row = [0]
             x = 1
             while len(row) < k + 1:
                 new = {x - a for a in row}
                 if not new & used:
                     row.append(x)
                     used |= new
                     x += 1
                 else:
                     x += 1
             greedy_rows.append(row)
         
         best_max = max(max(r) for r in greedy_rows)
         best_solution = [list(r) for r in greedy_rows]
         
         # 2. Backtracking search
         def dfs(row_idx, current_row, current_rows, used_diffs, current_max):
             nonlocal best_max, best_solution
             
             if row_idx == n_rows:
                 if current_max < best_max:
                     best_max = current_max
                     best_solution = current_rows + [current_row]
                 return True
                 
             # Pruning
             remaining = k + 1 - len(current_row)
             min_final = current_row[-1] + remaining
             if min_final >= best_max:
                 return False
             if current_max >= best_max:
                 return False
                 
             # Try values
             start = current_row[-1] + 1
             # Limit search to best_max
             # We can also limit based on the fact that we need 'remaining' more slots
             # so x cannot be too large either? No, x can be large, but then final will be larger.
             # The bound min_final handles the lower bound.
             # Upper bound for x:
             # If we pick x, the final value will be at least x + (remaining - 1).
             # So x + remaining - 1 < best_max => x < best_max - remaining + 1
             end = best_max - remaining + 1
             
             # Randomize order?
             # If we try in increasing order, we find the first valid solution quickly.
             # To find better ones, we might need to try larger values for early positions.
             # However, for DTS, the optimal solution is usually "dense".
             # Let's try increasing order first. If we have time, we can add randomization.
             
             for x in range(start, end + 1):
                 new_diffs = {x - a for a in current_row}
                 if new_diffs & used_diffs:
                     continue
                 
                 new_used = used_diffs | new_diffs
                 new_row = current_row + [x]
                 new_max = max(current_max, x)
                 
                 if len(new_row) == k + 1:
                     new_rows = current_rows + [new_row]
                     if dfs(row_idx + 1, [0], new_rows, new_used, new_max):
                         return True
                 else:
                     if dfs(row_idx, new_row, current_rows, new_used, new_max):
                         return True
             return False
             
         # Run DFS with time limit
         start_time = time.time()
         # We need to allow the DFS to run for some time.
         # The current DFS returns True if it finds a better solution.
         # It will keep searching until it finds one or exhausts the branch.
         # To prevent infinite loops or excessive time, we can add a global time check.
         
         # Modified DFS with time check
         def dfs_timed(row_idx, current_row, current_rows, used_diffs, current_max):
             nonlocal best_max, best_solution
             
             if time.time() - start_time > 110: # Leave 10s margin
                 return False
                 
             if row_idx == n_rows:
                 if current_max < best_max:
                     best_max = current_max
                     best_solution = current_rows + [current_row]
                 return True
                 
             remaining = k + 1 - len(current_row)
             min_final = current_row[-1] + remaining
             if min_final >= best_max:
                 return False
             if current_max >= best_max:
                 return False
                 
             start = current_row[-1] + 1
             end = best_max - remaining + 1
             
             # To explore more, we can shuffle the range?
             # No, increasing order is good for finding *a* solution.
             # Since we already have a greedy solution, we are looking for improvements.
             # The greedy solution is likely the first one found by this DFS if we started from scratch.
             # Since we have a bound, we prune branches that can't beat it.
             
             for x in range(start, end + 1):
                 if time.time() - start_time > 110:
                     return False
                 
                 new_diffs = {x - a for a in current_row}
                 if new_diffs & used_diffs:
                     continue
                 
                 new_used = used_diffs | new_diffs
                 new_row = current_row + [x]
                 new_max = max(current_max, x)
                 
                 if len(new_row) == k + 1:
                     new_rows = current_rows + [new_row]
                     if dfs_timed(row_idx + 1, [0], new_rows, new_used, new_max):
                         return True
                 else:
                     if dfs_timed(row_idx, new_row, current_rows, new_used, new_max):
                         return True
             return False
             
         # Start search
         dfs_timed(0, [0], [], set(), 0)
         
         return best_solution
