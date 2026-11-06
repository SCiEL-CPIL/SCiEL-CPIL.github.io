   import sys
   
   def construct():
       n = 128
       
       # Precompute all 4-term APs
       # A 4-term AP is defined by start a and difference d > 0
       # positions: a, a+d, a+2d, a+3d
       # All positions must be <= n
       aps = []
       for d in range(1, n):
           for a in range(1, n - 3 * d + 1):
               ap = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1) # 0-indexed
               aps.append(ap)
               
       # For each position, list of indices of APs it belongs to
       pos_aps = [[] for _ in range(n)]
       for i, ap in enumerate(aps):
           for pos in ap:
               pos_aps[pos].append(i)
               
       def is_valid_assignment(colors, pos, c, k):
           # Check all APs involving pos
           for ap_idx in pos_aps[pos]:
               ap = aps[ap_idx]
               # Get colors of the 4 positions
               # If any is -1, it's not fully assigned
               # We only check if fully assigned OR if 3 are assigned and cause immediate contradiction?
               # The immediate contradiction is: 3 assigned are same color.
               # If 3 are same, the 4th cannot fix it. So if we just assigned the 3rd one, we can fail now.
               
               assigned_colors = []
               unassigned_count = 0
               for p in ap:
                   if colors[p] == -1:
                       unassigned_count += 1
                   else:
                       assigned_colors.append(colors[p])
                       
               if unassigned_count == 0:
                   # Fully assigned
                   if len(set(assigned_colors)) < 3:
                       return False
               elif unassigned_count == 1:
                   # 3 assigned, 1 unassigned
                   # If the 3 assigned are all same, no color can fix this AP
                   if len(set(assigned_colors)) == 1:
                       return False
                   # If the 3 assigned use 2 colors, the unassigned one MUST be different from both.
                   # We don't check this here because we don't know the unassigned color yet.
                   # But we can check if the unassigned position is the one we are currently assigning?
                   # If unassigned_count == 1 and the unassigned position is NOT pos, then this AP is not fully determined by our current choice yet, 
                   # but it imposes a constraint on the future. We don't need to fail now, unless we are filling the last one?
                   # Wait, if unassigned_count == 1, and the unassigned position is `pos`, then `colors[pos]` is currently -1 in the array, 
                   # but we are trying to set it to `c`.
                   # So we should simulate the assignment for this check.
                   
                   # Let's refine: 
                   # We are checking if assigning `c` to `pos` is valid.
                   # So for this AP, the colors are:
                   # If p == pos, color is c.
                   # Else, color is colors[p].
                   
                   # Let's just construct the full set of colors for this AP assuming `pos` gets `c`.
                   # This is safer.
                   
                   pass # Logic handled below
                   
               # Actually, simpler: 
               # For each AP involving pos, construct the set of colors assuming pos=c.
               # If all 4 are assigned (i.e., other 3 were already assigned), check size >= 3.
               # If not all assigned, we can't check the full condition yet.
               # BUT, we CAN check the "3 same" condition.
               # If the other 3 are assigned and are all same, then regardless of c, the AP will have <= 2 colors?
               # If other 3 are A, A, A, and we pick c, set is {A, c}. Size 1 or 2. Always < 3.
               # So if the other 3 are assigned and same, it's INVALID.
               
               # So:
               other_colors = []
               for p in ap:
                   if p == pos:
                       continue
                   if colors[p] == -1:
                       # Not fully assigned, can't check full condition
                       # But can we check partial?
                       # If 2 others are assigned and same? No, that's not a contradiction yet.
                       # If 3 others are assigned? That means unassigned_count was 1 (which is pos).
                       # So if we see a -1 in `other_colors`, it means at least 2 are unassigned (pos and another).
                       # In that case, we can't check the "3 same" rule for the other 3 because they aren't all assigned.
                       # Wait, if unassigned_count > 1, we can't determine if the 3 assigned ones are same?
                       # We only care if 3 assigned ones are same.
                       # If we have 3 assigned and 1 unassigned (pos), we check the 3 assigned.
                       # If we have 2 assigned and 2 unassigned, we can't check "3 same" because we only have 2.
                       pass
                       
               # Let's rewrite the check cleanly inside the loop.
               # For each AP involving pos:
               #   Let S be the set of colors of the 3 OTHER positions.
               #   If any of the other 3 is -1, we cannot fully evaluate the AP yet.
               #   However, if all 3 others are assigned:
               #       Let C_other be the list of their colors.
               #       If len(set(C_other)) == 1: return False (3 same, impossible to fix)
               #       If len(set(C_other)) == 2: 
               #           # The 4th (pos) must be different from both.
               #           # We are assigning c. So c must not be in set(C_other).
               #           if c in set(C_other): return False
               #       If len(set(C_other)) == 3:
               #           # Always ok
               #           pass
               #   If fewer than 3 others are assigned, we can't check this constraint yet.
               
               # This logic covers all cases where the AP becomes "critical".
               
               others = []
               for p in ap:
                   if p != pos:
                       if colors[p] == -1:
                           others.append(-1)
                       else:
                           others.append(colors[p])
                           
               # If any other is -1, we can't check the full constraint for this AP yet.
               # But wait, if 2 are -1, we have only 1 other assigned. Can't check.
               # If 1 is -1, we have 2 others assigned. Can't check "3 same".
               # If 0 are -1, we have 3 others assigned. We CAN check.
               
               if -1 not in others:
                   set_others = set(others)
                   if len(set_others) == 1:
                       return False
                   if len(set_others) == 2:
                       if c in set_others:
                           return False
                   # len 3 is always ok
                       
           return True

       def solve(pos, k, colors):
           if pos == n:
               return True
               
           # Try colors 0 to k-1
           for c in range(k):
               if is_valid_assignment(colors, pos, c, k):
                   colors[pos] = c
                   if solve(pos + 1, k, colors):
                       return True
                   colors[pos] = -1
                   
           return False

       # Try to find minimum k
       for k in range(1, 129):
           colors = [-1] * n
           if solve(0, k, colors):
               # Re-label to 0..k-1? 
               # The task says "labels in 0..127". 
               # "Minimize the number of labels actually used".
               # The evaluator measures the raw value (number of labels used? or max label? 
               # "Minimize the number of labels actually used; labels need not be consecutive."
               # "The evaluator measures the raw value (lower is better)."
               # Usually "number of labels actually used" means the count of distinct integers in the output.
               # If I use colors 0, 1, 2, the count is 3.
               # If I use colors 0, 5, 10, the count is 3.
               # My solver uses 0..k-1, so count is k (if all used) or less.
               # I should return the list of colors.
               # To be safe, I can map the used colors to 0..m-1 where m is the count.
               # But the task says "labels in 0..127".
               # And "Minimize the number of labels actually used".
               # So the objective is the count of distinct labels.
               # My solver minimizes k. If it succeeds with k, the number of distinct labels is <= k.
               # It might not use all k colors.
               # I should compress the labels to ensure the count is minimized?
               # Actually, if I find a solution with k=3, the count is <= 3.
               # If I return it, the evaluator will count distinct values.
               # So I just return `colors`.
               
               return colors
               
       raise Exception("No solution found")
   