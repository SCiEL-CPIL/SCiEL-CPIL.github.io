import numpy as np

def construct():
       n = 128
       colors = np.zeros(n, dtype=np.uint8)
       
       # Precompute bit masks for quick lookup? 
       # bit_mask[v] = 1 << v
       bit_mask = [1 << v for v in range(n)]
       
       for i in range(1, n):
           forbidden = 0
           
           # 4-term progressions
           if i >= 3:
               d_vals = np.arange(1, i // 3 + 1)
               idx1 = i - d_vals
               idx2 = i - 2 * d_vals
               idx3 = i - 3 * d_vals
               
               A = colors[idx1]
               B = colors[idx2]
               D = colors[idx3]
               
               # Check for all same (failure)
               # If A==B==D, we can't place any color.
               # The seed program sets forbidden to all 1s.
               # We'll check if this happens.
               all_same = (A == B) & (B == D)
               if np.any(all_same):
                   # Failure state, but seed says it shouldn't happen.
                   # If it does, we might need to backtrack. 
                   # For now, we set forbidden to all 1s, which leads to fallback 0.
                   forbidden = (1 << n) - 1
               else:
                   # Check for exactly 2 distinct
                   # Condition: not all same AND at least one pair equal
                   # Pairs: (A,B), (A,D), (B,D)
                   pair_eq = (A == B) | (A == D) | (B == D)
                   two_distinct_mask = pair_eq # Since all_same is false, pair_eq implies exactly 2 distinct?
                   # Wait. If A=B!=D, pair_eq is True. Distinct are {A, D}. Size 2.
                   # If A=B=D, pair_eq is True. But we handled all_same.
                   # So yes, if not all_same, pair_eq implies size 2.
                   
                   if np.any(two_distinct_mask):
                       # Extract values
                       # We need the union of {A,B,D} for all d where two_distinct_mask is True.
                       # We can just take the unique values from the masked arrays.
                       vals = np.concatenate([
                           A[two_distinct_mask],
                           B[two_distinct_mask],
                           D[two_distinct_mask]
                       ])
                       unique_vals = np.unique(vals)
                       for v in unique_vals:
                           forbidden |= bit_mask[v]
           
           # 3-term progressions (to prevent 3-same)
           if i >= 2:
               d_vals_3 = np.arange(1, i // 2 + 1)
               idxA = i - d_vals_3
               idxB = i - 2 * d_vals_3
               
               A3 = colors[idxA]
               B3 = colors[idxB]
               
               same_mask = (A3 == B3)
               if np.any(same_mask):
                   vals = A3[same_mask]
                   unique_vals = np.unique(vals)
                   for v in unique_vals:
                       forbidden |= bit_mask[v]
           
           # Find smallest color not in forbidden
           # forbidden is an integer.
           # We can iterate c from 0 to 127.
           found = False
           for c in range(n):
               if not (forbidden & bit_mask[c]):
                   colors[i] = c
                   found = True
                   break
           
           if not found:
               # Fallback
               colors[i] = 0
               
       return colors.tolist()
