import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
          if len(remaining_weights) == 0 or remaining_capacity <= 0:
              return -1

          # Convert to numpy arrays for speed
          weights = np.asarray(remaining_weights)
          values = np.asarray(remaining_values)
          
          # Filter feasible items
          mask = weights <= remaining_capacity
          if not np.any(mask):
              return -1
          
          feasible_indices = np.where(mask)[0]
          feasible_weights = weights[mask]
          feasible_values = values[mask]
          
          N = len(feasible_weights)
          
          # Scaling factor
          S = 100
          C_int = int(round(remaining_capacity * S))
          
          # If capacity is 0 after scaling, no items can fit (unless weight 0)
          if C_int < 0:
              return -1

          # Scale weights
          # Handle potential floating point issues
          feasible_weights_int = np.array([int(round(w * S)) for w in feasible_weights], dtype=int)
          
          # Check for zero weight items with positive value
          # If any, we should pick the one with highest value immediately?
          # Actually DP handles it. But if C_int is 0, only weight 0 items can be picked.
          
          # DP Table
          # dp[i][w] max value using first i items with capacity w
          # Size (N+1) x (C_int+1)
          # To save memory/time, we can use 2 arrays if we don't need backtracking, 
          # but we DO need backtracking.
          
          # If N is large and C_int is large, this might be slow.
          # But N=100, C_int=1250 is small.
          
          dp = np.zeros((N + 1, C_int + 1), dtype=float)
          
          for i in range(1, N + 1):
              wi = feasible_weights_int[i-1]
              vi = feasible_values[i-1]
              
              # Copy previous row
              dp[i, :] = dp[i-1, :]
              
              if wi <= C_int:
                  # Update
                  # dp[i, w] = max(dp[i-1, w], dp[i-1, w-wi] + vi)
                  # Vectorized:
                  # For w from wi to C_int:
                  # dp[i, w] = max(dp[i, w], dp[i-1, w-wi] + vi)
                  # dp[i, wi:] = np.maximum(dp[i, wi:], dp[i-1, :C_int-wi+1] + vi)
                  
                  # Note: dp[i, wi:] currently holds dp[i-1, wi:]
                  # We compare it with dp[i-1, 0:C_int-wi+1] + vi
                  
                  # Slice lengths:
                  # LHS: C_int - wi + 1
                  # RHS: C_int - wi + 1
                  
                  dp[i, wi:] = np.maximum(dp[i, wi:], dp[i-1, :C_int-wi+1] + vi)
          
          # Backtrack
          i = N
          w = C_int
          while i > 0:
              if dp[i, w] == dp[i-1, w]:
                  i -= 1
              else:
                  # Item i-1 is in the solution
                  return int(feasible_indices[i-1])
          
          # If no item found, fallback to greedy
          # This happens if optimal value is 0
          if np.max(feasible_values) <= 0:
              return -1
          
          # Fallback: Greedy density
          densities = feasible_values / feasible_weights
          best_idx = np.argmax(densities)
          return int(feasible_indices[best_idx])
