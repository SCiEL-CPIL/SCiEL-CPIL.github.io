import numpy as np

def construct():
          n, m, p = 2, 3, 8
          rank = n * m * p
          
          # Create identity-like structures for the indices
          # U maps (i, j) to column c
          # V maps (j, k) to column c
          # W maps (k, i) to column c
          
          # We can generate the indices for each term
          i_idx, j_idx, k_idx = np.indices((n, m, p))
          
          # Flatten them
          i_flat = i_idx.flatten()
          j_flat = j_idx.flatten()
          k_flat = k_idx.flatten()
          
          # Row indices for U, V, W
          u_rows = i_flat * m + j_flat
          v_rows = j_flat * p + k_flat
          w_rows = k_flat * n + i_flat
          
          # Column indices are 0 to rank-1
          cols = np.arange(rank)
          
          # Initialize sparse-like or dense arrays
          U = np.zeros((n * m, rank), dtype=float)
          V = np.zeros((m * p, rank), dtype=float)
          W = np.zeros((p * n, rank), dtype=float)
          
          # Assign 1s
          U[u_rows, cols] = 1
          V[v_rows, cols] = 1
          W[w_rows, cols] = 1
          
          # Convert to lists to match interface if required, or return numpy arrays?
          # The interface says: "factor matrices of shapes ... with integer/half-integer entries"
          # The example returns lists of lists.
          # "Keep the function names and signatures ... EXACTLY as in the current program"
          # Current program returns {"U": U, "V": V, "W": W} where U, V, W are lists of lists.
          
          return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
