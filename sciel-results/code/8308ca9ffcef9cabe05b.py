import numpy as np

N = 400

def construct():
         # Fixed seed for reproducibility
         rng = np.random.default_rng(42)
         
         # Initial random binary sequence
         s = rng.choice([-1, 1], size=N)
         
         def get_ratio(seq):
             # Autocorrelation
             # np.correlate is O(N^2) but optimized C
             A = np.correlate(seq, seq, mode='full')
             peak = A[N-1]
             if peak == 0:
                 return float('inf')
             # Side lobes
             left = A[:N-1]
             right = A[N:]
             max_side = max(np.max(np.abs(left)), np.max(np.abs(right)))
             return max_side / peak

         current_ratio = get_ratio(s)
         
         # Local search
         improved = True
         while improved:
             improved = False
             for i in range(N):
                 # Flip bit
                 s[i] *= -1
                 new_ratio = get_ratio(s)
                 if new_ratio < current_ratio:
                     current_ratio = new_ratio
                     improved = True
                 else:
                     # Revert
                     s[i] *= -1
                     
         return s
