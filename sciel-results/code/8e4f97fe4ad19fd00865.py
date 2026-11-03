import numpy as np

def construct():
          N = 50
          DELTA = 0.5 / N
          
          best_score = 0
          best_h = np.ones(N) / N
          
          for k in range(1, N+1):
              # Try a few random sets
              for _ in range(100):
                  indices = np.random.choice(N, k, replace=False)
                  h = np.zeros(N)
                  h[indices] = 1.0 / k
                  l1 = np.sum(h)
                  l2_sq = np.sum(h**2)
                  h_conv = np.convolve(h, h, mode='full')
                  max_conv = np.max(h_conv)
                  score = (l2_sq / (l1 * DELTA * max_conv))
                  if score > best_score:
                      best_score = score
                      best_h = h.copy()
                      
          return list(best_h)
