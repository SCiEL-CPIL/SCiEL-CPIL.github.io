import numpy as np
import time

N = 600

def construct():
       start_time = time.time()
       deadline = start_time + 110 # 10s margin

       # Initialize with uniform
       f = np.ones(N) / N
       
       # We want to minimize max(conv(f, f))
       # Using a simple gradient step on the log-sum-exp approximation of max
       # Or just reducing the peak directly.
       
       for _ in range(1000):
           if time.time() > deadline:
               break
           
           c = np.convolve(f, f)
           # c has length 2N-1
           # Find max
           max_val = np.max(c)
           max_idx = np.argmax(c)
           
           # Gradient of max(c) w.r.t f
           # c[k] = sum f[i] f[k-i]
           # dc/d f[j] = sum_{k: k-i=j} f[k-i] ?
           # Actually, if we treat max as a soft function, or just use subgradient.
           # Subgradient of max(c) is the indicator of the max indices.
           # So we want to reduce f in a way that decreases c[max_idx].
           # c[max_idx] = sum_i f[i] f[max_idx-i]
           # To decrease this, we can decrease f[i] where f[max_idx-i] is large.
           
           # Simple heuristic:
           # Reduce f in the region that contributes most to the peak.
           # The contribution of f[i] to c[max_idx] is f[max_idx-i].
           # So we should reduce f[i] proportional to f[max_idx-i].
           
           # Update rule:
           # f_new = f - alpha * (f * f_reflected_at_peak)
           # But we must maintain sum(f)=1 and f>=0.
           
           # Let's try a multiplicative update (like EM)
           # f_new[i] = f[i] * (1 - alpha * f[peak-i])
           
           # Define the reflected vector
           # Indices in f are 0..N-1.
           # peak is 0..2N-2.
           # f[peak-i] is valid if 0 <= peak-i < N.
           
           contrib = np.zeros(N)
           # c[peak] = sum_{i=0}^{N-1} f[i] * f[peak-i] (where f is 0 outside 0..N-1)
           # So contrib[i] = f[peak-i] if valid, else 0.
           
           for i in range(N):
               j = max_idx - i
               if 0 <= j < N:
                   contrib[i] = f[j]
                   
           # Normalize contrib
           if np.sum(contrib) > 0:
               contrib_norm = contrib / np.sum(contrib)
           else:
               contrib_norm = np.ones(N) / N
               
           # Update
           alpha = 0.1
           f = f * (1 - alpha * contrib_norm)
           # Renormalize
           f = f / np.sum(f)
           # Clip
           f = np.clip(f, 0, None)
           f = f / np.sum(f)
           
       return list(f)
