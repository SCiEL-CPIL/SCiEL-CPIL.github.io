import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # The task: minimize max|autoconvolution| / (integral)^2
    # Interpretation: max over k!=0, denominator (sum f)^2
    # Seed: ones(N) -> value 1/N
       
       # If we assume k=0 is included, ones(N) is global optimum.
       # If we assume k=0 is excluded, ones(N) is likely local/global optimum.
       
       # Let's try a quick local search to see if we can beat ones(N)
       # under the k!=0 assumption.
       
       f = np.ones(N)
       S = np.sum(f)
       
       def calc_score(f):
           S = np.sum(f)
           if S == 0:
               return float('inf')
           # Compute autocorrelation
           # C(k) = sum f[i] f[i+k]
           # Use FFT for speed
           F = np.fft.fft(f, 2*N)
           C = np.fft.ifft(F * np.conj(F)).real[:N]
           # Exclude k=0
           max_corr = np.max(np.abs(C[1:]))
           return max_corr / (S * S)
       
       current_score = calc_score(f)
       
       # Local search: flip random signs
       # Budget: 5 seconds
       start = time.time()
       best_f = f.copy()
       best_score = current_score
       
       while time.time() - start < 5.0:
           # Flip a random index
           idx = np.random.randint(N)
           f[idx] *= -1
           new_score = calc_score(f)
           if new_score < best_score:
               best_score = new_score
               best_f = f.copy()
           else:
               f[idx] *= -1 # revert
               
       return best_f
   # EVOLVE-BLOCK-END
