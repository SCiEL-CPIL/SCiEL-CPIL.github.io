import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
       # Normalize inputs to O(1)
       tn = (t - 40.0) / 10.0
       xn = x / 0.2
       vn = v / 0.35
       
       # Extract params
       w1 = params[0]
       ct1 = params[1]
       cx1 = params[2]
       cv1 = params[3]
       w2 = params[4]
       ct2 = params[5]
       cx2 = params[6]
       cv2 = params[7]
       sigma_raw = params[8]
       bias = params[9]
       
       # Ensure sigma is positive and reasonably sized
       sigma = np.log1p(np.exp(sigma_raw)) + 0.5
       
       # Compute squared distances
       dist1_sq = (tn - ct1)**2 + (xn - cx1)**2 + (vn - cv1)**2
       dist2_sq = (tn - ct2)**2 + (xn - cx2)**2 + (vn - cv2)**2
       
       # Gaussian kernels
       # Clip exponent to avoid underflow/overflow issues, though exp(-large) -> 0 is fine
       exp1 = np.exp(-dist1_sq / (2.0 * sigma * sigma))
       exp2 = np.exp(-dist2_sq / (2.0 * sigma * sigma))
       
       y = bias + w1 * exp1 + w2 * exp2
       
       # Ensure finite output
       return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
