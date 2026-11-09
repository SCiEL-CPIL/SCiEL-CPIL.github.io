import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Assume a 4-point support structure [0, a, b, c] with 0 < a < b < c.
       This is based on the intuition that 0 is a very strong anchor for the LHS sum.
       The 4 points allow for a rich set of inequality combinations (e.g., 0+0+0 < 2a, 
       0+0+a < 2b, 0+a+a < 2c, etc.).
    2. Parameterize the measure using 6 variables:
       - x1, x2, x3 > 0 for the positive support points (log-transformed for optimization).
       - w1, w2, w3 > 0 for the weights of the positive points (softmax or direct with constraint).
       We will use a direct parameterization with bounds and handle the weight sum constraint 
       by normalizing internally, or better, use a simplex parameterization.
       
       Actually, let's use:
       s = [0, exp(u1), exp(u2), exp(u3)]
       w_raw = [v0, v1, v2, v3]
       w = w_raw / sum(w_raw)
       
       Variables: u1, u2, u3 (log support), v0, v1, v2, v3 (log weights? or raw?)
       Let's use raw weights but constrained to be positive? 
       Nelder-Mead doesn't handle bounds well without transformation.
       
       Better parameterization:
       u = [u1, u2, u3] -> s = [0, exp(u1), exp(u2), exp(u3)]
       v = [v1, v2, v3] -> weights for s[1], s[2], s[3]. 
       weight for s[0] is 1 - sum(w_pos).
       To ensure sum(w_pos) < 1, we can use a sigmoid or just constrain v_i in (0, 1) and 
       check validity. If sum > 1, we can project or penalize.
       
       Let's use a robust approach:
       Variables z (6-dim):
       z[0], z[1], z[2] : log support points a, b, c.
       z[3], z[4], z[5] : logits for weights.
       
       s = [0, exp(z0), exp(z1), exp(z2)]
       w_pos = softmax(z[3:6])  # This sums to 1 for the positive part? No.
       
       Let's define:
       w_pos_raw = exp(z[3:6])
       w_pos = w_pos_raw / (1 + sum(w_pos_raw))  # This ensures w_pos < 1 and w_0 > 0.
       w_0 = 1 - sum(w_pos)
       
       This maps R^6 to the valid interior of the 4-point simplex with s_0=0.
       This is a bijection from R^6 to the interior of the manifold.
    """
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob_4pt(s, w):
        """
        Calculate P[X1+X2+X3 < 2*X4] for a 4-point measure.
        s: [0, a, b, c]
        w: [p0, pa, pb, pc]
        """
        # s = np.array([0.0, s[1], s[2], s[3]]) # s already constructed
        # w = np.array([w[0], w[1], w[2], w[3]])
        
        n = 4
        # Precompute all sums of 3
        # Indices 0,1,2,3
        # s[0]=0
        # s1+s2+s3 < 2*s4
        
        p_total = 0.0
        
        # Loop over X4 index l
        for l in range(4):
            threshold = 2.0 * s[l]
            # Sum of w_i w_j w_k for s_i+s_j+s_k < threshold
            sum_w = 0.0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        if s[i] + s[j] + s[k] < threshold:
                            sum_w += w[i] * w[j] * w[k]
            p_total += w[l] * sum_w
            
        return p_total

    def objective(z):
        """
        Objective function for scipy.optimize.minimize.
        Minimizes -P.
        z: 6-dim array.
        """
        # Check time
        if time.time() > deadline:
            return 1e10
            
        # Decode z to s, w
        # s
        s = np.zeros(4)
        s[1] = np.exp(z[0])
        s[2] = np.exp(z[1])
        s[3] = np.exp(z[2])
        
        # w
        # w_pos_raw = exp(z[3:6])
        # To avoid overflow in exp, clip z
        z_w = np.clip(z[3:6], -50, 50)
        w_pos_raw = np.exp(z_w)
        sum_w_pos_raw = np.sum(w_pos_raw)
        
        # w_pos_i = w_pos_raw_i / (1 + sum_w_pos_raw)
        # w_0 = 1 / (1 + sum_w_pos_raw)
        
        w = np.zeros(4)
        w[0] = 1.0 / (1.0 + sum_w_pos_raw)
        w[1] = w_pos_raw[0] / (1.0 + sum_w_pos_raw)
        w[2] = w_pos_raw[1] / (1.0 + sum_w_pos_raw)
        w[3] = w_pos_raw[2] / (1.0 + sum_w_pos_raw)
        
        # Ensure sorted support? 
        # The mapping exp(z) doesn't guarantee order.
        # If s[1] > s[2], the measure is still valid, but our assumption "0 < a < b < c" 
        # is just a label. The probability calculation is symmetric in indices 
        # except for the specific values.
        # However, if we don't sort, we might visit the same measure multiple times 
        # with different permutations. That's fine for optimization, just redundant.
        # But wait, if s[1] and s[2] swap, the weights associated with them swap too 
        # in the parameterization? No, z[0] is tied to w[1].
        # So (s1, w1) is a pair. If we swap s1 and s2, we change the measure.
        # So we don't need to sort. The space covers all 4-point measures with s0=0.
        
        # Calculate probability
        p = calc_prob_4pt(s, w)
        
        return -p

    # Initial guesses
    # We want to explore a range of scales.
    # Log-uniform distribution for initial z.
    
    best_score = -np.inf
    best_z = None
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    # Each Nelder-Mead run might take ~10-100ms depending on convergence.
    # We have ~580s. 
    # Let's try 200 restarts.
    
    n_restarts = 200
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial z
        # z[0:3] (log support): uniform in [-2, 10] (range [0.1, 22000])
        # z[3:6] (log weights): uniform in [-3, 3] (range [0.05, 20])
        
        z0 = np.array([
            rng.uniform(-2, 10),
            rng.uniform(-2, 10),
            rng.uniform(-2, 10),
            rng.uniform(-3, 3),
            rng.uniform(-3, 3),
            rng.uniform(-3, 3)
        ])
        
        # Add some diversity: sometimes start near known good regions?
        # Current best score 1.3472 implies P ~ 0.337 (if seed is 1.0 for P~0.25?).
        # Actually, score = P. The task says "Score is proportional to P". 
        # "1.0 equals the seed program". 
        # Let's assume the score IS P.
        # So we are maximizing P.
        
        try:
            res = minimize(objective, z0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_z = res.x.copy()
                    
        except Exception as e:
            pass

    # Decode best_z
    if best_z is not None:
        s = np.zeros(4)
        s[1] = np.exp(best_z[0])
        s[2] = np.exp(best_z[1])
        s[3] = np.exp(best_z[2])
        
        z_w = np.clip(best_z[3:6], -50, 50)
        w_pos_raw = np.exp(z_w)
        sum_w_pos_raw = np.sum(w_pos_raw)
        
        w = np.zeros(4)
        w[0] = 1.0 / (1.0 + sum_w_pos_raw)
        w[1] = w_pos_raw[0] / (1.0 + sum_w_pos_raw)
        w[2] = w_pos_raw[1] / (1.0 + sum_w_pos_raw)
        w[3] = w_pos_raw[2] / (1.0 + sum_w_pos_raw)
        
        # Final validity check
        if np.any(s < -1e-9) or np.any(w < -1e-9) or np.sum(w) <= 1e-9:
            # Fallback to a safe default if something went wrong
            s = np.array([0.0, 1.0, 2.0, 4.0])
            w = np.array([0.4, 0.3, 0.2, 0.1])
        else:
            # Ensure float64
            s = s.astype(np.float64)
            w = w.astype(np.float64)
            
            return s, w
    else:
        # Fallback
        s = np.array([0.0, 1.0, 2.0, 4.0])
        w = np.array([0.4, 0.3, 0.2, 0.1])
        return s, w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s)
    print(w)
    # Quick score check
    p = 0.0
    n=4
    for l in range(n):
        th = 2*s[l]
        sw = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if s[i]+s[j]+s[k] < th:
                        sw += w[i]*w[j]*w[k]
        p += w[l]*sw
    print("Score:", p)
