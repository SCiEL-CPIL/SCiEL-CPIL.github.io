import numpy as np
from scipy.optimize import minimize_scalar
import time
import random

# Precompute powers for speed? No, z changes.
# But we can precompute the structure.

def fast_score(c):
    # c is a numpy array of 1s and -1s
    # Grid
    thetas = np.linspace(0, 2*np.pi, 500, endpoint=False)
    z = np.exp(1j * thetas)
    powers = np.arange(24)
    # Compute P(z)
    # c is shape (24,)
    # z is shape (500,)
    # We want sum_k c_k z^k
    # z_k = z^k for each k
    # Matrix M of shape (500, 24) where M[i, k] = z_i^k
    # P_i = sum_k M[i, k] c_k
    
    # Efficient way:
    # M = z[:, None] ** powers[None, :]
    # P = M @ c
    
    M = z[:, None] ** powers[None, :]
    P = M @ c
    mags = np.abs(P)
    
    max_val = np.max(mags)
    max_idx = np.argmax(mags)
    theta0 = thetas[max_idx]
    
    # Refine
    # Define objective
    def obj(theta):
        z_ref = np.exp(1j * theta)
        # Compute P(z_ref)
        # z_ref ** powers
        # This is a vector of 24 complex numbers
        # Dot with c
        v = np.dot(c, z_ref ** powers)
        return -np.abs(v)
        
    step = thetas[1] - thetas[0]
    # Bounds
    res = minimize_scalar(obj, bounds=(theta0 - step, theta0 + step), method='bounded', options={'xatol': 1e-10})
    if res.fun < 0:
        max_val = -res.fun
        # Check if there's another peak? 
        # For now, assume the global max is near the grid max.
        # If we want to be safe, check top 2.
        # But for local search, consistency is key.
        
    return max_val / np.sqrt(24)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # GRS
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:24])
    
    best_c = grs.copy()
    best_score = fast_score(best_c)
    
    # Local Search
    current_c = best_c.copy()
    current_score = best_score
    
    # Precompute powers for scoring
    powers = np.arange(24)
    
    def score_vec(c_vec):
        thetas = np.linspace(0, 2*np.pi, 500, endpoint=False)
        z = np.exp(1j * thetas)
        M = z[:, None] ** powers[None, :]
        P = M @ c_vec
        mags = np.abs(P)
        max_val = np.max(mags)
        max_idx = np.argmax(mags)
        theta0 = thetas[max_idx]
        
        def obj(theta):
            z_ref = np.exp(1j * theta)
            v = np.dot(c_vec, z_ref ** powers)
            return -np.abs(v)
            
        step = thetas[1] - thetas[0]
        res = minimize_scalar(obj, bounds=(theta0 - step, theta0 + step), method='bounded', options={'xatol': 1e-10})
        if res.fun < 0:
            max_val = -res.fun
        return max_val / np.sqrt(24)

    # Hill Climb
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(24):
            if time.time() > deadline:
                break
            # Flip bit i
            old_val = current_c[i]
            current_c[i] = -old_val
            s = score_vec(current_c)
            if s < current_score - 1e-9:
                current_score = s
                improved = True
                if s < best_score:
                    best_score = s
                    best_c = current_c.copy()
                    if best_score < 1.2583758 + 1e-6:
                        # Found optimal or near-optimal
                        return best_c.tolist()
            else:
                # Revert
                current_c[i] = old_val

    # Random Restarts
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        kick_c = current_c.copy()
        i1 = random.randint(0, 23)
        i2 = random.randint(0, 23)
        if i1 == i2:
            i2 = (i2 + 1) % 24
        kick_c[i1] = -kick_c[i1]
        kick_c[i2] = -kick_c[i2]
        
        kick_score = score_vec(kick_c)
        
        # Hill climb from kick
        improved_kick = True
        while improved_kick and time.time() < deadline:
            improved_kick = False
            for i in range(24):
                if time.time() > deadline:
                    break
                old_val = kick_c[i]
                kick_c[i] = -old_val
                s = score_vec(kick_c)
                if s < kick_score - 1e-9:
                    kick_score = s
                    improved_kick = True
                else:
                    kick_c[i] = old_val
        
        if kick_score < best_score:
            best_score = kick_score
            best_c = kick_c.copy()
            current_c = kick_c.copy()
            current_score = kick_score
            if best_score < 1.2583758 + 1e-6:
                return best_c.tolist()

    return best_c.tolist()
