import numpy as np

def heuristics(prizes, weights):
    # Handle potential shape variations
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes
        
    w = weights
    # w is (5, 100)
    
    # 1. Calculate total weight per constraint
    S = w.sum(axis=1) # (5,)
    
    # 2. Estimate Shadow Prices
    # Capacity is 25
    # Tighter constraints (higher S) have higher shadow prices
    # Using S / Capacity as a proxy for scarcity/price
    cap = 25.0
    lambdas = S / cap # (5,)
    
    # 3. Calculate Effective Cost per item
    # EC_i = sum_c lambda_c * W_ic
    # w is (5, 100), lambdas is (5,)
    # We need (100,) result
    # EC = w.T @ lambdas
    EC = w.T @ lambdas # (100,)
    
    # 4. Calculate Min-Ratio Efficiency
    # R_ic = P_i / W_ic
    # R_min_i = min_c R_ic
    # Avoid division by zero
    w_safe = w + 1e-9
    ratios = p[None, :] / w_safe # (5, 100)
    R_min = ratios.min(axis=0) # (100,)
    
    # 5. Calculate Shadow Price Heuristic
    # H_sp = P_i / (EC_i + epsilon)
    H_sp = p / (EC + 1e-9)
    
    # 6. Geometric Blend
    # eta = sqrt(H_sp * R_min)
    # Ensure positive values for log/sqrt
    H_sp_safe = np.maximum(H_sp, 1e-9)
    R_min_safe = np.maximum(R_min, 1e-9)
    
    eta = np.sqrt(H_sp_safe * R_min_safe)
    
    # Final safety check
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    
    return eta
