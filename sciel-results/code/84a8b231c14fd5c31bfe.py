import numpy as np

def heuristics(prizes, weights):
    # Handle potential shape issues
    # If prizes is (5, 100), sum to (100,)
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1) # unlikely, but safe
        
    # W is (5, 100)
    
    # Aggregate efficiency: sum of prizes / sum of weights
    W_sum = np.sum(W, axis=0)
    H_sum = P / (W_sum + 1e-9)
    
    # Bottleneck-aware efficiency: sum of prizes / soft-max of weights
    # This penalizes items that are heavy in any single constraint more heavily
    # than just looking at total weight
    beta = 5.0
    # Use log-sum-exp trick for stability
    W_max = np.max(W, axis=0)
    log_sum_exp = W_max + np.log(np.sum(np.exp(W - W_max), axis=0))
    W_softmax = log_sum_exp / beta
    
    H_bottleneck = P / (W_softmax + 1e-9)
    
    # Blend the two heuristics
    # alpha controls the weight of the bottleneck-aware component
    # A blend balances overall efficiency with bottleneck constraint awareness
    alpha = 0.5
    H = alpha * H_bottleneck + (1 - alpha) * H_sum
    
    # Ensure positive
    H = np.maximum(H, 1e-9)
    
    return H
