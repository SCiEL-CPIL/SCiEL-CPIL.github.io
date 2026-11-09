# EVOLVE-BLOCK-START
import numpy as np

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    best_P = None
    best_score = -1
    
    def shannon_entropy(p):
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))
    
    # Try different K values for x + 2y < K
    for K in range(5, 25):
        P_curr = np.zeros((n, n), dtype=np.float64)
        count = 0
        for y in range(n):
            for x in range(n):
                if x + 2 * y < K:
                    P_curr[y, x] = 1.0
                    count += 1
        
        if count == 0:
            continue
            
        P_curr /= count
        
        # Calculate Entropies
        # H_{-1}: x - y
        p_minus1 = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P_curr[y, x] > 0:
                    p_minus1[x - y + n - 1] += P_curr[y, x]
        H_minus1 = shannon_entropy(p_minus1)
        
        # H_0: x
        p_x = np.sum(P_curr, axis=0)
        H_0 = shannon_entropy(p_x)
        
        # H_1: x + y
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P_curr[y, x] > 0:
                    p_xy[x + y] += P_curr[y, x]
        H_1 = shannon_entropy(p_xy)
        
        # H_2: x + 2y
        p_x2y = np.zeros(3*n - 1)
        for y in range(n):
            for x in range(n):
                if P_curr[y, x] > 0:
                    p_x2y[x + 2*y] += P_curr[y, x]
        H_2 = shannon_entropy(p_x2y)
        
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            continue
            
        score = H_minus1 / M
        if score > best_score:
            best_score = score
            best_P = P_curr.copy()
            
    if best_P is None:
        return np.ones((n, n), dtype=np.float64) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
