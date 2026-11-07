import numpy as np
import time

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def construct():
    t0 = time.time()
    
    # Initialize with random binary sequence
    # 50 ones, 50 zeros
    h = np.zeros(N)
    h[:50] = 1.0
    np.random.shuffle(h)
    
    best_h = h.copy()
    best_val = objective(best_h)
    
    # Local search
    # We will iterate, trying to reduce the max correlation
    # To speed up, we only consider swaps that target the current max lag
    
    for it in range(50):
        if remaining() < 5:
            break
            
        c = np.correlate(h, 1.0 - h, mode="full")
        max_val = np.max(c)
        max_lag = np.argmax(c)
        
        # If we can't improve, break?
        # Or just continue to explore.
        
        # Identify indices i such that h[i]=1 and h[i+k]=0 for the max lag
        # Note: c[max_lag] corresponds to correlation at shift k = max_lag - (N-1)
        # Let's work with the shift k
        k_shift = max_lag - (N - 1)
        
        # Find i such that h[i]=1 and h[i+k_shift]=0
        # i must be valid for both h[i] and h[i+k_shift]
        valid_i = []
        for i in range(N):
            j = i + k_shift
            if 0 <= j < N:
                if h[i] == 1.0 and h[j] == 0.0:
                    valid_i.append(i)
        
        if not valid_i:
            # No direct way to reduce this lag by swapping 1->0 at i
            # Try other lags? Or just break?
            # Let's try to reduce the second highest lag
            c_sorted = np.sort(c)[::-1]
            # Just continue to next iteration, maybe a random perturbation?
            # Or try swapping to reduce a different high lag
            # For now, let's just pick a random i where h[i]=1 and random j where h[j]=0
            # and see if it helps
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            
            improved = False
            # Try a few random swaps
            for _ in range(20):
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                if i == j: continue
                
                h[i], h[j] = h[j], h[i]
                new_val = objective(h)
                if new_val < best_val:
                    best_h = h.copy()
                    best_val = new_val
                    improved = True
                    break
                else:
                    h[i], h[j] = h[j], h[i]
            
            if not improved:
                # Try to restart with a new random seed if stuck?
                # Or just break
                # Let's try a few more iterations
                continue

        # Try swapping h[i] with h[j] for j in zeros
        zeros = np.where(h == 0.0)[0]
        improved = False
        
        for i in valid_i:
            if improved:
                break
            for j in zeros:
                if i == j:
                    continue
                
                # Swap
                h[i], h[j] = h[j], h[i]
                
                # Evaluate
                new_val = objective(h)
                
                if new_val < best_val:
                    best_h = h.copy()
                    best_val = new_val
                    improved = True
                    break
                else:
                    # Undo swap
                    h[i], h[j] = h[j], h[i]
        
        if not improved:
            # Stuck in local minimum
            # Try a random perturbation
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            if len(ones) > 1 and len(zeros) > 1:
                # Swap two random pairs?
                # Just swap one random pair
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                if i != j:
                    h[i], h[j] = h[j], h[i]
                    # Accept even if worse? No, we want to minimize.
                    # But to escape local min, we might need to accept worse.
                    # However, the landscape is discrete and small.
                    # Let's just try to find any improvement from the new state.
                    # If no improvement found in next iters, we are stuck.
    
    return best_h
