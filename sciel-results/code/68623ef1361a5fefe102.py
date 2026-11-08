import time
import numpy as np

N = 100

def _compute_all_overlaps_linear(h):
    n = len(h)
    dx = 2.0 / n
    overlaps = np.zeros(n)
    for k in range(n):
        overlaps[k] = np.sum(h[:n-k] * (1 - h[k:])) * dx
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Try a few random starts
    for start_idx in range(50):
        if time.time() > deadline:
            break
            
        h = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        h[ones_indices] = 1.0
        
        overlaps = _compute_all_overlaps_linear(h)
        current_score = np.max(overlaps)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            max_k = np.argmax(overlaps)
            max_val = overlaps[max_k]
            
            # Try flipping bits
            best_flip = -1
            best_new_score = max_val
            
            # We can limit the search to bits that affect the current max
            # Or just try all
            for j in range(N):
                old_val = h[j]
                h[j] = 1.0 - old_val
                
                # Update overlaps
                # Only k where j is involved
                # Affected k: k <= j or k <= N-1-j
                # Let's just recompute all for simplicity if N is small
                # But N=100, 100 bits * 100 k's = 10k ops.
                # 100 bits = 1M ops.
                # This is too slow for many iterations.
                
                # Optimization:
                # Compute delta for affected k's
                # But we need the new max.
                # If the max_k is not affected, and no other k exceeds it, score stays same.
                # If max_k is affected, it might decrease.
                
                # Let's just recompute the specific max_k and its neighbors?
                # Risky.
                
                # Let's try a different move: Swap two bits.
                # Actually, let's just use the slow method but fewer iterations.
                
                new_overlaps = _compute_all_overlaps_linear(h)
                new_score = np.max(new_overlaps)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_flip = j
                    best_new_overlaps = new_overlaps.copy()
                
                h[j] = old_val
                
            if best_flip != -1:
                h[best_flip] = 1.0 - h[best_flip]
                overlaps = best_new_overlaps
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                improved = True
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
