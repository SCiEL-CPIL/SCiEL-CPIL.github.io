import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110
    
    # 1. Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Build incidence matrix A (n x num_aps)
    A = np.zeros((n, num_aps), dtype=np.float32)
    pos_aps = [[] for _ in range(n)]
    
    for idx, (p0, p1, p2, p3) in enumerate(aps):
        A[p0, idx] = 1.0
        A[p1, idx] = 1.0
        A[p2, idx] = 1.0
        A[p3, idx] = 1.0
        
        pos_aps[p0].append(idx)
        pos_aps[p1].append(idx)
        pos_aps[p2].append(idx)
        pos_aps[p3].append(idx)

    def compute_P_aps_fast(spins):
        p0 = spins[aps_arr[:, 0]]
        p1 = spins[aps_arr[:, 1]]
        p2 = spins[aps_arr[:, 2]]
        p3 = spins[aps_arr[:, 3]]
        return p0 * p1 * p2 * p3

    def get_score(spins):
        P_aps = compute_P_aps_fast(spins)
        S = float(np.sum(P_aps))
        # Number of mono APs = (M + S) / 2
        return (num_aps + S) / 2.0

    def local_search(spins, max_iters=1000):
        P_aps = compute_P_aps_fast(spins)
        S = float(np.sum(P_aps))
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            Sigma = A.T @ P_aps
            p_best = int(np.argmax(Sigma))
            max_val = Sigma[p_best]
            
            if max_val <= 1e-6: # Small epsilon for float error
                break
            
            # Flip p_best
            spins[p_best] *= -1.0
            
            # Update P_aps and S
            sum_old = 0.0
            for idx in pos_aps[p_best]:
                sum_old += P_aps[idx]
                P_aps[idx] *= -1.0
            S -= 2.0 * sum_old
            
        return spins

    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def to_spins(colors):
        return np.array([1.0 if c == 0 else -1.0 for c in colors], dtype=np.float32)
        
    def from_spins(spins):
        return [0 if s == 1.0 else 1 for s in spins]
        
    random.seed(42)
    
    best_score = float('inf')
    best_spins = None
    
    # Initial Thue-Morse
    init_colors = thue_morse(n)
    spins = to_spins(init_colors)
    
    # ILS Loop
    while time.time() < deadline - 5:
        # Local Search
        spins = local_search(spins, max_iters=500)
        
        # Evaluate
        score = get_score(spins)
        
        if score < best_score:
            best_score = score
            best_spins = spins.copy()
            
        # Perturb
        # Flip k random positions
        k = random.randint(3, 10)
        for _ in range(k):
            p = random.randint(0, n - 1)
            spins[p] *= -1.0
            
    # Final polish on best
    # best_spins is already locally optimal, but just in case
    # It's fast, so we can run it again
    best_spins = local_search(best_spins, max_iters=100)
    
    best_colors = from_spins(best_spins)
    return best_colors
