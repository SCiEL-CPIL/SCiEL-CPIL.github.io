import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for the 120s limit
    
    n = 24
    total_candidates = 1 << n
    
    # --- Stage 1: Coarse Grid Screening ---
    # Use a grid of 512 points to screen all 2^24 candidates
    # This is dense enough to not miss the top candidates
    grid_size_1 = 512
    angles_1 = np.linspace(0, 2 * np.pi, grid_size_1, endpoint=False)
    z_1 = np.exp(1j * angles_1)
    
    # Precompute Z_1[k, j] = z_1[j]^k
    Z_1 = np.ones((n, grid_size_1), dtype=np.complex128)
    for k in range(1, n):
        Z_1[k] = Z_1[k-1] * z_1
    
    # P_base(z) = sum_{k=0}^{23} z^k
    P_base_1 = np.sum(Z_1, axis=0)
    
    # We want to find top K1 candidates
    K1 = 5000
    # We will store (score, mask)
    # Using a heap is good, but sorting a smaller list at the end of batches is also fine.
    # Let's use a min-heap of size K1 to keep the smallest scores.
    
    import heapq
    heap = []
    
    batch_size = 10000
    
    t0 = time.time()
    for i in range(0, total_candidates, batch_size):
        if time.time() > deadline - 30.0:
            break
            
        end = min(i + batch_size, total_candidates)
        actual_batch_size = end - i
        
        # Generate masks
        masks = np.arange(i, end, dtype=np.int64)
        
        # Extract bits: shape (batch, n)
        # bit k of mask is 1 if coefficient c_k is -1
        # We need a matrix M where M[j, k] = 1 if bit k of masks[j] is 1
        # P_mask = P_base - 2 * sum_{k: M[j,k]=1} z^k
        # sum_{k} M[j,k] Z_1[k, :]
        
        # Efficient bit extraction
        # For each bit position k, (masks >> k) & 1
        # We can do this in a loop over k, which is 24 iterations.
        # Or use np.unpackbits? No, masks are 64-bit integers, unpackbits works on bytes.
        # Loop is fine and fast enough.
        
        M = np.zeros((actual_batch_size, n), dtype=np.float64)
        for k in range(n):
            M[:, k] = (masks >> k) & 1
        
        # Compute P for all candidates in batch
        # Z_1 is (n, grid_size_1). M is (batch, n).
        # M @ Z_1.T -> (batch, grid_size_1)
        P_batch = P_base_1[None, :] - 2.0 * (M @ Z_1.T)
        
        # Calculate max |P| for each candidate
        scores = np.max(np.abs(P_batch), axis=1)
        
        # Update heap
        # We want the smallest scores.
        # If heap is not full, push all.
        # Else, compare with heap[0] and replace if smaller.
        # This is O(batch * log(K)) which might be slow if batch is large.
        # Alternative: Find local minima in the batch, then merge.
        
        # Let's find the top L candidates in this batch
        # L can be small, e.g., 100.
        # Since scores are sorted in ascending order in a min-heap, we want the smallest.
        # np.argpartition gives the indices of the smallest L elements.
        L = min(100, actual_batch_size)
        if L < actual_batch_size:
            # Get indices of L smallest scores
            idxs = np.argpartition(scores, L)[:L]
            local_scores = scores[idxs]
            local_masks = masks[idxs]
        else:
            local_scores = scores
            local_masks = masks
            
        for s, m in zip(local_scores, local_masks):
            if len(heap) < K1:
                heapq.heappush(heap, (s, m))
            elif s < heap[0][0]:
                heapq.heapreplace(heap, (s, m))
                
    # Extract top K1 candidates
    top_candidates_1 = [m for s, m in heap]
    
    # --- Stage 2: Medium Grid Re-ranking ---
    grid_size_2 = 4096
    angles_2 = np.linspace(0, 2 * np.pi, grid_size_2, endpoint=False)
    z_2 = np.exp(1j * angles_2)
    
    Z_2 = np.ones((n, grid_size_2), dtype=np.complex128)
    for k in range(1, n):
        Z_2[k] = Z_2[k-1] * z_2
        
    P_base_2 = np.sum(Z_2, axis=0)
    
    K2 = 50
    heap2 = []
    
    # Process top_candidates_1 in batches of 100
    for i in range(0, len(top_candidates_1), 100):
        if time.time() > deadline - 10.0:
            break
            
        batch_masks = np.array(top_candidates_1[i:i+100], dtype=np.int64)
        actual_batch_size = len(batch_masks)
        
        M = np.zeros((actual_batch_size, n), dtype=np.float64)
        for k in range(n):
            M[:, k] = (batch_masks >> k) & 1
            
        P_batch = P_base_2[None, :] - 2.0 * (M @ Z_2.T)
        scores = np.max(np.abs(P_batch), axis=1)
        
        for s, m in zip(scores, batch_masks):
            if len(heap2) < K2:
                heapq.heappush(heap2, (s, m))
            elif s < heap2[0][0]:
                heapq.heapreplace(heap2, (s, m))
                
    top_candidates_2 = [m for s, m in heap2]
    
    # --- Stage 3: Fine Grid + Refinement ---
    grid_size_3 = 16384
    angles_3 = np.linspace(0, 2 * np.pi, grid_size_3, endpoint=False)
    z_3 = np.exp(1j * angles_3)
    
    Z_3 = np.ones((n, grid_size_3), dtype=np.complex128)
    for k in range(1, n):
        Z_3[k] = Z_3[k-1] * z_3
        
    P_base_3 = np.sum(Z_3, axis=0)
    
    best_score = float('inf')
    best_mask = 0
    
    # Evaluate top candidates on fine grid
    # We only have ~50 candidates, so we can do it one by one or in small batches
    for m in top_candidates_2:
        if time.time() > deadline - 5.0:
            break
            
        # Calculate P on fine grid
        # M is (1, n)
        coeff_mask = np.zeros(n, dtype=np.float64)
        for k in range(n):
            if (m >> k) & 1:
                coeff_mask[k] = 1.0
                
        P_fine = P_base_3 - 2.0 * (coeff_mask @ Z_3.T)
        
        # Find peak
        abs_P = np.abs(P_fine)
        peak_idx = np.argmax(abs_P)
        peak_val_coarse = abs_P[peak_idx]
        
        # If this coarse fine-grid value is already worse than best, skip refinement?
        # Refinement can only increase the accuracy (lower bound on max), but the true max
        # is >= coarse max. So if coarse max > best_score, it might still be the best
        # if best_score is also a lower bound? No, best_score is the *refined* score.
        # So if peak_val_coarse > best_score, it's possible the refined score is higher.
        # But if peak_val_coarse is significantly higher, it's unlikely to beat best_score.
        # Let's just refine all top candidates to be safe, as K2 is small.
        
        # Golden Section Refinement
        theta_peak = angles_3[peak_idx]
        delta = 2 * np.pi / grid_size_3
        a = theta_peak - delta
        b = theta_peak + delta
        
        # Construct coefficients for evaluation
        coeffs = np.ones(n, dtype=np.float64)
        for k in range(n):
            if (m >> k) & 1:
                coeffs[k] = -1.0
                
        def eval_P(theta):
            z_point = np.exp(1j * theta)
            # Horner's method or dot product
            # Dot product is faster for small N
            # Powers: 1, z, z^2, ..., z^23
            # We can compute powers iteratively
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
            
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = eval_P(c)
        fd = eval_P(d)
        
        for _ in range(60):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_P(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_P(d)
                
        refined_val = eval_P((a + b) / 2)
        
        if refined_val < best_score:
            best_score = refined_val
            best_mask = m
            
    # Return coefficients
    final_coeffs = np.ones(n, dtype=int)
    for k in range(n):
        if (best_mask >> k) & 1:
            final_coeffs[k] = -1
            
    return final_coeffs.tolist()
