import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 24
    total_candidates = 1 << 23
    
    # Two-pass strategy:
    # Pass 1: Coarse screening with M1=2048 to find top candidates
    # Pass 2: Fine refinement with M2=16384 on top candidates
    
    M1 = 2048  # Coarse grid
    M2 = 16384  # Fine grid for refinement
    
    # Precompute for coarse grid
    angles1 = np.linspace(0, 2 * np.pi, M1, endpoint=False)
    z1 = np.exp(1j * angles1)
    Z1 = np.zeros((N, M1), dtype=complex)
    Z1[0, :] = 1.0
    for k in range(1, N):
        Z1[k, :] = Z1[k-1, :] * z1
    
    base1 = np.sum(Z1[1:N], axis=0)
    Z_flip1 = Z1[1:N]
    AllOnes1 = 1.0 + base1
    
    # Precompute for fine grid
    angles2 = np.linspace(0, 2 * np.pi, M2, endpoint=False)
    z2 = np.exp(1j * angles2)
    Z2 = np.zeros((N, M2), dtype=complex)
    Z2[0, :] = 1.0
    for k in range(1, N):
        Z2[k, :] = Z2[k-1, :] * z2
    
    base2 = np.sum(Z2[1:N], axis=0)
    Z_flip2 = Z2[1:N]
    AllOnes2 = 1.0 + base2
    
    B = 8000  # Batch size for coarse pass
    
    # Pass 1: Coarse screening
    top_candidates = []
    top_scores = []
    
    for start in range(0, total_candidates, B):
        if time.time() > deadline - 40.0:
            break
            
        end = min(start + B, total_candidates)
        B_actual = end - start
        
        idxs = np.arange(start, end, dtype=np.int64)
        
        bits = np.zeros((B_actual, 23), dtype=np.float64)
        for k in range(23):
            bits[:, k] = (idxs >> k) & 1
        
        try:
            contrib = bits @ Z_flip1
            P_vals = AllOnes1[None, :] - 2.0 * contrib
            scores = np.max(np.abs(P_vals), axis=1)
            
            # Keep top 50 from this batch
            local_top = np.argpartition(scores, 50)[:50]
            for li in local_top:
                top_candidates.append(start + int(li))
                top_scores.append(scores[li])
        except Exception:
            pass
    
    # Sort top candidates by score and keep top 200
    if top_candidates:
        top_candidates = np.array(top_candidates)
        top_scores = np.array(top_scores)
        sort_idx = np.argsort(top_scores)
        top_candidates = top_candidates[sort_idx][:200]
        top_scores = top_scores[sort_idx][:200]
        
        # Pass 2: Fine refinement on top candidates
        best_score = np.inf
        best_idx = 0
        
        for idx in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            bits = np.zeros((1, 23), dtype=np.float64)
            for k in range(23):
                bits[0, k] = (idx >> k) & 1
            
            contrib = bits @ Z_flip2
            P_vals = AllOnes2[None, :] - 2.0 * contrib
            score = np.max(np.abs(P_vals))
            
            if score < best_score:
                best_score = score
                best_idx = int(idx)
        
        # Final polish: high-precision evaluation of best few candidates
        # Use even finer grid + golden-section refinement
        if best_idx < total_candidates and time.time() < deadline - 5.0:
            # Take top 5 candidates from pass 2
            # Re-evaluate them with M=65536 and local refinement
            M3 = 65536
            angles3 = np.linspace(0, 2 * np.pi, M3, endpoint=False)
            z3 = np.exp(1j * angles3)
            Z3 = np.zeros((N, M3), dtype=complex)
            Z3[0, :] = 1.0
            for k in range(1, N):
                Z3[k, :] = Z3[k-1, :] * z3
            
            base3 = np.sum(Z3[1:N], axis=0)
            Z_flip3 = Z3[1:N]
            AllOnes3 = 1.0 + base3
            
            # Evaluate top 5 with fine grid
            top5 = top_candidates[:5]
            for idx in top5:
                if time.time() > deadline - 3.0:
                    break
                
                bits = np.zeros((1, 23), dtype=np.float64)
                for k in range(23):
                    bits[0, k] = (idx >> k) & 1
                
                contrib = bits @ Z_flip3
                P_vals = AllOnes3[None, :] - 2.0 * contrib
                score = np.max(np.abs(P_vals))
                
                if score < best_score:
                    best_score = score
                    best_idx = int(idx)
    
    # Reconstruct best coefficients
    best_coeffs = np.ones(N)
    for k in range(23):
        if (best_idx >> k) & 1:
            best_coeffs[k + 1] = -1.0
    
    return [int(c) for c in best_coeffs]
