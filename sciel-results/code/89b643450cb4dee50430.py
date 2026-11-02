import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def build_h_from_blocks(block_heights, block_boundaries):
    """Build full h vector from block representation."""
    h = np.zeros(N)
    for i, (start, end) in enumerate(block_boundaries):
        h[start:end] = block_heights[i]
    return h

def objective_blocks(block_heights, block_boundaries, beta=50.0):
    """Objective for block-height optimization."""
    h = build_h_from_blocks(block_heights, block_boundaries)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_val + log_sum / beta
    return smooth_max / (s * s)

def objective_blocks_grad(block_heights, block_boundaries, beta=50.0):
    """Gradient of block objective w.r.t. block heights."""
    h = build_h_from_blocks(block_heights, block_boundaries)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(block_heights)
    
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    W = np.fft.fft(weights, n=n_fft)
    grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
    
    smooth_max_val = max_val + np.log(sum_exp) / beta
    
    grad_obj_h = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Map back to block heights
    grad_blocks = np.zeros(len(block_heights))
    for i, (start, end) in enumerate(block_boundaries):
        grad_blocks[i] = np.sum(grad_obj_h[start:end])
    
    return grad_blocks

def optimize_block_heights(block_heights, block_boundaries, deadline, max_iter=300):
    """Optimize block heights given fixed boundaries."""
    K = len(block_heights)
    bounds = [(0, None)] * K
    h0 = np.maximum(block_heights, 0)
    
    for beta in [30.0, 80.0, 200.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                objective_blocks,
                h0,
                jac=lambda x: objective_blocks_grad(x, block_boundaries, beta),
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h0 = np.maximum(res.x, 0)
        except Exception:
            pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    np.random.seed(42)
    
    # Strategy: search over block-based representations
    # For each number of blocks K, generate random block layouts and optimize heights
    
    results = []
    
    for K in [10, 15, 20, 25, 30]:
        if time.time() > deadline - 30:
            break
        
        n_layouts = max(3, int((deadline - time.time() - 15) / (K * 0.5)))
        n_layouts = min(n_layouts, 15)
        
        for layout_idx in range(n_layouts):
            if time.time() > deadline - 10:
                break
            
            # Generate random block boundaries
            # Blocks are contiguous, non-overlapping
            if layout_idx == 0:
                # Even blocks
                edges = np.linspace(0, N, K + 1).astype(int)
                block_boundaries = [(edges[i], edges[i+1]) for i in range(K)]
            elif layout_idx == 1:
                # Gaussian-weighted block sizes
                centers = np.random.uniform(0.1, 0.9, K) * N
                sizes = np.random.uniform(0.02, 0.15, K) * N
                # Make them contiguous and cover [0, N]
                # Sort by center
                order = np.argsort(centers)
                centers = centers[order]
                sizes = sizes[order]
                # Adjust to be contiguous
                block_boundaries = []
                pos = 0
                for i in range(K):
                    size = int(sizes[i])
                    size = max(1, min(size, N - pos))
                    block_boundaries.append((pos, pos + size))
                    pos += size
                # Fill remaining
                if pos < N:
                    block_boundaries[-1] = (block_boundaries[-1][0], N)
            else:
                # Random split points
                splits = np.sort(np.random.permutation(np.arange(1, N))[:K-1])
                splits = np.concatenate([[0], splits, [N]])
                block_boundaries = [(int(splits[i]), int(splits[i+1])) for i in range(K)]
            
            # Remove zero-width blocks
            block_boundaries = [(s, e) for s, e in block_boundaries if e > s]
            K_actual = len(block_boundaries)
            if K_actual < 2:
                continue
            
            # Try multiple initial height patterns
            for init_type in range(3):
                if time.time() > deadline - 8:
                    break
                
                if init_type == 0:
                    # All ones
                    block_heights = np.ones(K_actual)
                elif init_type == 1:
                    # Random positive
                    block_heights = np.random.uniform(0.1, 1.0, K_actual)
                else:
                    # Alternating high/low
                    block_heights = np.ones(K_actual)
                    block_heights[1::2] = 0.3
                
                # Optimize heights
                opt_heights = optimize_block_heights(
                    block_heights, block_boundaries, deadline, max_iter=200
                )
                
                h = build_h_from_blocks(opt_heights, block_boundaries)
                s = np.sum(h)
                if s > 0:
                    h = h / s * s  # keep as is, score is scale-invariant
                    score = compute_score(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
    
    # Additional: try structured patterns (symmetric, etc.)
    if time.time() < deadline - 15:
        # Symmetric two-block patterns
        for split_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 5:
                break
            split = int(N * split_frac)
            block_boundaries = [(0, split), (split, N)]
            for ratio in [0.5, 1.0, 2.0, 5.0, 10.0]:
                block_heights = np.array([1.0, ratio])
                opt_heights = optimize_block_heights(
                    block_heights, block_boundaries, deadline, max_iter=100
                )
                h = build_h_from_blocks(opt_heights, block_boundaries)
                score = compute_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        
        # Three-block symmetric
        for f1 in [0.2, 0.3, 0.4]:
            if time.time() > deadline - 5:
                break
            s1 = int(N * f1)
            s2 = N - s1
            block_boundaries = [(0, s1), (s1, s1 + s2//2), (s1 + s2//2, N)]
            for ratio in [0.5, 1.0, 2.0]:
                block_heights = np.array([1.0, ratio, 1.0])
                opt_heights = optimize_block_heights(
                    block_heights, block_boundaries, deadline, max_iter=100
                )
                h = build_h_from_blocks(opt_heights, block_boundaries)
                score = compute_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Final: fine-tune the best solution with full 600-dim optimization
    if time.time() < deadline - 8:
        try:
            def obj_full(h):
                s = np.sum(h)
                if s < 1e-10:
                    return 1e10
                n_fft = 2 * N - 1
                H = np.fft.fft(h, n=n_fft)
                conv = np.real(np.fft.ifft(H * H))[:2*N-1]
                max_val = np.max(conv)
                shifted = 100.0 * (conv - max_val)
                shifted = np.clip(shifted, -100, 100)
                log_sum = np.log(np.sum(np.exp(shifted)))
                smooth_max = max_val + log_sum / 100.0
                return smooth_max / (s * s)
            
            def grad_full(h):
                s = np.sum(h)
                if s < 1e-10:
                    return np.zeros_like(h)
                n_fft = 2 * N - 1
                H = np.fft.fft(h, n=n_fft)
                conv = np.real(np.fft.ifft(H * H))[:2*N-1]
                max_val = np.max(conv)
                shifted = 100.0 * (conv - max_val)
                shifted = np.clip(shifted, -100, 100)
                exp_shifted = np.exp(shifted)
                sum_exp = np.sum(exp_shifted)
                weights = exp_shifted / sum_exp
                W = np.fft.fft(weights, n=n_fft)
                grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
                smooth_max_val = max_val + np.log(sum_exp) / 100.0
                grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
                return grad_obj
            
            res = minimize(
                obj_full,
                best_h,
                jac=grad_full,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
