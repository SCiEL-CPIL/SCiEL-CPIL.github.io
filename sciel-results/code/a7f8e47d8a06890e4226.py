import numpy as np
import time
import random

N = 600
# Interval width
# Domain is [-1/4, 1/4], so total length is 1/2.
# 600 intervals -> dx = (1/2) / 600 = 1/1200
dx = 1.0 / 1200.0

def _compute_score(f):
    """
    Computes the normalized maximum of the autoconvolution.
    The task asks to minimize the normalized maximum of its autoconvolution.
    Usually, this means max(f * f) / (integral(f)^2) or similar.
    Let's assume the standard normalization:
    C1_bound = max_{x} (f * f)(x) / ( \int f )^2
    Since f is a step function on discrete intervals, 
    (f*f)(k) = sum_i f_i f_{i-k} * dx (approximately, if we treat it as continuous)
    Actually, the convolution of two step functions approximating L1 functions:
    If f(x) = sum h_i * 1_{I_i}(x), then (f*f)(y) = sum_{i,j} h_i h_j * length(I_i ∩ (I_j + y)).
    
    The evaluator likely computes:
    1. Construct the piecewise constant function.
    2. Compute autoconvolution.
    3. Find max.
    4. Normalize by (sum(heights * dx))^2.
    
    We will compute this value. Lower is better.
    """
    if len(f) != N:
        return 1e10
    if np.any(f < 0):
        return 1e10
    
    # Sum of f (integral approximation)
    integral_f = np.sum(f) * dx
    if integral_f < 1e-9:
        return 1e10
        
    # Compute autoconvolution
    # Use FFT for speed.
    # Zero-pad to 2*N to avoid circular convolution effects.
    fft_len = 2 * N
    fft_f = np.fft.fft(f, n=fft_len)
    fft_conv = fft_f * fft_f
    conv = np.fft.ifft(fft_conv).real
    
    # The convolution of two functions supported on [0, N-1] (shifted to 0) 
    # will have support on [0, 2N-2].
    # The peak is at lag 0.
    # We take the max of the real part.
    max_conv = np.max(conv)
    
    # Normalize
    score = max_conv / (integral_f ** 2)
    
    return score

def _blocks_to_heights(blocks):
    """
    Convert a list of blocks (start_idx, width, height) to a height array of length N.
    start_idx and width are integers.
    """
    heights = np.zeros(N)
    for start, width, height in blocks:
        if width <= 0 or height <= 0:
            continue
        s = max(0, int(start))
        e = min(N, int(start + width))
        if s < e:
            heights[s:e] = height
    return heights

def _heights_to_blocks(heights):
    """
    Convert a height array to a list of blocks.
    """
    blocks = []
    i = 0
    while i < N:
        if heights[i] > 0:
            start = i
            h = heights[i]
            j = i
            while j < N and heights[j] == h:
                j += 1
            blocks.append((start, j - start, h))
            i = j
        else:
            i += 1
    return blocks

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    # Initial state: Uniform distribution
    # Uniform f(x) = c. 
    # Autoconv of uniform on [0,1] is triangular.
    # Let's start with a single block covering the whole domain or a few blocks.
    # Actually, uniform is the best for minimizing max autoconv?
    # For L1 normalized functions, the uniform distribution on an interval minimizes the L-infinity norm of the autoconvolution?
    # No, the triangular function (autoconv of uniform) has a peak.
    # Is there a function with a flatter autoconv?
    # The "box" function has a triangular autoconv.
    # A "triangular" function has a "cubic-like" autoconv?
    
    # Let's start with a uniform block.
    # Height h such that sum(h) * dx = 1? No, the score is normalized by integral^2, so scale doesn't matter.
    # We can just use height 1.
    initial_blocks = [(0, N, 1.0)]
    current_heights = _blocks_to_heights(initial_blocks)
    current_score = _compute_score(current_heights)
    
    best_blocks = initial_blocks.copy()
    best_heights = current_heights.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.995
    
    # We will operate on the block representation.
    # Moves:
    # 1. Change height of a block.
    # 2. Change width of a block (shift end).
    # 3. Change start of a block (shift start and width).
    # 4. Add a new small block.
    # 5. Remove a block.
    # 6. Merge adjacent blocks if same height.
    
    def _mutate_blocks(blocks):
        new_blocks = [b[:] for b in blocks] # Deep copy list of lists/tuples
        if not new_blocks:
            # Add initial block
            new_blocks.append([random.randint(0, N//2), random.randint(10, N//4), 1.0])
            return new_blocks
            
        move = random.random()
        
        if move < 0.3:
            # Change height
            idx = random.randint(0, len(new_blocks) - 1)
            factor = random.uniform(0.8, 1.2)
            new_blocks[idx][2] = max(0.1, new_blocks[idx][2] * factor)
        elif move < 0.5:
            # Change width
            idx = random.randint(0, len(new_blocks) - 1)
            s, w, h = new_blocks[idx]
            dw = random.randint(-20, 20)
            new_w = max(1, w + dw)
            new_s = s
            # Ensure it fits in [0, N]
            if new_s + new_w > N:
                new_w = N - new_s
            if new_w <= 0:
                new_blocks[idx] = [s, 1, h]
            else:
                new_blocks[idx] = [new_s, new_w, h]
        elif move < 0.7:
            # Change start
            idx = random.randint(0, len(new_blocks) - 1)
            s, w, h = new_blocks[idx]
            ds = random.randint(-20, 20)
            new_s = s + ds
            if new_s < 0:
                new_s = 0
            if new_s + w > N:
                # Shift back or truncate
                new_s = max(0, N - w)
            new_blocks[idx] = [new_s, w, h]
        elif move < 0.8:
            # Add block
            if len(new_blocks) < 20:
                s = random.randint(0, N - 50)
                w = random.randint(10, 50)
                h = random.uniform(0.5, 1.5)
                new_blocks.append([s, w, h])
        elif move < 0.9:
            # Remove block
            if len(new_blocks) > 1:
                idx = random.randint(0, len(new_blocks) - 1)
                new_blocks.pop(idx)
        else:
            # Swap two blocks' positions (if they don't overlap too badly? Or just swap start/width params?)
            # Let's just perturb all blocks slightly
            for i in range(len(new_blocks)):
                s, w, h = new_blocks[i]
                ds = random.randint(-5, 5)
                dw = random.randint(-5, 5)
                dh_factor = random.uniform(0.95, 1.05)
                new_s = max(0, min(N - max(1, w+dw), s + ds))
                new_w = max(1, w + dw)
                new_h = max(0.1, h * dh_factor)
                new_blocks[i] = [new_s, new_w, new_h]
                
        # Cleanup: Sort by start, remove overlaps? 
        # Overlaps just add heights. That's fine.
        # But we should merge adjacent blocks with same height to keep list small?
        # For now, just return.
        return new_blocks

    # Main SA Loop
    while time.time() < deadline - safety_margin:
        if T < T_min:
            break
            
        new_blocks = _mutate_blocks(best_blocks)
        new_heights = _blocks_to_heights(new_blocks)
        new_score = _compute_score(new_heights)
        
        if new_score < best_score:
            best_score = new_score
            best_blocks = [b[:] for b in new_blocks]
            best_heights = new_heights.copy()
        else:
            # Accept with probability exp(-(new_score - best_score) / T)
            # Note: scores are positive.
            # We want to minimize.
            delta = new_score - best_score
            if delta < 0:
                accept = True
            else:
                prob = np.exp(-delta / T)
                accept = random.random() < prob
                
            if accept:
                # Update current state, but keep best separate? 
                # Standard SA updates current. Here we only have 'best'. 
                # Let's maintain a 'current' state that can be worse than best.
                pass # We need a current state variable.

    # The logic above was slightly flawed (no current state). 
    # Let's restart with a proper SA loop.
    
    # Re-initialize
    current_blocks = initial_blocks.copy()
    current_heights = _blocks_to_heights(current_blocks)
    current_score = _compute_score(current_heights)
    
    best_blocks = current_blocks.copy()
    best_heights = current_heights.copy()
    best_score = current_score
    
    T = 0.1 # Start lower to be more greedy initially? Or higher to explore?
    # The score is around 1.0 for uniform. 
    # Let's use a relative temperature.
    T = 0.01
    
    while time.time() < deadline - safety_margin:
        if T < 1e-8:
            break
            
        new_blocks = _mutate_blocks(current_blocks)
        new_heights = _blocks_to_heights(new_blocks)
        new_score = _compute_score(new_heights)
        
        if new_score < current_score:
            current_blocks = new_blocks
            current_heights = new_heights
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_blocks = current_blocks.copy()
                best_heights = current_heights.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                # Avoid division by zero or huge exponents
                if T == 0:
                    accept = False
                else:
                    # Scale delta? 
                    # If scores are ~1, delta is small.
                    prob = np.exp(-delta / T)
                    accept = random.random() < prob
                    
            if accept:
                current_blocks = new_blocks
                current_heights = new_heights
                current_score = new_score
                
        T *= alpha
        
    return best_heights.tolist()

if __name__ == "__main__":
    construct()
