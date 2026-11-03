"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all circular shifts."""
    # h is a 1D array of length N
    # For each shift s, overlap_s = sum_i min(h[i], 1 - h[(i+s) % N])
    # We want to minimize max_s overlap_s
    max_overlap = 0.0
    for s in range(N):
        shifted = np.roll(h, -s)
        # overlap = sum of min(h[i], 1 - shifted[i])
        overlap = np.sum(np.minimum(h, 1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _construct_sinusoidal(freq, phase, amplitude=0.5):
    """Construct a sinusoidal pattern with given frequency and phase."""
    i = np.arange(N)
    h = 0.5 + amplitude * np.sin(2 * np.pi * freq * i / N + phase)
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum is exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if abs(current_sum - target_sum) > 1e-10:
        # Scale and clip, then fine-tune
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
        # Final adjustment: distribute the difference
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-10:
            # Add a small constant to all elements, then clip
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            # If still off, adjust individual elements
            residual = target_sum - np.sum(h)
            if abs(residual) > 1e-10:
                # Adjust the element with most room
                if residual > 0:
                    # Find elements below 1
                    mask = h < 1.0
                    if np.any(mask):
                        idx = np.where(mask)[0]
                        room = 1.0 - h[idx]
                        order = np.argsort(-room)
                        for k in order:
                            add = min(residual, room[k])
                            h[idx[k]] += add
                            residual -= add
                            if abs(residual) < 1e-10:
                                break
                else:
                    mask = h > 0.0
                    if np.any(mask):
                        idx = np.where(mask)[0]
                        room = h[idx]
                        order = np.argsort(room)
                        for k in order:
                            sub = min(-residual, room[k])
                            h[idx[k]] -= sub
                            residual += sub
                            if abs(residual) < 1e-10:
                                break
    return h

def _construct_sawtooth(period, offset=0.0):
    """Construct a sawtooth pattern."""
    i = np.arange(N)
    # Sawtooth with given period
    t = (i + offset) % period / period
    # Triangle wave: 2*|t - 0.5| gives 1 at ends, 0 at middle
    h = 0.5 + 0.5 * (2 * np.abs(t - 0.5) - 0.5)  # This gives values in [0.25, 0.75]
    # Better: use a proper sawtooth that averages to 0.5
    h = 0.5 + 0.5 * (2 * t - 1)  # Linear ramp from 0 to 1
    h = np.clip(h, 0.0, 1.0)
    # Ensure sum is N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if abs(current_sum - target_sum) > 1e-10:
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-10:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            residual = target_sum - np.sum(h)
            if abs(residual) > 1e-10:
                if residual > 0:
                    mask = h < 1.0
                    if np.any(mask):
                        idx = np.where(mask)[0]
                        room = 1.0 - h[idx]
                        order = np.argsort(-room)
                        for k in order:
                            add = min(residual, room[k])
                            h[idx[k]] += add
                            residual -= add
                            if abs(residual) < 1e-10:
                                break
                else:
                    mask = h > 0.0
                    if np.any(mask):
                        idx = np.where(mask)[0]
                        room = h[idx]
                        order = np.argsort(room)
                        for k in order:
                            sub = min(-residual, room[k])
                            h[idx[k]] -= sub
                            residual += sub
                            if abs(residual) < 1e-10:
                                break
    return h

def _construct_binary_pattern(period):
    """Construct a binary-like pattern with 0s and 1s, adjusted to average 0.5."""
    i = np.arange(N)
    # Create a pattern with period 'period', half 0s and half 1s
    h = np.zeros(N)
    for i in range(N):
        if (i % period) < period // 2:
            h[i] = 1.0
        else:
            h[i] = 0.0
    # Adjust to ensure sum is exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if abs(current_sum - target_sum) > 1e-10:
        # Convert to fractional values
        diff = target_sum - current_sum
        # Adjust elements
        if diff > 0:
            # Need to increase sum, set some 0s to fractional values
            zero_idx = np.where(h == 0.0)[0]
            for idx in zero_idx:
                if diff <= 0:
                    break
                add = min(1.0, diff)
                h[idx] = add
                diff -= add
        else:
            # Need to decrease sum, set some 1s to fractional values
            one_idx = np.where(h == 1.0)[0]
            for idx in one_idx:
                if diff >= 0:
                    break
                sub = min(1.0, -diff)
                h[idx] = 1.0 - sub
                diff += sub
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap_max(best_h)
    
    # Try various sinusoidal patterns
    for freq in range(1, 20):
        if time.time() > deadline:
            break
        for phase in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            if time.time() > deadline:
                break
            h = _construct_sinusoidal(freq, phase)
            score = _compute_overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try sawtooth patterns
    for period in range(2, N + 1):
        if time.time() > deadline:
            break
        for offset in np.linspace(0, period, 10, endpoint=False):
            if time.time() > deadline:
                break
            h = _construct_sawtooth(period, offset)
            score = _compute_overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try binary patterns
    for period in range(2, N + 1):
        if time.time() > deadline:
            break
        h = _construct_binary_pattern(period)
        score = _compute_overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final check
    if time.time() > deadline - 2.0:
        return best_h
    
    return best_h
