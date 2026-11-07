"""
Erdos minimum overlap: minimize max correlation of h with 1-h.

New approach: Direct combinatorial/structural optimization using
binary-ish patterns and local exchange heuristics, combined with
a novel "gradient-free" pattern search over structured families.

Key insight: The optimal solution likely has a periodic or quasi-periodic
structure. We explore families of structured patterns (block codes,
Run-length encodings, low-discrepancy sequences) and refine them.
"""

import numpy as np
import time

N = 100


def _value(h):
    """Compute the evaluator's value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _project(h):
    """Project h to feasible set: 0<=h<=1, sum(h)=N/2 within tolerance."""
    h = np.clip(h.astype(float), 0.0, 1.0)
    target = N / 2.0
    for _ in range(50):
        diff = np.sum(h) - target
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h > 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] -= diff / cnt
        else:
            mask = h < 1.0 - 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] -= diff / cnt  # diff negative, so adds
        h = np.clip(h, 0.0, 1.0)
    # Final tiny correction
    diff = np.sum(h) - target
    if abs(diff) > 1e-12:
        idx = np.argmax(h) if diff > 0 else np.argmin(h)
        h[idx] -= diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _try_pattern(pattern, h_base=None):
    """
    Given a binary or continuous pattern of length N, try to use it as a
    starting point and refine via local exchanges.
    """
    if h_base is None:
        h = pattern.copy()
    else:
        h = h_base.copy()
    h = _project(h)
    return h


def _local_exchange(h, max_rounds=500, rng=None):
    """
    Local search using subgradient descent on the max correlation.
    For the lag with maximum correlation, compute gradient and move
    in the negative gradient direction while maintaining feasibility.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    best_h = h.copy()
    best_val = _value(best_h)
    step_size = 0.1
    
    for round_num in range(max_rounds):
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        lag = max_idx - (N - 1)
        
        # Compute gradient of corr[lag] w.r.t. h
        # corr[lag] = sum_{i} h[i] * (1 - h[i+lag]) for valid i
        # = sum_i h[i] - sum_i h[i]*h[i+lag]
        # d/dh[i] = 1 - h[i+lag] for i where i+lag in range
        #           - h[i-lag] for i where i-lag in range
        #           1 - h[i+lag] - h[i-lag] if both exist
        grad = np.zeros(N)
        
        if lag >= 0:
            # For i in [0, N-lag): term h[i]*(1-h[i+lag])
            # d/dh[i] += (1 - h[i+lag])
            # d/dh[i+lag] += -h[i]
            for i in range(N - lag):
                grad[i] += (1 - best_h[i + lag])
                grad[i + lag] += (-best_h[i])
        else:
            # lag < 0, let L = -lag
            # For i in [L, N): term h[i]*(1-h[i-L])
            # d/dh[i] += (1 - h[i-L])
            # d/dh[i-L] += -h[i]
            L = -lag
            for i in range(L, N):
                grad[i] += (1 - best_h[i - L])
                grad[i - L] += (-best_h[i])
        
        # Move in negative gradient direction
        # But we need to maintain sum(h) = N/2, so project gradient to sum=0
        grad_mean = np.mean(grad)
        grad_centered = grad - grad_mean
        
        # Try several step sizes
        improved = False
        for alpha in [step_size, step_size * 0.5, step_size * 0.1]:
            new_h = best_h - alpha * grad_centered
            new_h = _project(new_h)
            v = _value(new_h)
            if v < best_val - 1e-12:
                best_h = new_h
                best_val = v
                improved = True
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(5):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                delta = rng.uniform(0.05, 0.2) * (1 if rng.random() < 0.5 else -1)
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    return best_h, best_val


def _generate_structured_patterns():
    """Generate diverse structured patterns."""
    patterns = []
    
    # 1. Block codes with various block sizes and offsets
    for block_size in range(2, 20):
        for offset in range(block_size):
            h = np.zeros(N)
            for i in range(N):
                block_idx = (i + offset) // block_size
                h[i] = 1.0 if block_idx % 2 == 0 else 0.0
            patterns.append(h)
    
    # 2. Low-discrepancy-like: van der Corput sequence
    def van_der_corput(n, base=2):
        res = 0.0
        f = 1.0 / base
        while n > 0:
            res += f * (n % base)
            n //= base
            f /= base
        return res
    
    for base in [2, 3, 5]:
        vd = np.array([van_der_corput(i, base) for i in range(N)])
        # Map to [0,1] with mean 0.5
        h = 0.5 + 0.5 * (2 * vd - 1)
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    # 3. Thue-Morse-like
    def thue_morse(n):
        # Count bits
        return bin(n).count('1') % 2
    
    for shift in range(3):
        tm = np.array([thue_morse(i + shift) for i in range(N)])
        h = tm.astype(float)
        patterns.append(h)
    
    # 4. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 25]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h = np.clip(h, 0, 1)
            patterns.append(h)
    
    # 5. Sawtooth waves
    for period in [5, 7, 10, 11, 13, 17, 20, 25]:
        t = np.arange(N) % period / period
        h = 2 * t
        h = np.where(h < 1, h, 2 - h)  # Triangle wave
        patterns.append(h)
    
    # 6. Random with structure: clusters
    rng = np.random.default_rng(42)
    for n_clusters in [2, 3, 4, 5, 7, 10]:
        centers = rng.uniform(0, N, n_clusters)
        widths = rng.uniform(2, 5, n_clusters)
        h = np.zeros(N)
        for c, w in zip(centers, widths):
            dist = np.abs(np.arange(N) - c)
            h += np.exp(-dist**2 / (2 * w**2))
        h = h / np.max(h) if np.max(h) > 0 else h
        h = 0.5 + 0.5 * (2 * h - 1)
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    return patterns


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(100)
    
    # Start with seed
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Phase 1: Try structured patterns
    patterns = _generate_structured_patterns()
    
    for pat in patterns:
        if time.time() > deadline - 60.0:
            break
        h = _project(pat)
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Local exchange refinement on best and some others
    # Refine the best
    if time.time() < deadline - 50.0:
        h_ref, v_ref = _local_exchange(best_h, max_rounds=300, rng=rng)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()
    
    # Phase 3: Multi-start local exchange from diverse points
    if time.time() < deadline - 30.0:
        # Take top few patterns and refine
        scored = []
        for pat in patterns[:50]:
            h = _project(pat)
            v = _value(h)
            scored.append((v, h))
        scored.sort(key=lambda x: x[0])
        
        for v, h in scored[:15]:
            if time.time() > deadline - 20.0:
                break
            h_ref, v_ref = _local_exchange(h, max_rounds=150, rng=rng)
            if v_ref < best_val:
                best_val = v_ref
                best_h = h_ref.copy()
    
    # Phase 4: Final polish with more iterations
    if time.time() < deadline - 5.0:
        best_h, best_val = _local_exchange(best_h, max_rounds=200, rng=rng)
    
    # Final projection
    best_h = _project(best_h)
    final_val = _value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure sum is exactly N/2
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-10:
        # Distribute correction evenly among elements not at bounds
        mask = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] -= diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
        else:
            # Fallback: adjust one element
            idx = np.argmax(best_h) if diff > 0 else np.argmin(best_h)
            best_h[idx] -= diff
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
