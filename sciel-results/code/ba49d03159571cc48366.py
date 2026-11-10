import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s usage
    deadline = start_time + 145.0 

    # 1. Precompute all 4-term arithmetic progressions (APs)
    # Indices are 0-based: i1, i2, i3, i4
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    num_aps = len(aps)
    ap_indices = np.array(aps, dtype=np.int32)
    
    # 2. Precompute "other" indices for each AP position
    # other_indices[ap_idx, pos_idx, k] is the index of the k-th other element in the AP
    other_indices = np.zeros((num_aps, 4, 3), dtype=np.int32)
    for pos in range(4):
        for k in range(3):
            other_indices[:, pos, k] = ap_indices[:, (pos + k + 1) % 4]
            
    # 3. Precompute bit-to-AP mapping for fast delta calculation
    bit_ap_ids = []
    bit_positions = []
    bit_others = []
    
    # First pass: count how many APs each bit is in
    counts = np.zeros(N, dtype=np.int32)
    for ap_id in range(num_aps):
        for pos in range(4):
            bit_idx = ap_indices[ap_id, pos]
            counts[bit_idx] += 1
            
    # Allocate arrays
    for b in range(N):
        M = counts[b]
        bit_ap_ids.append(np.zeros(M, dtype=np.int32))
        bit_positions.append(np.zeros(M, dtype=np.int32))
        bit_others.append(np.zeros((M, 3), dtype=np.int32))
        
    # Fill arrays
    cursors = np.zeros(N, dtype=np.int32)
    for ap_id in range(num_aps):
        for pos in range(4):
            bit_idx = ap_indices[ap_id, pos]
            idx = cursors[bit_idx]
            bit_ap_ids[bit_idx][idx] = ap_id
            bit_positions[bit_idx][idx] = pos
            bit_others[bit_idx][idx] = other_indices[ap_id, pos, :]
            cursors[bit_idx] += 1

    def count_mono(colors):
        c0 = colors[ap_indices[:, 0]]
        c1 = colors[ap_indices[:, 1]]
        c2 = colors[ap_indices[:, 2]]
        c3 = colors[ap_indices[:, 3]]
        mono_mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mono_mask)

    def get_single_flip_deltas(colors):
        """
        Compute delta for flipping each single bit.
        delta[b] = (new_count - old_count) if we flip b.
        """
        deltas = np.zeros(N, dtype=np.int32)
        for b in range(N):
            others = bit_others[b]
            if others.shape[0] == 0:
                continue
            
            o0 = colors[others[:, 0]]
            o1 = colors[others[:, 1]]
            o2 = colors[others[:, 2]]
            
            others_mono = (o0 == o1) & (o1 == o2)
            cb = colors[b]
            is_same = (cb == o0)
            
            mono_and_same = np.sum(others_mono & is_same)
            mono_and_diff = np.sum(others_mono & ~is_same)
            
            deltas[b] = mono_and_diff - mono_and_same
            
        return deltas

    def get_block_flip_delta(colors, start, end):
        """
        Compute delta for flipping a contiguous block [start, end].
        This is more expensive but allows escaping local minima.
        """
        old_count = count_mono(colors)
        
        # Flip
        colors[start:end+1] = 1 - colors[start:end+1]
        
        new_count = count_mono(colors)
        
        # Unflip
        colors[start:end+1] = 1 - colors[start:end+1]
        
        return new_count - old_count

    # 4. Simulated Annealing with Block Flips
    rng = np.random.default_rng(42)
    
    # Initialize
    colors = rng.integers(0, 2, size=N, dtype=np.uint8)
    current_count = count_mono(colors)
    
    best_count = current_count
    best_colors = colors.copy()
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    max_block_len = 10
    
    steps = 0
    while time.time() < deadline and T > T_min:
        # Choose move type
        # 80% single flip, 20% block flip
        if rng.random() < 0.8:
            # Single flip
            b = rng.integers(0, N)
            delta = get_single_flip_deltas(colors)[b]
            # Update colors
            colors[b] = 1 - colors[b]
        else:
            # Block flip
            length = rng.integers(1, max_block_len + 1)
            start = rng.integers(0, N - length + 1)
            end = start + length - 1
            delta = get_block_flip_delta(colors, start, end)
            # Update colors
            colors[start:end+1] = 1 - colors[start:end+1]
            
        new_count = current_count + delta
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvements
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()
        else:
            # Accept worse solutions with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                current_count = new_count
        
        # Cooling
        T *= cooling_rate
        
        # Re-heating strategy: if stuck for a while, re-heat
        # Simple heuristic: if no improvement in last 1000 steps, re-heat
        # We'll track steps since last improvement
        # For simplicity, we'll just use a fixed schedule but add a restart logic if T is very low
        
        steps += 1
        
        # Periodic check for time
        if steps % 1000 == 0 and time.time() > deadline - 10.0:
            break
            
    # Final Polish: Steepest Descent from best state
    if time.time() < deadline - 5.0:
        colors = best_colors.copy()
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            deltas = get_single_flip_deltas(colors)
            min_delta = np.min(deltas)
            if min_delta < 0:
                candidates = np.where(deltas == min_delta)[0]
                b = candidates[0]
                colors[b] = 1 - colors[b]
                current_count += min_delta
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            else:
                break

    return best_colors.tolist()

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
