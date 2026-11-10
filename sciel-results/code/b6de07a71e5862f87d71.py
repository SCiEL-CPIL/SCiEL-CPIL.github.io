import time
import numpy as np
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

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
    # For each bit b, we need the list of (ap_id, pos) pairs it participates in.
    # We will store this as pre-allocated arrays for vectorized access.
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
    # We need a cursor to track the next available slot for each bit
    cursors = np.zeros(N, dtype=np.int32)
    for ap_id in range(num_aps):
        for pos in range(4):
            bit_idx = ap_indices[ap_id, pos]
            idx = cursors[bit_idx]
            bit_ap_ids[bit_idx][idx] = ap_id
            bit_positions[bit_idx][idx] = pos
            bit_others[bit_idx][idx] = other_indices[ap_id, pos, :]
            cursors[bit_idx] += 1

    def get_deltas(colors):
        """
        Compute the change in mono-count for flipping each bit.
        Returns an array of shape (N,) with deltas.
        """
        deltas = np.zeros(N, dtype=np.int32)
        for b in range(N):
            others = bit_others[b]
            if others.shape[0] == 0:
                continue
            
            # Get colors of the other 3 elements in each AP containing b
            o0 = colors[others[:, 0]]
            o1 = colors[others[:, 1]]
            o2 = colors[others[:, 2]]
            
            # Check if the other 3 are monochromatic
            others_mono = (o0 == o1) & (o1 == o2)
            
            # If others are mono:
            # - If colors[b] == o0, the AP is currently mono. Flipping breaks it -> delta -= 1
            # - If colors[b] != o0, the AP is currently not mono. Flipping makes it mono -> delta += 1
            # If others are not mono, flipping b cannot make the AP mono (others are mixed).
            
            cb = colors[b]
            is_same = (cb == o0)
            
            # Count contributions
            # mono_and_same: APs that are currently mono and will become non-mono (-1)
            # mono_and_diff: APs that are currently non-mono and will become mono (+1)
            
            mono_and_same = np.sum(others_mono & is_same)
            mono_and_diff = np.sum(others_mono & ~is_same)
            
            deltas[b] = mono_and_diff - mono_and_same
            
        return deltas

    def count_mono(colors):
        c0 = colors[ap_indices[:, 0]]
        c1 = colors[ap_indices[:, 1]]
        c2 = colors[ap_indices[:, 2]]
        c3 = colors[ap_indices[:, 3]]
        mono_mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mono_mask)

    # 4. Iterated Local Search
    rng = np.random.default_rng(42)
    
    best_count = np.inf
    best_colors = None
    
    # Run multiple restarts
    while time.time() < deadline:
        # Initialize with random colors
        colors = rng.integers(0, 2, size=N, dtype=np.uint8)
        current_count = count_mono(colors)
        
        # Local Search Phase: Steepest Descent
        improved = True
        while improved and time.time() < deadline:
            improved = False
            deltas = get_deltas(colors)
            
            # Find the bit with the most negative delta (best improvement)
            min_delta = np.min(deltas)
            
            if min_delta < 0:
                # Find all bits with this min delta and pick one randomly (or first)
                candidates = np.where(deltas == min_delta)[0]
                b = candidates[0] # Pick first for determinism, or random for diversity
                
                colors[b] = 1 - colors[b]
                current_count += min_delta
                improved = True
                
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            else:
                # No improvement possible with single flip
                break
        
        # Perturbation Phase: Flip a random number of bits to escape local minimum
        # Usually 1-5% of bits
        num_flips = rng.integers(1, 6)
        flip_indices = rng.choice(N, size=num_flips, replace=False)
        colors[flip_indices] = 1 - colors[flip_indices]
        
        # Optional: If perturbation led to a better state immediately, keep it?
        # No, standard ILS usually restarts local search from the perturbed state.
        # However, if the perturbed state is better than the local minimum, 
        # the next local search will start from there.
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    if best_colors is None:
        # Fallback
        return [0] * N
        
    return best_colors.tolist()

# Quick self-test to ensure it runs and doesn't crash
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     print(res[:10])
