import time
import numpy as np

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

    def get_deltas_single(colors):
        """
        Compute the change in mono-count for flipping each single bit.
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

    def get_deltas_multi(colors, bits):
        """
        Compute the change in mono-count for flipping a specific set of bits simultaneously.
        This is done by checking each AP to see if flipping the subset of bits in that AP
        results in a monochromatic AP, comparing to the current state.
        """
        # For each AP, determine which of the 4 positions are in the 'bits' set
        # and what their current colors are.
        # An AP is affected if it contains at least one bit from 'bits'.
        
        # To make this efficient, we can precompute for each AP a bitmask of which positions
        # are in the flipped set? No, 'bits' changes every call.
        
        # Alternative: Iterate over APs.
        # For each AP, get the 4 indices.
        # Check if any index is in 'bits'.
        # If so, calculate the new colors of the 4 positions (flipping those in 'bits').
        # Check if new colors are mono. Check if old colors were mono.
        # Delta = new_mono - old_mono.
        
        # This is O(num_aps) per evaluation. With ~3000 APs, this is very fast in Python if vectorized?
        # Actually, vectorizing "is any of these 4 indices in bits" is tricky without a lookup table.
        
        # Let's use a lookup table approach:
        # Create a boolean array 'is_flipped' of size N.
        is_flipped = np.zeros(N, dtype=bool)
        is_flipped[bits] = True
        
        # Get colors of all elements in all APs
        # c[i, j] is color of element j in AP i
        c = colors[ap_indices] # Shape (num_aps, 4)
        
        # Determine which positions in each AP are flipped
        # f[i, j] is True if element j in AP i is flipped
        f = is_flipped[ap_indices] # Shape (num_aps, 4)
        
        # Compute new colors
        # If flipped, color becomes 1 - color. Else color.
        # c_new = c ^ f (since colors are 0/1 and f is bool/0-1)
        c_new = c.astype(np.int32) ^ f.astype(np.int32)
        
        # Check if AP is mono in old state
        old_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        # Check if AP is mono in new state
        new_mono = (c_new[:, 0] == c_new[:, 1]) & (c_new[:, 1] == c_new[:, 2]) & (c_new[:, 2] == c_new[:, 3])
        
        # An AP contributes to the delta only if it was affected (i.e., at least one bit flipped)
        # If no bit in the AP was flipped, old_mono == new_mono, so delta is 0.
        # We can just sum (new_mono - old_mono) for all APs.
        delta = np.sum(new_mono.astype(np.int32) - old_mono.astype(np.int32))
        
        return delta

    # 4. Iterated Local Search with Multi-bit Moves
    rng = np.random.default_rng(42)
    
    best_count = np.inf
    best_colors = None
    
    # Run multiple restarts
    while time.time() < deadline:
        # Initialize with random colors
        colors = rng.integers(0, 2, size=N, dtype=np.uint8)
        current_count = count_mono(colors)
        
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try Single Bit Flips
            deltas = get_deltas_single(colors)
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
                continue # Restart search with new state
            
            # 2. If no single bit improvement, try 2-bit flips
            # To avoid O(N^2) = 65k checks which might be slow if get_deltas_multi is expensive,
            # we can sample a subset or just do it all if fast enough.
            # get_deltas_multi is O(num_aps) ~ 3000 ops. 65k * 3000 = 195M ops. Too slow in Python loops.
            # But we can vectorize the 2-bit search?
            
            # Let's try a different approach for multi-bit:
            # Randomly sample a few 2-bit and 3-bit flips to escape.
            # Or, use the "steepest descent" on a sampled neighborhood.
            
            # Let's sample 100 random 2-bit pairs and 50 random 3-bit triples.
            best_multi_delta = 0
            best_multi_bits = None
            
            # 2-bit flips
            for _ in range(200):
                if time.time() > deadline - 5.0: break
                bits = rng.choice(N, size=2, replace=False)
                d = get_deltas_multi(colors, bits)
                if d < best_multi_delta:
                    best_multi_delta = d
                    best_multi_bits = bits
            
            # 3-bit flips
            for _ in range(50):
                if time.time() > deadline - 5.0: break
                bits = rng.choice(N, size=3, replace=False)
                d = get_deltas_multi(colors, bits)
                if d < best_multi_delta:
                    best_multi_delta = d
                    best_multi_bits = bits
            
            if best_multi_delta < 0:
                # Apply the best multi-bit flip found
                for b in best_multi_bits:
                    colors[b] = 1 - colors[b]
                current_count += best_multi_delta
                improved = True
                
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
                continue # Restart search
            
            # 3. If no improvement with sampled multi-bits, we are stuck.
            # Perturbation: Flip a random number of bits
            num_flips = rng.integers(3, 10)
            flip_indices = rng.choice(N, size=num_flips, replace=False)
            colors[flip_indices] = 1 - colors[flip_indices]
            
            # Check time
            if time.time() > deadline - 5.0:
                break

    if best_colors is None:
        return [0] * N
        
    return best_colors.tolist()

# Quick self-test
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     print(res[:10])
