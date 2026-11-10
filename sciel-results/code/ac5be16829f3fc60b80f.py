import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    # The evaluator kills after 150s. Use a safe margin.
    deadline = start_time + 145.0

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    # d can go up to (N-1)//3 = 85
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    # ap_list[i] is a list of (ap_id, other_indices_tuple)
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta(colors, idx):
        """
        Calculate change in mono count if we flip colors[idx].
        """
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def lfsr_sequence(taps, initial_state, length):
        """
        Generate a bit sequence using LFSR.
        taps: list of positions (1-indexed from LSB) to XOR for feedback.
        initial_state: integer representing the initial register state.
        """
        state = initial_state
        seq = []
        for _ in range(length):
            # Extract LSB
            bit = state & 1
            seq.append(bit)
            # Feedback
            feedback = 0
            for tap in taps:
                feedback ^= (state >> (tap - 1)) & 1
            # Shift right and add feedback to MSB
            state = (state >> 1) | (feedback << (len(taps) - 1))
        return seq

    def local_search(colors, max_time):
        """
        Deterministic hill climbing with random restarts within the local search phase.
        Returns best colors and count found.
        """
        best_count = count_mono(colors)
        best_colors = colors[:]
        
        while time.time() < max_time:
            # Random restart
            current_colors = [random.randint(0, 1) for _ in range(N)]
            current_count = count_mono(current_colors)
            
            improved = True
            while improved:
                improved = False
                # Try flipping each bit, pick the best improvement
                best_delta = 0
                best_idx = -1
                
                for idx in range(N):
                    d = get_delta(current_colors, idx)
                    if d < best_delta:
                        best_delta = d
                        best_idx = idx
                
                if best_idx != -1 and best_delta < 0:
                    current_colors[best_idx] = 1 - current_colors[best_idx]
                    current_count += best_delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = current_colors[:]
            
            # Check time periodically
            if time.time() > max_time - 1.0:
                break

        return best_colors, best_count

    # Strategy:
    # 1. Try a few LFSR-based initializations which might have better global structure.
    # 2. Run local search from these and random starts.
    
    best_global_count = float('inf')
    best_global_colors = None

    # Some known primitive polynomials for LFSR (x^n + ... + 1)
    # For simplicity, we'll use small taps that generate good pseudo-randomness.
    # x^8 + x^4 + x^3 + x^2 + 1 (taps 8,4,3,2)
    # x^16 + x^14 + x^13 + x^11 + 1 (taps 16,14,13,11)
    
    lfsr_configs = [
        ([8, 4, 3, 2, 1], 0b10110110), # x^8+x^4+x^3+x^2+1, init 0xB6
        ([16, 14, 13, 11, 1], 0b1001110011100101), # x^16+x^14+x^13+x^11+1
        ([12, 11, 10, 9, 1], 0b101100110011), # x^12+x^11+x^10+x^9+1
    ]
    
    # Phase 1: LFSR starts
    for taps, init in lfsr_configs:
        if time.time() > deadline - 2.0:
            break
        
        # Generate 256 bits
        seq = lfsr_sequence(taps, init, N)
        colors = seq[:]
        
        # Run local search with a time slice
        # Allocate ~10s per start? No, let's do dynamic.
        # We have ~140s total.
        # Let's do 5 LFSR starts (including random variations of init) and then fill the rest with random.
        
        # Actually, let's just run local search on the LFSR sequence.
        # To maximize speed, we can run a short local search here.
        # But local_search as defined does random restarts internally. 
        # We want to specifically optimize the LFSR seed first.
        
        # Let's modify local_search to accept a specific start or just run it.
        # If we pass the LFSR sequence, the first iteration of the while loop in local_search
        # will do a random restart, ignoring the LFSR sequence.
        # So we need a version that starts from the given colors.
        
        def local_search_from(colors, max_time):
            best_count = count_mono(colors)
            best_colors = colors[:]
            
            current_colors = colors[:]
            current_count = best_count
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                best_delta = 0
                best_idx = -1
                
                for idx in range(N):
                    d = get_delta(current_colors, idx)
                    if d < best_delta:
                        best_delta = d
                        best_idx = idx
                
                if best_idx != -1 and best_delta < 0:
                    current_colors[best_idx] = 1 - current_colors[best_idx]
                    current_count += best_delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = current_colors[:]
            
            return best_colors, best_count

        # Run local search from LFSR start with 20s budget
        lc_colors, lc_count = local_search_from(colors, time.time() + 20.0)
        
        if lc_count < best_global_count:
            best_global_count = lc_count
            best_global_colors = lc_colors

    # Phase 2: Random starts with local search
    # Use remaining time for random restarts
    while time.time() < deadline - 2.0:
        random_colors = [random.randint(0, 1) for _ in range(N)]
        # Allocate 10s per random start
        lc_colors, lc_count = local_search_from(random_colors, time.time() + 10.0)
        
        if lc_count < best_global_count:
            best_global_count = lc_count
            best_global_colors = lc_colors

    if best_global_colors is None:
        # Fallback
        best_global_colors = [0] * N

    return best_global_colors
