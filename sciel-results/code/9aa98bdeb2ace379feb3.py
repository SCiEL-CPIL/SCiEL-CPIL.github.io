import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a <= 256
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    # ap_list[i] is a list of (ap_id, other_indices_tuple)
    # This allows fast delta calculation: flipping i changes the monochromatic status 
    # of each AP containing i.
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
        If colors[idx] is 0, flipping to 1:
        - APs that were mono-0 and remain mono-0 (others all 0): no change? No, they become mixed.
          Wait. If others are all 0, and current is 0, AP is mono. If we flip to 1, AP is NOT mono. Delta -1.
        - If others are all 1, and current is 0, AP is NOT mono. If we flip to 1, AP IS mono. Delta +1.
        - If others are mixed, AP is NOT mono. If we flip, AP is still NOT mono. Delta 0.
        
        So for each AP containing idx:
        If all others have color C:
           If current color is C, delta -= 1 (was mono, becomes non-mono)
           If current color is 1-C, delta += 1 (was non-mono, becomes mono)
        Else:
           delta += 0
        """
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            # Check if others are all same color
            # others is a tuple of 3 indices
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                # Others are all o0
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Initialize with random coloring
    colors = [random.randint(0, 1) for _ in range(N)]
    current_count = count_mono(colors)
    best_count = current_count
    best_colors = colors[:]

    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We can also try a few random restarts if time permits, but SA usually converges well.
    # Let's stick to one long run with periodic restarts if stuck, or just long SA.
    # Given N=256 and ~2000 APs, delta calculation is O(1) per flip (approx, since each idx is in few APs).
    # Max APs per idx: d=1 (4 APs), d=2 (3 APs)... roughly < 10 APs per index.
    # So 1 flip is very fast. We can do millions of steps.

    steps = 0
    while time.time() < deadline:
        # Pick a random index to flip
        idx = random.randint(0, N - 1)
        
        delta = get_delta(colors, idx)
        new_count = current_count + delta
        
        if new_count < current_count:
            # Always accept improvement
            colors[idx] = 1 - colors[idx]
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]
        else:
            # Accept worse with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                colors[idx] = 1 - colors[idx]
                current_count = new_count
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            # Restart with randomization or lower T further
            T = min_T
            
        steps += 1
        # Check time more frequently to avoid overshooting
        if steps % 1000 == 0:
            if time.time() > deadline - 2.0:
                break

    return best_colors

import math
