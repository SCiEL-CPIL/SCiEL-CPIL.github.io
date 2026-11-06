import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 128
    
    # Precompute all 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0 and a+3d <= N
    # We use 0-based indices internally for speed, but logic is 1-based.
    # Positions 0..127 correspond to 1..128.
    aps = []
    for a in range(0, N):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end < N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map index to list of AP indices it belongs to
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)

    def is_valid(colors):
        """Check if the coloring is valid."""
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Strategy:
    # 1. Try to find a valid coloring with k=3 colors using a structured heuristic.
    # 2. If that fails or is too slow, try k=4, etc.
    # 3. Use Local Search to minimize colors for a fixed k? No, the constraint is "at least 3 colors in every AP".
    #    The objective is to MINIMIZE the number of distinct labels used.
    #    So we want to find a valid coloring using the fewest distinct colors.
    #    Let's try k=3 first. If a valid coloring with 3 colors exists, that's optimal (since we need at least 3 colors in some AP, 
    #    wait, do we? The condition is "Every nonconstant four-term progression ... must use AT LEAST three distinct colors".
    #    This implies we cannot have a valid coloring with only 1 or 2 colors total, because if we only used 2 colors globally,
    #    any AP would use at most 2 colors, violating the condition.
    #    So the minimum possible number of colors is 3.
    #    If we can find a valid coloring with 3 colors, we are done (score 1.0 relative to seed if seed uses 3? 
    #    Wait, the seed program uses a greedy approach that might use more. 
    #    The task says "Minimize the number of labels actually used".
    #    If 3 is the theoretical minimum, finding a 3-coloring is the best possible result.
    
    # Let's try to construct a 3-coloring.
    # A common strategy for avoiding monochromatic or 2-color APs is to use modular arithmetic.
    # For 3 colors, let's try c[i] = i % 3.
    # Let's test if c[i] = i % 3 works.
    # AP: a, a+d, a+2d, a+3d.
    # Colors: a%3, (a+d)%3, (a+2d)%3, (a+3d)%3 = a%3.
    # The set of colors is {a%3, (a+d)%3, (a+2d)%3}.
    # If d is a multiple of 3, then a%3 = (a+d)%3 = (a+2d)%3 = (a+3d)%3. All same color. Invalid.
    # So i%3 does NOT work.
    
    # We need a coloring where no AP has <= 2 distinct colors.
    # This is equivalent to saying that for any AP, the colors are not all in a set of size < 3.
    
    # Let's use a randomized local search to find a valid coloring with k=3.
    # If k=3 is impossible, we try k=4, etc.
    
    def local_search(k, max_iters=50000):
        """
        Try to find a valid coloring using colors 0..k-1.
        Uses simulated annealing / hill climbing with restarts.
        """
        best_colors = None
        best_score = float('inf')
        
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            colors = [random.randint(0, k-1) for _ in range(N)]
            
            # Calculate initial violations
            # Violation for an AP is 1 if len(set(colors)) < 3, else 0.
            # We want to minimize total violations.
            
            def get_violations(cols):
                viol = 0
                for ap in aps:
                    if len({cols[ap[0]], cols[ap[1]], cols[ap[2]], cols[ap[3]]}) < 3:
                        viol += 1
                return viol
            
            cur_viol = get_violations(colors)
            
            if cur_viol == 0:
                return colors
            
            temp = 1.0
            cooling = 0.9995
            
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                
                if cur_viol == 0:
                    return colors
                    
                # Pick a random position and change its color
                pos = random.randint(0, N-1)
                old_color = colors[pos]
                new_color = random.randint(0, k-1)
                
                if new_color == old_color:
                    continue
                
                # Calculate change in violations
                # Only APs containing pos are affected
                delta_viol = 0
                for ap_idx in ap_indices[pos]:
                    ap = aps[ap_idx]
                    # Before
                    set_before = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    viol_before = 1 if len(set_before) < 3 else 0
                    
                    # After
                    colors[pos] = new_color
                    set_after = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    viol_after = 1 if len(set_after) < 3 else 0
                    
                    delta_viol += viol_after - viol_before
                
                new_viol = cur_viol + delta_viol
                
                if new_viol < cur_viol:
                    cur_viol = new_viol
                    if cur_viol == 0:
                        return colors
                else:
                    # Accept with probability
                    if random.random() < pow(temp, new_viol - cur_viol):
                        cur_viol = new_viol
                    else:
                        colors[pos] = old_color
                
                temp *= cooling
                
                if temp < 0.01:
                    break
                    
        return None

    # Try k=3
    for k in range(3, 10):
        if time.time() > deadline:
            break
            
        # Try a few structured initializations before random
        # 1. i % k
        # 2. (i // k) % k
        # 3. Random
        
        found = False
        for init_type in range(3):
            if time.time() > deadline:
                break
                
            if init_type == 0:
                colors = [i % k for i in range(N)]
            elif init_type == 1:
                colors = [(i // k) % k for i in range(N)]
            else:
                colors = [random.randint(0, k-1) for _ in range(N)]
                
            # Check if valid
            if is_valid(colors):
                return colors
                
            # Local search from this start
            res = local_search(k, max_iters=10000)
            if res is not None:
                return res
                
    # Fallback: Greedy construction (slower but guaranteed to find *some* valid coloring, though possibly with many colors)
    # This is the seed's approach, but we'll implement it faster if possible.
    # Actually, if local search fails for k=3..9, it's unlikely to succeed quickly for higher k with random restarts.
    # Let's use a greedy algorithm that picks the smallest color that doesn't create a violation with previous positions?
    # No, greedy forward assignment is tricky because future positions matter.
    
    # Let's try a simple greedy: assign colors one by one, ensuring that for any AP ending at the current position,
    # the condition is met? No, APs can extend further.
    
    # A valid coloring is guaranteed to exist (e.g., all distinct colors).
    # We must return a valid coloring.
    # If we haven't found one, we use a "safe" construction:
    # c[i] = i. This uses 128 colors. It is valid because any 4 distinct positions have 4 distinct colors.
    # But we want to minimize colors.
    
    # Let's try one more thing: Random restarts with k=3 for longer if time permits?
    # Or just return a valid coloring with minimal colors found so far.
    
    # Since local_search returns None if it doesn't find a valid one, we need a fallback valid coloring.
    # The most efficient fallback is to use a known valid pattern or just c[i]=i.
    # c[i]=i is valid.
    # But can we do better?
    # Let's try k=4 with more iterations if k=3 failed.
    
    # Actually, the local search above is quite fast. If it fails for k=3, it might fail for k=4.
    # Let's assume k=3 is possible and just keep trying.
    
    # If we run out of time, we must return a valid coloring.
    # The safest valid coloring is c[i] = i.
    # But we can try to compress it? No, just return c[i]=i to ensure validity.
    # Wait, the seed program's fallback is a greedy algorithm that produces a valid coloring.
    # Let's replicate the seed's fallback to ensure we don't score 0.
    
    # Seed fallback logic:
    # colors=[]
    # for x in range(1,129):
    #     for color in range(128):
    #         if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
    #             and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
    #             colors.append(color);break
    
    # This greedy logic is O(N^3) or so, which is fine for N=128.
    # It ensures that when we add color `color` at position `x`, all APs ending at `x` are valid?
    # Let's check the condition:
    # 1. `len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3`
    #    This checks APs (x-3d, x-2d, x-d, x). The set of colors in this AP must have size >= 3.
    # 2. `any(colors[x-1-j*d]!=color for j in (1,2))` for d in range(1, (x-1)//2 + 1)
    #    This checks APs (x-2d, x-d, x) ? No, 3-term APs?
    #    The task is about 4-term APs.
    #    The seed's second condition seems to be checking something else, maybe 3-term APs to prevent 2-color patterns?
    #    Actually, if every 4-term AP has >= 3 colors, does it imply something about 3-term APs?
    #    Not necessarily.
    #    
    #    Let's just use the simple valid coloring c[i] = i if local search fails.
    #    c[i] = i uses 128 colors.
    #    The seed likely uses fewer.
    #    But c[i]=i is definitely valid.
    
    # Let's try to run the seed's fallback to get a better score than 128 colors.
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check all APs ending at x-1 (0-based)
            # AP: (x-1-3d, x-1-2d, x-1-d, x-1)
            # Indices: i0 = x-1-3d, i1 = x-1-2d, i2 = x-1-d, i3 = x-1
            # We are assigning color to i3.
            # The colors of i0, i1, i2 are already set.
            # We need len({color, c[i0], c[i1], c[i2]}) >= 3
            
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                i0 = x - 1 - 3 * d
                i1 = x - 1 - 2 * d
                i2 = x - 1 - d
                # i3 is current position
                existing = {colors[i0], colors[i1], colors[i2]}
                if len({color} | existing) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            # If no color works (should not happen), use a large number
            colors.append(127)
            
    return colors

# EVOLVE-BLOCK-END
