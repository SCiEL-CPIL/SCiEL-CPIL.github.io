import time
import random

def construct():
    N = 128
    # Precompute all 4-APs (0-indexed)
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    def is_bad_ap(colors, ap):
        # Returns True if the AP has fewer than 3 distinct colors
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        return len({c0, c1, c2, c3}) < 3

    def search(K, time_limit):
        start_time = time.time()
        # Initial random coloring
        coloring = [random.randint(0, K - 1) for _ in range(N)]
        
        # Precompute bad status for each AP
        is_bad = [False] * num_aps
        bad_count = 0
        for idx, ap in enumerate(aps):
            if is_bad_ap(coloring, ap):
                is_bad[idx] = True
                bad_count += 1
                
        if bad_count == 0:
            return coloring
            
        best_count = bad_count
        best_coloring = coloring[:]
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() - start_time < time_limit:
            # Pick a random position
            i = random.randint(0, N - 1)
            old_c = coloring[i]
            
            # Pick a new color
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            delta = 0
            # Calculate delta by checking only affected APs
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                was_bad = is_bad[ap_idx]
                
                # Get current colors
                c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                
                # Apply new color to the specific position in the AP
                if ap[0] == i:
                    c0 = new_c
                elif ap[1] == i:
                    c1 = new_c
                elif ap[2] == i:
                    c2 = new_c
                elif ap[3] == i:
                    c3 = new_c
                    
                new_distinct = len({c0, c1, c2, c3})
                is_now_bad = new_distinct < 3
                
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Simulated Annealing probability
                # Use a simple Boltzmann distribution
                if random.random() < (1.0 / (1.0 + delta * T)):
                    accept = True
            
            if accept:
                coloring[i] = new_c
                # Update is_bad and bad_count
                for ap_idx in pos_to_aps[i]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    is_bad[ap_idx] = len({c0, c1, c2, c3}) < 3
                bad_count += delta
                
                if bad_count < best_count:
                    best_count = bad_count
                    best_coloring = coloring[:]
                    if best_count == 0:
                        return best_coloring
            
            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T

        return best_coloring if best_count == 0 else None

    # Try 3 colors first
    result = search(3, 25.0)
    if result is not None:
        # Verify it's valid
        valid = True
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                valid = False
                break
        if valid:
            return result

    # Try 4 colors
    result = search(4, 25.0)
    if result is not None:
        valid = True
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                valid = False
                break
        if valid:
            return result

    # Try 5 colors
    result = search(5, 10.0)
    if result is not None:
        valid = True
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                valid = False
                break
        if valid:
            return result

    # Fallback: Identity coloring (always valid, uses 128 colors)
    return list(range(N))
