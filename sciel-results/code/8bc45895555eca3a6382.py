import numpy as np
import random
import time

def construct():
    N = 128
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps_arr)
    
    # Precompute for each position, which APs it's in
    pos_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps_arr):
        for p in ap:
            pos_aps[p].append(i)
    
    def ap_bad(ca, cb, cc, cd):
        """Check if a 4-tuple uses < 3 distinct colors"""
        if ca == cb == cc == cd:
            return True
        # Check for exactly 2 colors
        if ca == cb:
            if cc == cd and cc != ca:
                return True
            if cc == ca and cd != ca:
                return True
            if cd == ca and cc != ca:
                return True
        if ca == cc:
            if cb == cd and cb != ca:
                return True
            if cb == ca and cd != ca:
                return True
            if cd == ca and cb != ca:
                return True
        if ca == cd:
            if cb == cc and cb != ca:
                return True
            if cb == ca and cc != ca:
                return True
            if cc == ca and cb != ca:
                return True
        if cb == cc:
            if ca == cd and ca != cb:
                return True
            if ca == cb and cd != cb:
                return True
            if cd == cb and ca != cb:
                return True
        if cb == cd:
            if ca == cc and ca != cb:
                return True
            if ca == cb and cc != cb:
                return True
            if cc == cb and ca != cb:
                return True
        if cc == cd:
            if ca == cb and ca != cc:
                return True
            if ca == cc and cb != cc:
                return True
            if cb == cc and ca != cc:
                return True
        return False
    
    def count_bad(colors):
        bad = 0
        for i in range(num_aps):
            a, b, c, d = aps_arr[i]
            if ap_bad(colors[a], colors[b], colors[c], colors[d]):
                bad += 1
        return bad
    
    # Try increasing k
    for k in range(3, 8):
        start_time = time.time()
        
        for attempt in range(40):
            if time.time() - start_time > 5.0:
                break
            
            colors = np.random.randint(0, k, size=N, dtype=np.int32)
            bad = count_bad(colors)
            
            if bad == 0:
                return colors.tolist()
            
            # Local search
            for _ in range(15000):
                if time.time() - start_time > 5.0:
                    break
                
                if bad == 0:
                    return colors.tolist()
                
                # Pick a random bad AP
                while True:
                    idx = random.randint(0, num_aps - 1)
                    a, b, c, d = aps_arr[idx]
                    if ap_bad(colors[a], colors[b], colors[c], colors[d]):
                        break
                
                # Try changing each position to each color, pick best improvement
                best_impr = 0
                best_pos = -1
                best_col = -1
                for pos in [a, b, c, d]:
                    old = colors[pos]
                    for nc in range(k):
                        if nc == old:
                            continue
                        colors[pos] = nc
                        # Recompute bad count for affected APs only
                        new_bad = 0
                        for ap_i in pos_aps[pos]:
                            aa, bb, cc, dd = aps_arr[ap_i]
                            if ap_bad(colors[aa], colors[bb], colors[cc], colors[dd]):
                                new_bad += 1
                        # We need to compare to old bad count for these APs
                        # This is tricky; simpler to just do full recompute occasionally
                        colors[pos] = old
                    # For efficiency, just do random hill climbing
                    nc = random.randint(0, k - 1)
                    old = colors[pos]
                    colors[pos] = nc
                    new_bad = count_bad(colors)
                    if new_bad < bad:
                        bad = new_bad
                        best_impr = bad
                        break
                    else:
                        colors[pos] = old
                        break
                else:
                    # No improvement found in this AP, try next
                    continue
                
                if bad == 0:
                    return colors.tolist()
        
        if time.time() - start_time > 5.0:
            break
    
    # Fallback: structured pattern
    # Try coloring by (i % 3, i // 3 % 3) -> 9 colors
    colors = [(i % 3) * 3 + (i // 3) % 3 for i in range(N)]
    if count_bad(np.array(colors, dtype=np.int32)) == 0:
        return colors
    
    # Try (i % 4) * 2 + (i // 4) % 2 -> 8 colors
    colors = [(i % 4) * 2 + (i // 4) % 2 for i in range(N)]
    if count_bad(np.array(colors, dtype=np.int32)) == 0:
        return colors
    
    # Try i % 5
    colors = [i % 5 for i in range(N)]
    if count_bad(np.array(colors, dtype=np.int32)) == 0:
        return colors
    
    # Final fallback
    return list(range(N))
