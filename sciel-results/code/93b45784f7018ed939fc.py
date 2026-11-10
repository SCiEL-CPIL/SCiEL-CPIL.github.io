import random
import time
import math

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """For each position, store list of AP indices it belongs to."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
    return pos_to_aps

def _thue_morse(n):
    """Generate Thue-Morse coloring for 0..n-1."""
    colors = [0] * n
    for i in range(1, n):
        # Thue-Morse: t(n) = t(n//2) XOR (n%2)
        # Or simply: count of 1s in binary representation mod 2
        colors[i] = bin(i).count('1') % 2
    return colors

def _score(colors):
    """Count monochromatic 4-APs."""
    count = 0
    # This is O(N^2) roughly, but N=256 is small.
    # We will use the precomputed aps list in the main loop for speed.
    return count

def _local_search(aps, pos_to_aps, colors, max_passes=100, deadline=None):
    """
    Fast greedy local search.
    """
    colors = list(colors)
    n = len(colors)
    
    # Precompute mono status for all APs
    # mono_flags[idx] = 1 if mono, 0 otherwise
    mono_flags = [0] * len(aps)
    current_score = 0
    for idx, (a, b, c, d) in enumerate(aps):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            mono_flags[idx] = 1
            current_score += 1
            
    best_score = current_score
    best_colors = colors[:]
    
    for _pass in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Randomize order to avoid cycles
        order = list(range(n))
        random.shuffle(order)
        
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new
            
            delta = 0
            for ap_idx in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                # Check if now mono
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono
            
            if delta < 0:
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                # Revert
                colors[i] = old
                for ap_idx in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        
        if not improved:
            break
    
    return best_colors, best_score

def _simulated_annealing(aps, pos_to_aps, colors, start_temp=10.0, end_temp=0.01, max_iter=20000, deadline=None):
    """
    Simulated Annealing with 1-bit flips.
    """
    colors = list(colors)
    n = len(colors)
    
    mono_flags = [0] * len(aps)
    current_score = 0
    for idx, (a, b, c, d) in enumerate(aps):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            mono_flags[idx] = 1
            current_score += 1
            
    best_score = current_score
    best_colors = colors[:]
    
    temp = start_temp
    # Use geometric cooling
    cooling = (end_temp / start_temp) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        temp *= cooling
        
        i = random.randint(0, n - 1)
        old = colors[i]
        new = 1 - old
        colors[i] = new
        
        delta = 0
        for ap_idx in pos_to_aps[i]:
            a, b, c, d = aps[ap_idx]
            was_mono = mono_flags[ap_idx]
            now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
            if was_mono and not now_mono:
                delta -= 1
            elif not was_mono and now_mono:
                delta += 1
            mono_flags[ap_idx] = now_mono
        
        new_score = current_score + delta
        if new_score < current_score or random.random() < math.exp(-max(delta, 0) / max(temp, 1e-10)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
        else:
            colors[i] = old
            for ap_idx in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0

    return best_colors, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    random.seed(42)
    
    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)
    
    best_score = float('inf')
    best_pop = None
    
    # Initial diverse set
    # 1. Thue-Morse
    tm = _thue_morse(N)
    # 2. Random
    rand1 = [random.randint(0, 1) for _ in range(N)]
    # 3. Block 8
    block8 = [(i // 8) % 2 for i in range(N)]
    # 4. Block 16
    block16 = [(i // 16) % 2 for i in range(N)]
    # 5. Thue-Morse shifted
    tm_shift = [tm[(i + 1) % N] for i in range(N)]
    
    candidates = [tm, rand1, block8, block16, tm_shift]
    
    # Phase 1: Local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 60:
            break
        ls_pop, s = _local_search(aps, pos_to_aps, cand, max_passes=200, deadline=deadline - 60)
        if s < best_score:
            best_score = s
            best_pop = ls_pop[:]
            
    # Phase 2: Simulated Annealing from best solution
    # Run multiple SA restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Start from current best
        start_pop = best_pop[:]
        
        # Add some noise if not the first restart
        if restart > 0:
            # Flip a few random bits
            for _ in range(10):
                idx = random.randint(0, N - 1)
                start_pop[idx] = 1 - start_pop[idx]
                
        sa_pop, s = _simulated_annealing(aps, pos_to_aps, start_pop, 
                                         start_temp=5.0, end_temp=0.01, 
                                         max_iter=5000, 
                                         deadline=deadline - 10)
        if s < best_score:
            best_score = s
            best_pop = sa_pop[:]
            
        # Polish with local search
        if time.time() < deadline - 5:
            ls_pop, s = _local_search(aps, pos_to_aps, best_pop, max_passes=100, deadline=deadline - 5)
            if s < best_score:
                best_score = s
                best_pop = ls_pop[:]
                
    return best_pop
