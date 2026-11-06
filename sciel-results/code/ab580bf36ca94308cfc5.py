# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def _mul(a, b):
    """Multiply in GF(16) with primitive polynomial 0b10011."""
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_seed_32():
    """Greenwood-Gleason colouring of K32 (two copies of K16 with colour 3 between)."""
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index in the flat colors list for pair (i,j), i<j, n vertices."""
    return i * (i - 1) // 2 + (j - i - 1)

def _count_mono_triangles(colors, n):
    """
    Count the number of monochromatic triangles in a coloring of K_n.
    colors: flat list of length n*(n-1)//2
    n: number of vertices
    Returns: (count, triangles) where triangles is a list of (i, j, k, color)
    """
    count = 0
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    count += 1
                    triangles.append((i, j, k, cij))
    return count, triangles

def _simulate_annealing(initial_colors, n, deadline, rng):
    """
    Run Simulated Annealing to minimize monochromatic triangles.
    Returns the best coloring found (as a list) and its triangle count.
    """
    current_colors = initial_colors[:]
    current_n = n
    
    # Initial energy
    current_energy, _ = _count_mono_triangles(current_colors, current_n)
    
    best_colors = current_colors[:]
    best_energy = current_energy
    
    # SA Parameters
    # We need to cool down. The number of steps depends on time.
    # Let's estimate steps per second.
    start_time = time.time()
    elapsed = 0
    T = 10.0 # Initial temperature
    cooling_rate = 0.9995 # Depends on total steps, will adjust
    
    # We don't know total steps in advance, so we use a time-based cooling.
    # Or we can run for a fixed number of steps and see how long it takes.
    # Let's run for as many steps as possible within the deadline.
    
    step_count = 0
    last_time = time.time()
    
    # Estimate steps per second
    # A single step: pick random edge, pick new color (1-3 choices), calculate delta.
    # Delta calculation:
    # Let edge (u,v) change from c_old to c_new.
    # Triangles affected are those involving (u,v) and any w.
    # There are n-2 such triangles.
    # For each w:
    #   Old state: (u,v,c_old), (u,w,c_uw), (v,w,c_vw)
    #   New state: (u,v,c_new), (u,w,c_uw), (v,w,c_vw)
    #   Delta = 1 if new is mono, 0 if not - (1 if old is mono, 0 if not)
    
    while time.time() < deadline:
        # Pick random edge
        i = rng.randint(0, current_n - 1)
        j = rng.randint(0, current_n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        idx = _color_index(current_n, i, j)
        old_color = current_colors[idx]
        
        # Pick new color
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            continue
            
        # Calculate delta in energy
        delta = 0
        for w in range(current_n):
            if w == i or w == j:
                continue
                
            # Get colors of (i,w) and (j,w)
            # Need indices
            if w < i:
                iwi = w
                jwi = i
                idx_iw = _color_index(current_n, iwi, jwi)
            else:
                iwi = i
                jwi = w
                idx_iw = _color_index(current_n, iwi, jwi)
                
            if w < j:
                jwj = w
                vwj = j
                idx_jw = _color_index(current_n, jwj, vwj)
            else:
                jwj = j
                vwj = w
                idx_jw = _color_index(current_n, jwj, vwj)
                
            c_iw = current_colors[idx_iw]
            c_jw = current_colors[idx_jw]
            
            # Check if (i, w, j) was mono with old_color
            # Mono if c_iw == old_color and c_jw == old_color
            was_mono = 1 if (c_iw == old_color and c_jw == old_color) else 0
            is_mono = 1 if (c_iw == new_color and c_jw == new_color) else 0
            
            delta += is_mono - was_mono
            
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            current_colors[idx] = new_color
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = current_colors[:]
        else:
            # Accept with probability exp(-delta / T)
            # If T is very small, this is 0
            if T > 0.01:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_colors[idx] = new_color
                    current_energy += delta
        
        # Cooling
        # We want T to go to 0 over the duration.
        # Simple approach: T *= cooling_rate
        # But we don't know total steps.
        # Let's use a time-based cooling: T = T0 * (1 - elapsed/total_time)
        # We don't know total_time exactly, but we have deadline.
        # Let's say total_time = deadline - start_time
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        if total_time > 0:
            T = 10.0 * (1.0 - elapsed / total_time)
            if T < 0.01:
                T = 0.01
                
        step_count += 1
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_colors, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = random.Random(42)
    
    # Target n. We know 50 is possible. 51 is the target.
    # Let's try to find a valid coloring for n=51 first.
    # If we fail, try n=50.
    
    # Strategy:
    # 1. Try SA on K51.
    # 2. If SA finds 0 mono triangles, we have a valid K51.
    # 3. If not, try SA on K50.
    # 4. If not, fallback to seed K32.
    
    # We can also try to initialize with a random coloring or a heuristic one.
    # Random initialization is standard for SA.
    
    best_result = None
    best_n = 0
    
    # Try n=51
    for n_target in [51, 50]:
        if time.time() > deadline - 5: # Need 5s for verification and return
            break
            
        # Run SA
        # Initial colors: random
        num_edges = n_target * (n_target - 1) // 2
        initial_colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # We might want to run SA multiple times with different seeds if time permits
        # But SA is slow. Let's try one long run first.
        
        # Actually, let's try to be smarter.
        # If we start from a known valid K32 and extend it randomly, then run SA?
        # No, SA on full graph is better for global search.
        
        # Let's run SA for a fixed portion of the time
        sa_deadline = time.time() + (deadline - time.time()) * 0.9
        if sa_deadline > deadline:
            sa_deadline = deadline - 1
            
        best_colors, best_energy = _simulate_annealing(initial_colors, n_target, sa_deadline, rng)
        
        if best_energy == 0:
            # Found a valid coloring!
            best_result = {'n': n_target, 'colors': best_colors}
            best_n = n_target
            break # Found 51, done. Or 50 if 51 failed.
            
    # If no valid coloring found via SA, fallback to seed
    if best_result is None:
        seed_colors = _build_seed_32()
        best_result = {'n': 32, 'colors': seed_colors}
        best_n = 32
    else:
        # Verify the result just in case
        # This is crucial because SA might claim 0 but have bugs or we might have
        # misunderstood the energy function.
        n = best_n
        colors = best_result['colors']
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                cij = colors[_color_index(n, i, j)]
                for k in range(j + 1, n):
                    if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback
            seed_colors = _build_seed_32()
            best_result = {'n': 32, 'colors': seed_colors}
            best_n = 32

    return best_result
# EVOLVE-BLOCK-END
