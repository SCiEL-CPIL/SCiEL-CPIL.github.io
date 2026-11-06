import time
import random
import sys
import math

_START_TIME = time.time()
# The evaluator kills at 150s. We should aim to finish by 145s to be safe.
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            # Only consider neighbors j > i to avoid double counting
            # Mask out bits <= i
            mask = ~((1 << (i + 1)) - 1)
            neigh = adj[c][i] & mask
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                # Count common neighbors k > j
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _sa_search(n, time_budget, seed=42):
    """
    Simulated Annealing search for a valid 4-coloring of K_n.
    Returns a list of colors if successful, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    m = n * (n - 1) // 2
    
    # Initialize colors
    # If n <= 32, we can use the exact seed.
    # If n > 32, we extend the seed.
    if n <= 32:
        colors = _seed_32()[:m]
    else:
        # Get seed for first 32 vertices
        seed_colors = _seed_32()
        colors = [0] * m
        idx = 0
        for i in range(32):
            for j in range(i + 1, 32):
                colors[idx] = seed_colors[idx]
                idx += 1
        # Randomly assign colors for remaining edges
        for i in range(32):
            for j in range(33, n):
                colors[idx] = rng.randint(0, 3)
                idx += 1
        for i in range(33):
            for j in range(i + 1, n):
                colors[idx] = rng.randint(0, 3)
                idx += 1

    adj = _build_adj(colors, n)
    current_energy = _count_mono_triangles(adj, n)
    
    if current_energy == 0:
        return colors

    best_energy = current_energy
    best_colors = colors[:]
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.001
    cooling = 0.9995 # Slow cooling to explore more
    
    while True:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random edge
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        idx = _compute_index(n, u, v)
        current_color = colors[idx]
        
        # Try changing to a random other color
        new_color = rng.randint(0, 3)
        if new_color == current_color:
            continue
            
        # Calculate Delta Energy
        # Removing edge (u,v) from current_color:
        # Triangles removed = common neighbors of u and v in current_color
        removed = (adj[current_color][u] & adj[current_color][v]).bit_count()
        
        # Adding edge (u,v) to new_color:
        # Triangles added = common neighbors of u and v in new_color
        added = (adj[new_color][u] & adj[new_color][v]).bit_count()
        
        delta = added - removed
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept move
            colors[idx] = new_color
            
            # Update adjacency
            adj[current_color][u] &= ~(1 << v)
            adj[current_color][v] &= ~(1 << u)
            adj[new_color][u] |= (1 << v)
            adj[new_color][v] |= (1 << u)
            
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors[:]
                
            if current_energy == 0:
                return colors
                
        T *= cooling
        if T < T_min:
            # Restart with higher temperature if stuck
            T = 1.0
            # Optionally re-randomize if stuck for too long? 
            # For now, just keep going.
            
    return best_colors if best_energy == 0 else None

def construct():
    try:
        # Target n=50 first
        if _time_left() > 10:
            # Use a portion of the time for n=50
            # Give it 130s if possible, but leave room for fallbacks
            budget = min(130.0, _time_left() - 2.0)
            if budget > 1.0:
                result = _sa_search(50, budget, seed=12345)
                if result is not None:
                    return {'n': 50, 'colors': result}
        
        # If n=50 failed, try n=49
        if _time_left() > 5:
            budget = min(20.0, _time_left() - 1.0)
            if budget > 0.5:
                result = _sa_search(49, budget, seed=54321)
                if result is not None:
                    return {'n': 49, 'colors': result}
                    
        # Try n=48
        if _time_left() > 2:
            budget = min(10.0, _time_left() - 0.5)
            if budget > 0.5:
                result = _sa_search(48, budget, seed=11111)
                if result is not None:
                    return {'n': 48, 'colors': result}

        # Fallback to known valid n=32
        seed_colors = _seed_32()
        return {'n': 32, 'colors': seed_colors}
        
    except Exception as e:
        # Absolute fallback
        try:
            seed_colors = _seed_32()
            return {'n': 32, 'colors': seed_colors}
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
