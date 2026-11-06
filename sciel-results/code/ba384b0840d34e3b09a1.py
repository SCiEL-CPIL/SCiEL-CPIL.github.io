"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Base: Greenwood-Gleason K16 (Clebsch graph x3) doubled to K32.
    Uses Simulated Annealing to extend the graph.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
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

    # Base K16: colour of edge (i,j) = log[i^j] % 3 for i!=j, i,j in 0..15
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    # Precompute adjacency for K32
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = get_color_32(i, j)
            adj[i][j] = c
            adj[j][i] = c

    def try_extend_sa(adj, n, time_limit):
        """
        Attempt to extend the graph by 1 vertex using Simulated Annealing.
        Returns (new_n, new_adj) if successful, else None.
        """
        t_start = time.time()
        current_adj = [row[:] for row in adj]
        current_n = n
        
        # We want to find a color_map of length current_n
        # such that no triangle (i, j, new) is monochromatic.
        
        # Energy function: count of monochromatic triangles involving the new vertex.
        def calc_energy(color_map):
            energy = 0
            # For each pair (i, j) in the existing graph
            # If color_map[i] == color_map[j] == c and adj[i][j] == c, then it's a conflict.
            # We can optimize this by iterating over colors.
            # But simple O(n^2) is fine for n~32.
            for i in range(current_n):
                ci = color_map[i]
                for j in range(i + 1, current_n):
                    if color_map[j] == ci and current_adj[i][j] == ci:
                        energy += 1
            return energy

        best_map = None
        best_energy = float('inf')
        
        # Simulated Annealing
        T = 1000.0
        cooling = 0.999
        steps = 0
        max_steps = 1000000 # Hard cap on steps, but we check time
        
        # Initialize randomly
        color_map = [random.randint(0, 3) for _ in range(current_n)]
        current_energy = calc_energy(color_map)
        
        if current_energy == 0:
            best_map = color_map[:]
            best_energy = 0
        else:
            best_map = color_map[:]
            best_energy = current_energy

        while time.time() - t_start < time_limit:
            if current_energy == 0:
                break
            
            # Pick a random index to flip
            idx = random.randint(0, current_n - 1)
            old_c = color_map[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            color_map[idx] = new_c
            new_energy = calc_energy(color_map)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_map = color_map[:]
                    if best_energy == 0:
                        break
            else:
                color_map[idx] = old_c
            
            T *= cooling
            steps += 1
            if steps > 500000: # Prevent infinite loop if cooling is too slow
                break

        if best_energy == 0:
            # Success! Add the vertex
            new_n = current_n + 1
            new_row = best_map[:] + [0]
            for i in range(current_n):
                current_adj[i].append(best_map[i])
            current_adj.append(new_row)
            return new_n, current_adj
        else:
            return None

    # Attempt to extend K32
    current_n = 32
    current_adj = adj
    extended = False
    
    # Try to extend as far as possible
    # We give each extension attempt a portion of the time
    # But we can also just run until time runs out
    # Since extension is hard, we try to add 1 vertex. If successful, try again.
    
    while time.time() < deadline - 10.0:
        # Allocate time for this extension attempt
        # If we have lots of time, give it more.
        # Let's just use the remaining time, but break if we succeed.
        # Actually, if we fail, we should restart with a fresh SA run?
        # The SA inside try_extend_sa runs until time_limit.
        
        # Let's try to extend by 1.
        # Give it up to 10 seconds? Or all remaining?
        # If it fails, we might be stuck.
        # Let's give it 5 seconds per attempt.
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
            
        attempt_limit = min(5.0, remaining - 1.0)
        
        result = try_extend_sa(current_adj, current_n, attempt_limit)
        if result:
            current_n, current_adj = result
            extended = True
            # Continue trying to extend
        else:
            # If we fail, we might try a few more times with fresh random seeds?
            # The SA inside uses random initialization.
            # If we fail 5 times in a row, we might give up.
            # But for now, let's just break to avoid wasting time if it's truly hard.
            # Or we can try a few more times.
            # Let's try 3 times.
            success = False
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                attempt_limit = min(5.0, deadline - time.time() - 1.0)
                if attempt_limit < 1.0:
                    break
                result = try_extend_sa(current_adj, current_n, attempt_limit)
                if result:
                    current_n, current_adj = result
                    success = True
                    break
            if not success:
                break

    # Build final result
    colors = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            colors.append(current_adj[i][j])
    return {'n': current_n, 'colors': colors}

# EVOLVE-BLOCK-END
