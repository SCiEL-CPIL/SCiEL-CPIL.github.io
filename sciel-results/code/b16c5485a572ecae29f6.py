import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (aim for 115s to be safe)

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
    # ap_list[i] is a list of ap_ids
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            ap_list[idx].append(ap_id)

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta_single_flip(colors, idx):
        """
        Calculate change in mono count if we flip colors[idx].
        Returns the delta (new_count - old_count).
        """
        delta = 0
        c = colors[idx]
        for ap_id in ap_list[idx]:
            ap = aps[ap_id]
            # Get the other 3 indices
            others = [j for j in ap if j != idx]
            o0 = colors[others[0]]
            # Check if the other 3 are monochromatic
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    # Currently mono, flipping breaks it -> delta -= 1
                    delta -= 1
                else:
                    # Currently not mono (because c != o0), flipping makes it mono -> delta += 1
                    delta += 1
        return delta

    def local_search(colors, current_count):
        """
        Deterministic steepest descent local search.
        Repeatedly finds the flip that gives the largest negative delta (improvement).
        Stops when no improvement is possible.
        """
        improved = True
        while improved:
            improved = False
            best_idx = -1
            best_delta = 0  # We want the most negative delta (largest reduction)
            
            # Evaluate all possible single flips
            for idx in range(N):
                d = get_delta_single_flip(colors, idx)
                if d < best_delta:
                    best_delta = d
                    best_idx = idx
            
            if best_idx != -1:
                # Apply the best flip
                colors[best_idx] = 1 - colors[best_idx]
                current_count += best_delta
                improved = True
                
                # If we hit 0, we can stop immediately
                if current_count == 0:
                    break
                    
        return colors, current_count

    best_count = float('inf')
    best_colors = [0] * N
    
    # Number of restarts to attempt
    # Each local search run takes variable time. 
    # Let's do as many as possible.
    restarts = 0
    
    while time.time() < deadline:
        restarts += 1
        
        # Initialize random coloring
        # For the very first run, maybe try a specific pattern? No, random is fine.
        # However, a "balanced" random start (128 zeros, 128 ones) might be better than pure random?
        # Pure random is 50/50 expectation. Let's stick to pure random for simplicity and diversity.
        colors = [random.randint(0, 1) for _ in range(N)]
        current_count = count_mono(colors)
        
        # Run local search
        colors, current_count = local_search(colors, current_count)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
            # If we found a very good solution, we might want to keep searching
            # but maybe reduce the perturbation strength? 
            # For now, just keep going.
            
            # If best_count is extremely low (e.g., < 5), we can break early to save time?
            # The task says "Minimize the exact number". 0 is the theoretical min.
            # If we hit 0, we can return immediately.
            if best_count == 0:
                break

        # Perturbation: Escape local minimum
        # Strategy: Flip 2 random distinct bits.
        # This is a "kick" to jump to a different basin of attraction.
        # We do this before the next restart loop iteration (which re-initializes? No, 
        # the current code re-initializes at the start of the while loop).
        
        # Wait, the current structure re-initializes at the start of the loop.
        # That means every iteration is a fresh random start.
        # This is "Multi-Restart Local Search".
        # This is a valid strategy. 
        # Is it better than SA? 
        # SA can move through bad states. Multi-restart only starts from random and goes down.
        # If the landscape is rugged, multi-restart might get stuck in many shallow minima.
        # However, it is much faster per "cycle".
        
        # Let's add a bit of "intelligent" restart.
        # Instead of pure random, maybe start from the previous best solution and perturb it?
        # No, that's hill-climbing with restarts (Tabu-like).
        # Pure random restarts explore the space uniformly.
        
        # Let's check time. If we are close to deadline, stop.
        if time.time() > deadline - 2.0:
            break

    return best_colors
