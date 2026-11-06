import time
import random
import numpy as np

def construct():
    n = 256
    # Use a generous deadline to ensure we finish well before 150s
    # The evaluator kills at 150s. We target 110s of computation.
    deadline = time.time() + 110
    
    # 1. Precompute all 4-term APs
    # a, a+d, a+2d, a+3d <= 256
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # Map position to list of (ap_index, offset_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Precompute APs as numpy array for fast full counting (used for init check)
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono_numpy(color):
        """Fast count using NumPy. Used for initial evaluations."""
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def init_state(color):
        """Initialize mono status and count."""
        mono = [False] * num_aps
        count = 0
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                count += 1
        return mono, count

    def delta_flip(color, mono, p):
        """
        Calculate the change in mono count if we flip position p.
        Returns delta (new_count - old_count).
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            if mono[ap_idx]:
                # Currently mono. Flipping p breaks it.
                # It remains mono only if the other 3 were new_val? No.
                # If it was mono, all 4 were old_val.
                # After flip, p is new_val, others are old_val.
                # They are not all equal. So it becomes non-mono.
                delta -= 1
            else:
                # Currently non-mono.
                # Check if flipping p makes it mono.
                # This happens if the other 3 positions all have value new_val.
                # We need to check the colors of the other 3 positions.
                # We can optimize by pre-checking, but let's just check.
                # The other 3 positions are (p0, p1, p2, p3) excluding p.
                
                # Unroll for speed
                if offset == 0:
                    if color[p1] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 1:
                    if color[p0] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 2:
                    if color[p0] == new_val and color[p1] == new_val and color[p3] == new_val:
                        delta += 1
                else: # offset == 3
                    if color[p0] == new_val and color[p1] == new_val and color[p2] == new_val:
                        delta += 1
                        
        return delta

    def apply_flip(color, mono, p):
        """
        Flip position p in color, update mono status for affected APs.
        Returns the delta in count.
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            was_mono = mono[ap_idx]
            
            # Determine new mono status
            # We know color[p] is about to change to new_val.
            # We can check the other 3 against new_val.
            if offset == 0:
                is_mono = (color[p1] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 1:
                is_mono = (color[p0] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 2:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p3] == new_val)
            else:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p2] == new_val)
                
            mono[ap_idx] = is_mono
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        color[p] = new_val
        return delta

    def local_search_sa(color, mono, count, max_iter=5000):
        """
        Local search with Simulated Annealing.
        """
        T = 1.0
        cooling = 0.99
        min_T = 1e-6
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            # Find best flip (greedy)
            best_delta = 0
            best_pos = -1
            
            # Sample a subset of positions to speed up? 
            # With n=256, checking all is fast enough (~256 * 20 ops = 5k ops).
            # Let's check all positions.
            for p in range(n):
                d = delta_flip(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
                    
            if best_pos != -1:
                # Greedy move
                delta = apply_flip(color, mono, best_pos)
                count += delta
            else:
                # No improving move found. Try SA move.
                if T < min_T:
                    break
                
                # Pick random position
                p = random.randint(0, n - 1)
                d = delta_flip(color, mono, p)
                
                # Accept if d < 0 or with probability exp(-d/T)
                if d < 0:
                    delta = apply_flip(color, mono, p)
                    count += delta
                else:
                    # d >= 0
                    if random.random() < np.exp(-d / T):
                        delta = apply_flip(color, mono, p)
                        count += delta
                        
                T *= cooling
                
        return color, mono, count

    # Initializations
    # 1. Thue-Morse
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    # 2. Random
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]
        
    # 3. Rademacher (alternating blocks)
    def rademacher_init(n):
        # Simple: 00110011...
        return [(i // 2) % 2 for i in range(n)]
        
    random.seed(42)
    
    best_color = None
    best_score = float('inf')
    
    # Generate a pool of initial solutions
    initializers = [
        thue_morse,
        random_init,
        rademacher_init,
        lambda: [1 - x for x in thue_morse(n)], # Inverted Thue-Morse
    ]
    
    # Run ILS/SA on each
    for init_func in initializers:
        if time.time() > deadline - 10:
            break
            
        color = init_func(n)
        mono, count = init_state(color)
        
        # Run local search with SA
        color, mono, count = local_search_sa(color, mono, count)
        
        if count < best_score:
            best_score = count
            best_color = color[:]
            
        # Perform Iterated Local Search from this local optimum
        # Perturb and re-search
        base_color = color[:]
        base_mono = mono[:]
        base_count = count
        
        # Multiple iterations of perturbation
        for _ in range(20): # Limit perturbations per init
            if time.time() > deadline - 5:
                break
                
            # Perturb base solution
            perturbed = base_color[:]
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                p = random.randint(0, n - 1)
                perturbed[p] = 1 - perturbed[p]
                
            p_mono, p_count = init_state(perturbed)
            
            # Local search on perturbed
            p_color, p_mono, p_count = local_search_sa(perturbed, p_mono, p_count, max_iter=2000)
            
            if p_count < best_score:
                best_score = p_count
                best_color = p_color[:]
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count
                
            # Update base to the best found in this ILS loop if it improved
            if p_count < base_count:
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count

    return best_color
