import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for j in range(4):
            pos = aps[idx][j]
            pos_aps[pos].append((idx, j))
    
    # Precompute for fast vectorized delta computation
    pos_ap_indices = []
    pos_ap_slots = []
    for pos in range(n):
        idxs = [ap_idx for ap_idx, _ in pos_aps[pos]]
        slots = [slot for _, slot in pos_aps[pos]]
        pos_ap_indices.append(np.array(idxs, dtype=np.int32))
        pos_ap_slots.append(np.array(slots, dtype=np.int32))
    
    def count_mono_np(color_arr):
        """Count monochromatic 4-APs using numpy."""
        c = color_arr[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def fast_delta_flip(color_arr, pos):
        """
        Vectorized delta computation for flipping position pos.
        Returns the change in mono count.
        """
        ap_idxs = pos_ap_indices[pos]
        slots = pos_ap_slots[pos]
        if len(ap_idxs) == 0:
            return 0
        
        sub_aps = aps_arr[ap_idxs]
        sub_slots = slots
        
        cur_colors = color_arr[sub_aps]
        old_c = color_arr[pos]
        new_c = 1 - old_c
        
        was_mono = np.zeros(len(ap_idxs), dtype=bool)
        will_mono = np.zeros(len(ap_idxs), dtype=bool)
        
        for slot in range(4):
            mask = (sub_slots == slot)
            if not np.any(mask):
                continue
            rows = np.where(mask)[0]
            other_slots = [s for s in range(4) if s != slot]
            other_colors = cur_colors[rows][:, other_slots]
            
            was_mono[rows] = np.all(other_colors == old_c, axis=1)
            will_mono[rows] = np.all(other_colors == new_c, axis=1)
        
        return int(np.sum(will_mono) - np.sum(was_mono))
    
    def fast_delta_swap(color_arr, pos1, pos2):
        """
        Delta computation for swapping colors at pos1 and pos2.
        Returns the change in mono count.
        """
        if pos1 == pos2:
            return 0
        
        # Get all APs containing either pos1 or pos2
        set1 = set(pos_ap_indices[pos1].tolist())
        set2 = set(pos_ap_indices[pos2].tolist())
        all_aps = list(set1 | set2)
        
        if not all_aps:
            return 0
        
        all_aps_arr = np.array(all_aps, dtype=np.int32)
        sub_aps = aps_arr[all_aps_arr]
        
        old_c1 = color_arr[pos1]
        old_c2 = color_arr[pos2]
        new_c1 = 1 - old_c1
        new_c2 = 1 - old_c2
        
        # Current colors of these APs
        cur_colors = color_arr[sub_aps]  # shape (k, 4)
        
        # For each AP, determine what happens after swap
        # We need to check if the AP is mono before and after
        
        # Before: mono if all 4 are same
        was_mono = (cur_colors[:, 0] == cur_colors[:, 1]) & (cur_colors[:, 1] == cur_colors[:, 2]) & (cur_colors[:, 2] == cur_colors[:, 3])
        
        # After swap: for each AP, replace pos1's color with new_c1 and pos2's color with new_c2
        after_colors = cur_colors.copy()
        
        # For each position in the AP, if it's pos1, set to new_c1; if pos2, set to new_c2
        for j in range(4):
            col = sub_aps[:, j]
            mask1 = (col == pos1)
            mask2 = (col == pos2)
            after_colors[mask1, j] = new_c1
            after_colors[mask2, j] = new_c2
        
        will_mono = (after_colors[:, 0] == after_colors[:, 1]) & (after_colors[:, 1] == after_colors[:, 2]) & (after_colors[:, 2] == after_colors[:, 3])
        
        return int(np.sum(will_mono) - np.sum(was_mono))
    
    def local_search(color, max_iters=300, use_swaps=True):
        """Greedy local search with flip and swap moves."""
        color_arr = np.array(color, dtype=np.int8)
        cur_count = count_mono_np(color_arr)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            best_delta = 0
            best_move = None  # (type, pos) or (type, pos1, pos2)
            
            # Try all single flips
            for pos in range(n):
                d = fast_delta_flip(color_arr, pos)
                if d < best_delta:
                    best_delta = d
                    best_move = ('flip', pos)
            
            # Try some random swaps
            if use_swaps:
                for _ in range(50):
                    p1 = random.randint(0, n - 1)
                    p2 = random.randint(0, n - 1)
                    if p1 == p2:
                        continue
                    d = fast_delta_swap(color_arr, p1, p2)
                    if d < best_delta:
                        best_delta = d
                        best_move = ('swap', p1, p2)
            
            if best_move is not None:
                if best_move[0] == 'flip':
                    pos = best_move[1]
                    color_arr[pos] = 1 - color_arr[pos]
                else:
                    p1, p2 = best_move[1], best_move[2]
                    color_arr[p1], color_arr[p2] = color_arr[p2], color_arr[p1]
                cur_count += best_delta
                improved = True
        
        return color_arr.tolist(), cur_count
    
    def simulated_annealing(color, max_iters=3000, t0=3.0, t1=0.01):
        """Simulated annealing with flip and swap moves."""
        color_arr = np.array(color, dtype=np.int8)
        cur_count = count_mono_np(color_arr)
        best_color = color_arr.copy()
        best_count = cur_count
        t = t0
        cooling = (t1 / t0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: 70% flip, 30% swap
            if random.random() < 0.7:
                pos = random.randint(0, n - 1)
                d = fast_delta_flip(color_arr, pos)
                new_count = cur_count + d
                if new_count < cur_count or (d > 0 and random.random() < np.exp(-d / max(t, 1e-10))):
                    color_arr[pos] = 1 - color_arr[pos]
                    cur_count = new_count
                    if cur_count < best_count:
                        best_count = cur_count
                        best_color = color_arr.copy()
            else:
                p1 = random.randint(0, n - 1)
                p2 = random.randint(0, n - 1)
                if p1 != p2:
                    d = fast_delta_swap(color_arr, p1, p2)
                    new_count = cur_count + d
                    if new_count < cur_count or (d > 0 and random.random() < np.exp(-d / max(t, 1e-10))):
                        color_arr[p1], color_arr[p2] = color_arr[p2], color_arr[p1]
                        cur_count = new_count
                        if cur_count < best_count:
                            best_count = cur_count
                            best_color = color_arr.copy()
            
            t *= cooling
        
        return best_color.tolist(), best_count
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def random_color(n):
        return [random.randint(0, 1) for _ in range(n)]
    
    # Multi-restart local search with diverse initializations
    best_color = None
    best_score = float('inf')
    
    # Generate diverse starting points
    initial_colors = []
    
    # Thue-Morse and complement
    tm = thue_morse(n)
    initial_colors.append(tm)
    initial_colors.append([1 - x for x in tm])
    
    # Structured patterns with various block sizes
    for block in [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 17, 20, 32, 33, 64]:
        initial_colors.append([(i // block) % 2 for i in range(n)])
    
    # More random initializations
    for _ in range(12):
        initial_colors.append(random_color(n))
    
    # Evaluate all and keep track of best
    for init in initial_colors:
        if time.time() > deadline - 15:
            break
        polished, sc = local_search(init, max_iters=200, use_swaps=True)
        if sc < best_score:
            best_score = sc
            best_color = polished[:]
    
    # If we have time, do more restarts from perturbed best
    num_extra_restarts = 0
    while time.time() < deadline - 20:
        num_extra_restarts += 1
        # Perturb best solution
        perturbed = best_color[:]
        num_flips = random.randint(4, 12)
        flip_positions = random.sample(range(n), num_flips)
        for pos in flip_positions:
            perturbed[pos] = 1 - perturbed[pos]
        
        polished, sc = local_search(perturbed, max_iters=150, use_swaps=True)
        if sc < best_score:
            best_score = sc
            best_color = polished[:]
        
        # Occasionally try SA from perturbed
        if num_extra_restarts % 3 == 0 and time.time() < deadline - 25:
            sa_color, sa_score = simulated_annealing(perturbed, max_iters=1500)
            if sa_score < best_score:
                best_score = sa_score
                best_color = sa_color[:]
    
    # Final SA polish on best
    if time.time() < deadline - 10:
        sa_color, sa_score = simulated_annealing(best_color, max_iters=3000)
        if sa_score < best_score:
            best_color = sa_color
            best_score = sa_score
    
    # Final local search
    if time.time() < deadline - 3:
        polished, sc = local_search(best_color, max_iters=200, use_swaps=True)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    return best_color
