import time
import random

def construct():
    n = 256
    deadline = time.time() + 110
    
    # Thue-Morse initialization
    color = [bin(i).count('1') % 2 for i in range(n)]
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # For each position, list of (ap_index, offset)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Track mono status of each AP
    def compute_mono_status(color):
        mono = [False] * num_aps
        count = 0
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                count += 1
        return mono, count
    
    def delta_flip_fast(color, mono, p):
        """Compute delta in mono count when flipping position p."""
        delta = 0
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            if mono[ap_idx]:
                # Was mono, after flip it's not mono (since we change one element)
                delta -= 1
            else:
                # Was not mono. After flip, check if becomes mono.
                # The other 3 elements must all have the same color as the new value of p.
                new_val = 1 - color[p]
                # Check the other 3 positions
                others = []
                for j, pos in enumerate((p0, p1, p2, p3)):
                    if j != offset:
                        others.append(color[pos])
                if len(others) == 3 and all(v == new_val for v in others):
                    delta += 1
        return delta
    
    # Greedy local search
    def local_search(color, max_sweeps=100):
        mono, current = compute_mono_status(color)
        for sweep in range(max_sweeps):
            # Find positions in mono APs
            mono_positions = set()
            for idx in range(num_aps):
                if mono[idx]:
                    p0, p1, p2, p3 = aps[idx]
                    mono_positions.update([p0, p1, p2, p3])
            
            if not mono_positions:
                break
            
            best_delta = 0
            best_pos = -1
            for p in mono_positions:
                d = delta_flip_fast(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            
            if best_pos < 0:
                break
            
            # Apply flip
            color[best_pos] = 1 - color[best_pos]
            # Update mono status for APs containing best_pos
            for ap_idx, offset in pos_aps[best_pos]:
                p0, p1, p2, p3 = aps[ap_idx]
                mono[ap_idx] = (color[p0] == color[p1] == color[p2] == color[p3])
                if mono[ap_idx]:
                    current += 1
                else:
                    current -= 1
            # Wait, this double-counts. Let me redo.
            
            if time.time() > deadline - 5:
                break
        
        return color, current
    
    # Hmm, the mono status update is tricky. Let me just recompute each time.
    
    def local_search_v2(color, max_sweeps=200):
        current = 0
        mono = [False] * num_aps
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                current += 1
        
        for sweep in range(max_sweeps):
            mono_positions = set()
            for idx in range(num_aps):
                if mono[idx]:
                    p0, p1, p2, p3 = aps[idx]
                    mono_positions.update([p0, p1, p2, p3])
            
            if not mono_positions:
                break
            
            best_delta = 0
            best_pos = -1
            for p in mono_positions:
                d = delta_flip_fast(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            
            if best_pos < 0:
                break
            
            color[best_pos] = 1 - color[best_pos]
            
            # Recompute mono status for affected APs
            for ap_idx, offset in pos_aps[best_pos]:
                p0, p1, p2, p3 = aps[ap_idx]
                was_mono = mono[ap_idx]
                mono[ap_idx] = (color[p0] == color[p1] == color[p2] == color[p3])
                if mono[ap_idx] and not was_mono:
                    current += 1
                elif not mono[ap_idx] and was_mono:
                    current -= 1
            
            if time.time() > deadline - 5:
                break
        
        return color, current
    
    color, score = local_search_v2(color)
    
    best_color = color[:]
    best_score = score
    
    # Random restarts
    random.seed(42)
    while time.time() < deadline - 10:
        trial = best_color[:]
        n_flips = random.randint(5, 20)
        for _ in range(n_flips):
            p = random.randint(0, n - 1)
            trial[p] = 1 - trial[p]
        trial, s = local_search_v2(trial, max_sweeps=100)
        if s < best_score:
            best_score = s
            best_color = trial[:]
    
    return best_color
