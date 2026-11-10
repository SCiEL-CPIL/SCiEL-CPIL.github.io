import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    # Point representation: index 0..4095
    # x = i // N, y = i % N
    num_points = N * N
    
    # Precompute coordinates
    xs = np.array([i // N for i in range(num_points)], dtype=np.int32)
    ys = np.array([i % N for i in range(num_points)], dtype=np.int32)
    coords = list(zip(xs.tolist(), ys.tolist()))
    
    # Data structures for incremental validity checks
    selected = []
    point_set = set()
    # dist_sets[q] = set of squared distances from q to other selected points
    dist_sets = {}
    
    def _dist2_idx(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy

    def _can_add(p_idx):
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique (no isosceles with vertex p)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx] (no isosceles with vertex q)
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        point_set.remove(p_idx)
        # Remove from list
        try:
            idx = selected.index(p_idx)
            selected.pop(idx)
        except ValueError:
            pass
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            if d in dist_sets[q_idx]:
                dist_sets[q_idx].remove(d)
        
        if p_idx in dist_sets:
            del dist_sets[p_idx]

    def _save_state():
        return {
            'selected': selected[:],
            'point_set': point_set.copy(),
            'dist_sets': {k: set(v) for k, v in dist_sets.items()}
        }

    def _restore_state(state):
        selected.clear()
        selected.extend(state['selected'])
        point_set.clear()
        point_set.update(state['point_set'])
        dist_sets.clear()
        for k, v in state['dist_sets'].items():
            dist_sets[k] = set(v)

    rng = random.Random(12345)
    
    # 1. Greedy Initialization
    best_size = 0
    best_state = None
    
    candidates = list(range(num_points))
    rng_init = random.Random(42)
    rng_init.shuffle(candidates)
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected)
    best_state = _save_state()

    # 2. Local Search with Kick
    stuck_count = 0
    max_stuck_before_remove = 50
    kick_batch_size = 200
    
    # Precompute all pairwise squared distances for faster _can_add if needed, 
    # but current O(K) is fine. 
    # Let's optimize _can_add by using numpy for batch distance calculation?
    # No, K is small. The bottleneck is the number of random tries.
    
    # Improvement: Instead of purely random tries, we can maintain a list of "promising" candidates
    # or just increase the batch size.
    
    # Another improvement: "Conflict-based removal"
    # When we are stuck, remove the point that causes the most conflicts.
    # Conflict with point q: number of unselected points p such that d(p,q) is already in dist_sets[q] OR d(p,q) is in seen_dists for some other p'... 
    # This is hard to compute exactly. 
    # Approximation: Remove the point q that has the largest number of "blocked" distances.
    # i.e., for each q, count how many squared distances are in dist_sets[q]. 
    # The larger the set, the more it blocks.
    
    while time.time() < deadline:
        # 1. Try to add points
        added_any = False
        
        # Batch of random additions
        for _ in range(kick_batch_size):
            if time.time() > deadline:
                break
                
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in point_set:
                continue
                
            if _can_add(p_idx):
                _add(p_idx)
                added_any = True
                stuck_count = 0
                
        if added_any:
            if len(selected) > best_size:
                best_size = len(selected)
                best_state = _save_state()
            continue
        
        # 2. If no adds, we are stuck
        stuck_count += 1
        
        if stuck_count > max_stuck_before_remove:
            if selected:
                # Strategy: Remove the point with the largest dist_set size (most blocking)
                # Or random if sizes are similar.
                # Let's try: pick the point with max len(dist_sets[q])
                # This is O(K).
                
                # To add diversity, sometimes pick random, sometimes max-blocking
                if rng.random() < 0.5:
                    # Random removal
                    remove_idx = rng.randint(0, len(selected) - 1)
                    p_idx_to_remove = selected[remove_idx]
                else:
                    # Remove most blocking
                    max_len = -1
                    p_idx_to_remove = selected[0]
                    for q_idx in selected:
                        l = len(dist_sets[q_idx])
                        if l > max_len:
                            max_len = l
                            p_idx_to_remove = q_idx
                
                _remove(p_idx_to_remove)
                stuck_count = 0
                
                # Repair pass
                for _ in range(kick_batch_size):
                    if time.time() > deadline:
                        break
                    p_idx = rng.randint(0, num_points - 1)
                    if p_idx in point_set:
                        continue
                    if _can_add(p_idx):
                        _add(p_idx)
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_state = _save_state()
                
        # Optional: If stuck for a very long time, restart from best state?
        if stuck_count > 500:
            _restore_state(best_state)
            stuck_count = 0

    # Convert best_state selected indices back to (x, y) tuples
    result = []
    for idx in best_state['selected']:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
