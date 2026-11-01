import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    balls_matrix = np.zeros((N, ball_size), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls.append(neighbors)
        balls_matrix[w] = neighbors
    
    # Precompute which words are in each ball (for faster checks)
    # Actually, we'll use the matrix directly
    
    best_size = N
    best_set = None
    
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def greedy_construct():
        """Greedy set cover construction"""
        selected = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        
        while np.any(uncovered):
            # Find word covering most uncovered
            gains = uncovered[balls_matrix].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            ball = balls[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
        
        return selected, cover_count
    
    def prune(selected, cover_count):
        """Remove redundant codewords"""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected
    
    def local_search(selected, cover_count, time_limit):
        """Local search with swaps and removals"""
        while time.time() < time_limit:
            if not selected:
                break
            
            # Randomly choose between swap and removal
            if random.random() < 0.3:
                # Try removal
                i = np.random.randint(len(selected))
                u = selected[i]
                if np.min(cover_count[balls[u]]) > 1:
                    # Safe to remove
                    cover_count[balls[u]] -= 1
                    selected.pop(i)
            else:
                # Try swap
                i = np.random.randint(len(selected))
                u = selected[i]
                v = np.random.randint(N)
                if v == u:
                    continue
                
                # Check if swap is safe
                # Risky words are those covered only by u
                risky = (cover_count[balls[u]] == 1)
                if np.any(risky):
                    # Check if v covers all risky words
                    risky_indices = np.where(risky)[0]
                    # Check if v's ball contains all risky words
                    # Create a set for v's ball for O(1) lookup
                    ball_v_set = set(balls[v])
                    if all(r in ball_v_set for r in risky_indices):
                        # Safe swap
                        cover_count[balls[u]] -= 1
                        cover_count[balls[v]] += 1
                        selected[i] = v
                    else:
                        continue
                else:
                    # No risky words, safe swap
                    cover_count[balls[u]] -= 1
                    cover_count[balls[v]] += 1
                    selected[i] = v
            
            # Prune after each successful move
            selected = prune(selected, cover_count)
        
        return selected, cover_count
    
    # Main loop: multiple restarts with different random seeds
    while time.time() < deadline - 10.0:
        # Greedy construction
        selected, cover_count = greedy_construct()
        
        # Initial prune
        selected = prune(selected, cover_count)
        
        # Local search
        selected, cover_count = local_search(selected, cover_count, deadline - 10.0)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 403:
                break
    
    if best_set is None:
        return [idx_to_list(w) for w in range(N)]
    
    return [idx_to_list(w) for w in best_set]
