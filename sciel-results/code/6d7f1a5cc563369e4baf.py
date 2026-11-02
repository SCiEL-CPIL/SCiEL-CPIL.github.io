import time
import random
import numpy as np

def _word_to_idx(w, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + w[i]
    return idx

def _idx_to_word(idx, q, n):
    w = [0] * n
    for i in range(n-1, -1, -1):
        w[i] = idx % q
        idx //= q
    return tuple(w)

def _get_neighbors(cw, q, n, radius=1):
    """Yields tuples of neighbors."""
    yield tuple(cw)
    for i in range(n):
        orig = cw[i]
        for v in range(q):
            if v != orig:
                new_cw = list(cw)
                new_cw[i] = v
                yield tuple(new_cw)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate the baseline product code (729 words)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    # Note: This assumes q=3. The task is specific to q=3, n=8.
    # If q or n are different, this might not be optimal, but the task is fixed.
    
    baseline_code = []
    for h in tetracode:
        for tail in np.ndindex(q, q, q, q):
            baseline_code.append(tuple(h + list(tail)))
    
    best_code = baseline_code[:]
    best_size = len(best_code)
    
    # 2. Local Search / Simulated Annealing
    # State: code (list of tuples), cover_count (numpy array)
    
    total_space = q ** n
    cover_count = np.zeros(total_space, dtype=np.int32)
    
    # Initialize cover count for baseline
    for c in baseline_code:
        for nb in _get_neighbors(c, q, n, radius):
            cover_count[_word_to_idx(nb, q, n)] += 1
            
    current_code = baseline_code[:]
    current_size = len(current_code)
    
    # Helper to check validity (all cover_count > 0)
    def is_valid():
        return np.all(cover_count > 0)
    
    # Helper to update cover count
    def add_cw(c):
        for nb in _get_neighbors(c, q, n, radius):
            cover_count[_word_to_idx(nb, q, n)] += 1
            
    def remove_cw(c):
        for nb in _get_neighbors(c, q, n, radius):
            cover_count[_word_to_idx(nb, q, n)] -= 1

    # Local Search Loop
    # We try to reduce the size.
    # Move 1: Remove a random codeword if it doesn't break validity.
    # Move 2: Swap a codeword with a random one if it maintains validity and allows future removals?
    # Actually, simple "Remove if possible" is stuck at 729.
    # We need to allow temporary invalidity or use a different heuristic.
    
    # Let's try a Greedy Construction from scratch as an alternative path.
    # If Greedy finds something smaller, we use it.
    
    def greedy_construct():
        covered = np.zeros(total_space, dtype=bool)
        code = []
        # Randomized greedy: pick random uncovered word, add random neighbor
        # This is slow. 
        # Better: Start with empty, pick word that covers most uncovered?
        # Too slow for 6561.
        
        # Instead, let's do a "Random Restart" local search on the product code
        # with a "Swap" move that can break the structure.
        
        pass

    # Let's implement a simple Simulated Annealing
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if not is_valid():
            # Repair: add a random codeword covering an uncovered word
            uncovered_idx = np.where(cover_count == 0)[0]
            if len(uncovered_idx) > 0:
                target_idx = uncovered_idx[random.randint(0, len(uncovered_idx)-1)]
                target_word = _idx_to_word(target_idx, q, n)
                # Pick a random neighbor of target_word to add
                neighbors = list(_get_neighbors(target_word, q, n, radius))
                new_cw = neighbors[random.randint(0, len(neighbors)-1)]
                if new_cw not in current_code:
                    current_code.append(new_cw)
                    add_cw(new_cw)
                    current_size += 1
            continue
        
        # Valid state. Try to improve (reduce size).
        improved = False
        
        # 1. Try removals
        # Pick a random codeword, check if removable
        if current_size > 0:
            i = random.randint(0, current_size - 1)
            c = current_code[i]
            # Check if all neighbors have cover_count > 1
            removable = True
            for nb in _get_neighbors(c, q, n, radius):
                if cover_count[_word_to_idx(nb, q, n)] <= 1:
                    removable = False
                    break
            
            if removable:
                remove_cw(c)
                current_code.pop(i)
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
        
        # 2. Try swaps (if no removal)
        if not improved:
            # Pick a random codeword to swap
            i = random.randint(0, current_size - 1)
            c_old = current_code[i]
            
            # Generate a random new codeword
            c_new = tuple(random.randint(0, q-1) for _ in range(n))
            
            # Check if swap is valid
            # Temporarily remove c_old
            for nb in _get_neighbors(c_old, q, n, radius):
                cover_count[_word_to_idx(nb, q, n)] -= 1
            
            # Temporarily add c_new
            for nb in _get_neighbors(c_new, q, n, radius):
                cover_count[_word_to_idx(nb, q, n)] += 1
            
            if is_valid():
                # Accept swap
                current_code[i] = c_new
                # Update best if size is same? No, size is same.
                # But it might enable future removals.
                # We don't update best_code here unless we find a smaller one later.
                improved = True
            else:
                # Revert
                for nb in _get_neighbors(c_new, q, n, radius):
                    cover_count[_word_to_idx(nb, q, n)] -= 1
                for nb in _get_neighbors(c_old, q, n, radius):
                    cover_count[_word_to_idx(nb, q, n)] += 1
            
        if not improved:
            # Stuck? Try adding a random word to explore?
            # Or just cool down?
            # For now, just continue, hoping random swaps lead somewhere.
            pass
            
        T *= cooling

    return [list(c) for c in best_code]
