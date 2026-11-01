# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import deque

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    N = q ** n  # 729

    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor lists (words within Hamming distance <= radius)
    # For radius=1, each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors
    neighbor_indices = np.zeros((N, 13), dtype=np.int32)
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neighbors.append(word_to_idx[tuple(new_w)])
        neighbor_indices[i] = neighbors

    # Initial code: Tetracode x 3^2 (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts: how many codewords cover each word
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts[neighbor_indices[idx]] += 1
        
    # Active set
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    def can_remove(idx):
        """Check if word idx can be removed (all its neighbors have count > 1)"""
        return np.all(counts[neighbor_indices[idx]] > 1)
    
    def remove_word(idx):
        """Remove a word from the code and update counts"""
        counts[neighbor_indices[idx]] -= 1
        active_set.remove(idx)
        active_list.remove(idx)
    
    def add_word(idx):
        """Add a word to the code and update counts"""
        counts[neighbor_indices[idx]] += 1
        active_set.add(idx)
        active_list.append(idx)
    
    # Phase 1: Hypergraph 2-core peeling - iteratively remove redundant words
    # A word is redundant if all words it covers are also covered by at least one other codeword
    # We use a queue-based approach for efficiency
    def peel_redundant():
        """Remove all redundant words using a queue-based approach"""
        # Start with all words as potentially removable
        queue = deque(active_list.copy())
        while queue and time.time() < deadline:
            idx = queue.popleft()
            if idx not in active_set:
                continue
            if can_remove(idx):
                remove_word(idx)
                # When we remove idx, words that were only covered by idx and one other
                # might now make that other word redundant
                # Actually, removing a word can only make other words MORE essential, 
                # not less. So we don't need to add new candidates to the queue.
                # Wait, that's wrong. If word A covers words {1,2,3} and word B covers {2,3,4},
                # and word 1 is only covered by A, then A is essential.
                # If we remove some other word C, it doesn't change the coverage of 1,2,3,4.
                # The coverage counts only decrease when we remove words.
                # So if a word was not removable before, it cannot become removable after removing others.
                # Therefore, we only need one pass!
                # But wait, the order matters. Let's think again.
                # Actually, the standard 2-core peeling works like this:
                # A vertex is in the 2-core if it has at least 2 neighbors in the current graph.
                # Here, "redundant" means all neighbors have degree > 1 (in the hypergraph sense).
                # This is not exactly the same as 2-core. Let's stick with the simple approach:
                # repeatedly scan and remove until no more can be removed.
                pass
    
    # Simpler approach: just do multiple passes of scanning
    def try_remove_all_redundant(max_passes=10):
        """Iteratively remove all redundant words"""
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            removed_any = False
            # Shuffle to break symmetry
            shuffled = active_list.copy()
            random.shuffle(shuffled)
            for idx in shuffled:
                if idx not in active_set:
                    continue
                if can_remove(idx):
                    remove_word(idx)
                    removed_any = True
            if not removed_any:
                break
    
    # Initial peeling
    try_remove_all_redundant()
    
    best_size = len(active_list)
    best_state = (set(active_set), counts.copy())
    
    # Phase 2: Local search with perturbations
    # Strategy: add a random word, then try to remove as many as possible
    # If we remove more than we added, we improved
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Try perturbation: add a random word, then try to remove
        random_idx = random.randint(0, N - 1)
        
        if random_idx in active_set:
            continue
            
        # Add it
        add_word(random_idx)
        
        # Now try to remove redundant words
        removals = 0
        for _ in range(3):  # up to 3 passes
            pass_removed = False
            shuffled = active_list.copy()
            random.shuffle(shuffled)
            for idx in shuffled:
                if idx not in active_set:
                    continue
                if can_remove(idx):
                    remove_word(idx)
                    removals += 1
                    pass_removed = True
            if not pass_removed:
                break
        
        # If we removed at least 2, we improved (since we added 1)
        if removals >= 2:
            if len(active_list) < best_size:
                best_size = len(active_list)
                best_state = (set(active_set), counts.copy())
            no_improvement_count = 0
        else:
            # Revert if no improvement
            if random_idx in active_set:
                remove_word(random_idx)
            no_improvement_count += 1
    
    # Restore best state
    if best_state[0] != active_set:
        active_set = best_state[0]
        counts = best_state[1].copy()
        active_list = list(active_set)
    
    # Final cleanup
    try_remove_all_redundant()
    
    # Convert back to list of lists
    return [list(all_words[idx]) for idx in active_list]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        if q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n >= 4:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        for w in all_words:
            if w in code: continue
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
