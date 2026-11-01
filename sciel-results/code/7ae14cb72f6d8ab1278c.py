# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code using Branch-and-Bound exact set cover.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    # Total number of words in the space
    N = q ** n
    if N > 100000:
        # Fallback for large spaces: use a simple constructive approach
        return _constructive_cover(q, n, radius)

    # Generate all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    # Precompute balls: for each word, which words are within distance radius
    # balls[i] is a boolean array of length num_words
    # For radius 1, word i covers word j if they differ in at most 1 position
    
    # Efficient ball computation using numpy
    # For each position, compute which words differ in that position
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    
    if radius >= 1:
        # Precompute word_to_idx mapping for fast lookup
        word_to_idx = {}
        for idx, w in enumerate(all_words):
            word_to_idx[tuple(w)] = idx
        
        for i in range(num_words):
            base = all_words[i].copy()
            for p in range(n):
                old_val = base[p]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    base[p] = new_val
                    key = tuple(base)
                    if key in word_to_idx:
                        j = word_to_idx[key]
                        balls[i, j] = True
                    base[p] = old_val
    
    # Convert balls to bitsets for faster operations
    # Use Python integers as bitsets
    bitsets = []
    for i in range(num_words):
        bits = 0
        row = balls[i]
        # Convert boolean array to integer bitset
        # This is slow in pure Python, let's use a helper or just keep as boolean arrays
        # Actually, for 6561 words, boolean arrays are fine, but bitsets are faster for AND/OR
        # Let's use numpy arrays for simplicity and rely on numpy operations
        
    # Branch and Bound
    # State: covered_mask (boolean array), current_selection (list of indices), depth
    # Lower bound: simple counting - number of uncovered words / max coverage per word
    
    best_solution = None
    best_size = num_words + 1
    
    # Initial greedy solution to get a good upper bound
    initial_solution = _greedy_cover_numpy(all_words, balls, num_words, deadline)
    best_size = len(initial_solution)
    best_solution = initial_solution
    
    # Convert best_solution to indices for internal use
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx
    
    best_indices = [word_to_idx[tuple(c)] for c in best_solution]
    
    # Compute initial covered mask
    covered = np.zeros(num_words, dtype=bool)
    for idx in best_indices:
        covered |= balls[idx, :]
    
    # Sort words by coverage size descending for better branching
    coverage_sizes = balls.sum(axis=1)
    order = np.argsort(-coverage_sizes)
    
    # Branch and bound search
    def bnb(covered, current_indices, depth, deadline):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        # Check if covered is complete
        if covered.all():
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_solution = [all_words[i].tolist() for i in current_indices]
            return
        
        # Lower bound: number of uncovered words / max possible coverage
        uncovered_count = (~covered).sum()
        if uncovered_count == 0:
            return
        
        # Simple lower bound: at least ceil(uncovered_count / max_ball_size) more words needed
        # But we need to be careful: max_ball_size is the size of the largest ball
        # Actually, a tighter bound: for each uncovered word, it must be covered by some word in the ball
        # Let's use a simple bound: uncovered_count / max_coverage
        max_cov = coverage_sizes.max()
        lb = len(current_indices) + int(np.ceil(uncovered_count / max_cov))
        if lb >= best_size:
            return
        
        # Find the uncovered word with the fewest covering candidates (most constrained)
        # This is a good heuristic for branching
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            return
        
        # For each uncovered word, count how many potential codewords can cover it
        # potential_candidates[j] = number of i such that balls[i, j] is True and i is not in current_indices
        # This is expensive to compute exactly, so let's use a simpler heuristic:
        # Pick the uncovered word that has the fewest neighbors in the remaining candidate set
        
        # To speed up, let's just pick the first uncovered word or use a random one
        # Better: pick the word with the smallest degree in the remaining graph
        # For simplicity, let's pick the uncovered word with the fewest total neighbors
        # (approximation: use precomputed degree)
        
        # Precompute degrees
        degrees = balls.sum(axis=0)
        
        # Pick the uncovered word with the smallest degree
        min_degree = float('inf')
        target_word = -1
        for j in uncovered_indices:
            if degrees[j] < min_degree:
                min_degree = degrees[j]
                target_word = j
        
        if target_word == -1:
            return
        
        # Get all words that can cover target_word
        candidates = np.where(balls[:, target_word])[0]
        
        # Sort candidates by coverage size descending
        candidates = candidates[np.argsort(-coverage_sizes[candidates])]
        
        # Branch on each candidate
        for idx in candidates:
            if time.time() > deadline:
                return
            
            # Skip if already in current_indices
            if idx in current_indices:
                continue
            
            # Check if adding this word helps
            new_covered = covered | balls[idx, :]
            if (new_covered == covered).all():
                continue
            
            # Recurse
            bnb(new_covered, current_indices + [idx], depth + 1, deadline)
            
            # If we found a better solution, we might be able to prune
            if len(current_indices) + 1 >= best_size:
                break
    
    # Start the search
    bnb(covered, best_indices, 0, deadline)
    
    # If the search didn't improve, use the greedy solution
    if best_solution is None:
        return initial_solution
    
    return best_solution


def _greedy_cover_numpy(all_words, balls, num_words, deadline):
    """Greedy set cover using numpy."""
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    
    while not covered.all() and time.time() < deadline:
        # Find word that covers the most uncovered words
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]
    
    return [all_words[i].tolist() for i in selected]


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    # Use the tetracode construction for ternary case
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    # General fallback: greedy
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    # Compute balls
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True
    
    return _greedy_cover_numpy(all_words, balls, num_words, time.time() + 110)
# EVOLVE-BLOCK-END
