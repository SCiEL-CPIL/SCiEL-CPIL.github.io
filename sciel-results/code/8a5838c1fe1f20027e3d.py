"""Football pool problem for 6 matches: ternary covering code K_3(6,1).
Uses greedy set cover with randomized restarts and redundant word removal."""
import time
import random
import itertools

def _ball_indices(q, n, radius):
    """Precompute for each word index, the set of word indices within Hamming distance radius."""
    total = q ** n
    # Map word (tuple) to index
    word_to_idx = {}
    idx_to_word = []
    for idx in range(total):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        word = tuple(digits)
        idx_to_word.append(word)
        word_to_idx[word] = idx
    
    # For each word, compute all words within Hamming distance <= radius
    ball_map = {}
    for idx in range(total):
        word = idx_to_word[idx]
        ball = {idx}
        # Generate all words within distance 1
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = list(word)
                new_word[pos] = val
                new_word = tuple(new_word)
                ball.add(word_to_idx[new_word])
        ball_map[idx] = ball
    
    return ball_map, idx_to_word

def _greedy_set_cover(ball_map, n, q, radius, initial=None):
    """Greedy set cover to find a covering code."""
    total = q ** n
    uncovered = set(range(total))
    if initial is not None:
        selected = list(initial)
        for idx in selected:
            uncovered -= ball_map[idx]
    else:
        selected = []
    
    if not uncovered:
        return selected
    
    while uncovered:
        # Find the word that covers the most uncovered points
        best_idx = None
        best_count = -1
        for idx in range(total):
            if idx in selected:
                continue
            count = len(uncovered & ball_map[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_count <= 0:
            break
        selected.append(best_idx)
        uncovered -= ball_map[best_idx]
    
    return selected

def _remove_redundant(ball_map, selected, total):
    """Remove redundant words from the selected set."""
    if not selected:
        return selected
    
    selected_set = set(selected)
    # For each selected word, check if it's still needed
    to_remove = []
    for idx in selected:
        # Temporarily remove idx
        selected_set.discard(idx)
        # Check if all points covered by idx are still covered by others
        ball = ball_map[idx]
        still_covered = True
        for point in ball:
            if point not in selected_set:
                # Check if point is covered by any other selected word
                covered = False
                for other in selected_set:
                    if point in ball_map[other]:
                        covered = True
                        break
                if not covered:
                    still_covered = False
                    break
            else:
                # point is itself a selected word, so it's covered
                pass
        if still_covered:
            to_remove.append(idx)
        else:
            selected_set.add(idx)
    
    return [idx for idx in selected if idx not in set(to_remove)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    total = q ** n
    ball_map, idx_to_word = _ball_indices(q, n, radius)
    
    best_solution = None
    best_size = float('inf')
    
    # Initial solution: use the tetracode-based construction as a starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_words.append(tuple(word))
    
    # Convert to indices
    word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
    initial_indices = [word_to_idx[w] for w in initial_words if w in word_to_idx]
    
    # Greedy set cover from this initial point
    solution = _greedy_set_cover(ball_map, n, q, radius, initial=initial_indices)
    solution = _remove_redundant(ball_map, solution, total)
    
    if len(solution) < best_size:
        best_size = len(solution)
        best_solution = solution[:]
    
    # Now do randomized restarts with greedy
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random perturbation: start with a random subset or empty
        if rng.random() < 0.3:
            # Start empty
            initial = None
        else:
            # Start with a random small set
            size = rng.randint(0, 20)
            initial = rng.sample(range(total), min(size, total))
        
        solution = _greedy_set_cover(ball_map, n, q, radius, initial=initial)
        solution = _remove_redundant(ball_map, solution, total)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
    
    # Convert indices back to words
    if best_solution is None:
        # Fallback: return the initial tetracode-based solution
        return [list(w) for w in initial_words]
    
    return [list(idx_to_word[idx]) for idx in best_solution]
