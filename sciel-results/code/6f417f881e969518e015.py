"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use an advanced local search heuristic
    with multiple restarts and efficient coverage tracking to find minimal covering codes.
    
    Strategy:
    1. Precompute neighbor sets for all words in the space.
    2. Use a greedy algorithm to get an initial valid covering code.
    3. Apply local search: 
       a. Try removing codewords if coverage is maintained.
       b. Try swapping codewords to improve coverage or enable further removals.
    4. Use multiple random restarts with different initial seeds to escape local minima.
    5. Return the best solution found within time constraints.
    """
    # Check if we're in the specific case we want to optimize for
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Generic fallback for other parameters
    return _construct_generic(q, n, radius)

def _construct_3_6_1():
    """
    Optimized construction for q=3, n=6, radius=1.
    Uses efficient local search with multiple restarts.
    """
    total_words = 3 ** 6  # 729
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors using bit manipulation for efficiency
    # Encode each word as an integer (base-3)
    def encode(word):
        code = 0
        for digit in word:
            code = code * 3 + digit
        return code
    
    def decode(code):
        word = []
        for _ in range(6):
            word.append(code % 3)
            code //= 3
        return tuple(reversed(word))
    
    # Create mapping from encoded word to index
    encoded_words = [encode(w) for w in all_words]
    enc_to_idx = {enc: i for i, enc in enumerate(encoded_words)}
    
    # Precompute neighbors for each word
    # neighbors[i] = list of indices of words within Hamming distance 1 of word i
    neighbors = [[] for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i)  # Include self
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                neighbors[i].append(j)
    
    # Create adjacency matrix for faster coverage computation
    # Use numpy for vectorized operations
    adj_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in neighbors[i]:
            adj_matrix[i, j] = 1
    
    def _greedy_initial():
        """Generate an initial covering code using greedy algorithm."""
        uncovered = np.ones(total_words, dtype=np.uint8)
        code = set()
        
        while uncovered.any():
            # Find word that covers most uncovered words
            best_word = -1
            max_covered = -1
            
            for i in range(total_words):
                if i in code:
                    continue
                covered_count = np.sum(uncovered[neighbors[i]])
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = i
            
            if best_word == -1:
                break
            
            code.add(best_word)
            uncovered[neighbors[best_word]] = 0
        
        return code
    
    def _local_search(code_set, max_time=10):
        """
        Apply local search to minimize code size while maintaining coverage.
        Returns the improved code set.
        """
        start_time = time.time()
        code = set(code_set)
        
        while time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Try removing words
            for c in list(code):
                new_code = code - {c}
                if _is_valid(new_code):
                    code = new_code
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swapping words
            code_list = list(code)
            for i, c_remove in enumerate(code_list):
                # Try adding each word not in code
                for c_add in range(total_words):
                    if c_add in code:
                        continue
                    new_code = (code - {c_remove}) | {c_add}
                    if _is_valid(new_code):
                        code = new_code
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return code
    
    def _is_valid(code_set):
        """Check if code_set covers all words."""
        if not code_set:
            return total_words == 0
        
        covered = np.zeros(total_words, dtype=np.uint8)
        for c in code_set:
            covered[neighbors[c]] = 1
        
        return np.all(covered == 1)
    
    def _coverage_deficit(code_set):
        """Calculate how many words are not covered."""
        if not code_set:
            return total_words
        
        covered = np.zeros(total_words, dtype=np.uint8)
        for c in code_set:
            covered[neighbors[c]] = 1
        
        return np.sum(covered == 0)
    
    # Multiple restarts with different initializations
    best_code = None
    best_size = total_words  # Worst case
    start_time = time.time()
    time_limit = 115  # Leave some buffer
    
    for restart in range(20):  # Up to 20 restarts
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Tetracode-based initialization
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            initial_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    word = tuple(h + list(tail))
                    initial_code.add(word_to_idx[word])
        elif restart == 1:
            # Greedy initialization
            initial_code = _greedy_initial()
        else:
            # Random initialization with greedy completion
            random.seed(restart * 12345)
            initial_code = set()
            
            # Start with some random words
            num_random = min(50, total_words // 3)
            for _ in range(num_random):
                initial_code.add(random.randint(0, total_words - 1))
            
            # Complete with greedy
            uncovered = np.ones(total_words, dtype=np.uint8)
            for c in initial_code:
                uncovered[neighbors[c]] = 0
            
            while uncovered.any():
                best_word = -1
                max_covered = -1
                for i in range(total_words):
                    if i in initial_code:
                        continue
                    covered_count = np.sum(uncovered[neighbors[i]])
                    if covered_count > max_covered:
                        max_covered = covered_count
                        best_word = i
                
                if best_word == -1:
                    break
                
                initial_code.add(best_word)
                uncovered[neighbors[best_word]] = 0
        
        # Apply local search
        improved_code = _local_search(initial_code, max_time=5)
        
        # Update best if better
        if len(improved_code) < best_size:
            best_size = len(improved_code)
            best_code = improved_code
            print(f"Restart {restart}: Found code of size {best_size}")
    
    # Final local search on the best code with more time
    if best_code is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0:
            best_code = _local_search(best_code, max_time=remaining_time)
    
    # Convert back to list of lists
    if best_code is None:
        # Fallback to greedy
        best_code = _greedy_initial()
    
    result = [list(all_words[i]) for i in sorted(best_code)]
    return result

def _construct_generic(q, n, radius):
    """
    Generic construction for other parameter values.
    Uses greedy algorithm with local search improvement.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        # Generate all words within Hamming distance 'radius'
        # For efficiency, use iterative approach
        def _generate_neighbors(word, max_dist):
            result = set()
            _gen_neighbors_recursive(word, 0, max_dist, result)
            return result
        
        def _gen_neighbors_recursive(word, pos, remaining_dist, result):
            if pos == n:
                result.add(word)
                return
            
            # Option 1: Don't change this position
            if remaining_dist >= 0:
                _gen_neighbors_recursive(word, pos + 1, remaining_dist, result)
            
            # Option 2: Change this position
            if remaining_dist > 0:
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = list(word)
                    new_word[pos] = val
                    _gen_neighbors_recursive(tuple(new_word), pos + 1, remaining_dist - 1, result)
        
        neighbors[w] = _generate_neighbors(w, radius)
    
    # Greedy algorithm
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        # Find the word that covers the most uncovered words
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
    
    # Local search: try to remove words
    improved = True
    while improved:
        improved = False
        for c in list(code):
            new_code = [x for x in code if x != c]
            # Check coverage
            covered = set()
            for cc in new_code:
                covered.update(neighbors[cc])
            if covered == set(all_words):
                code = new_code
                improved = True
                break
    
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
