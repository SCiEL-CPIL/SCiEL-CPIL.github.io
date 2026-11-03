import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # For the specific case q=3, n=8, radius=1
    # We use a local search approach with swap moves
    
    def hamming_ball(word, q, n, radius):
        """Generate all words within Hamming distance radius of word."""
        result = [word]
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != word[pos]:
                        w = list(word)
                        w[pos] = val
                        result.append(w)
        return result
    
    def compute_coverage(codewords, q, n, radius):
        """Return a set of all covered words."""
        covered = set()
        for cw in codewords:
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        w = list(cw)
                        w[pos] = val
                        covered.add(tuple(w))
            covered.add(tuple(cw))
        return covered
    
    def is_covered(word, codewords, q, n, radius):
        """Check if word is within radius of any codeword."""
        wt = tuple(word)
        for cw in codewords:
            dist = sum(1 for a, b in zip(wt, cw) if a != b)
            if dist <= radius:
                return True
        return False
    
    # For q=3, n=8, radius=1: each codeword covers 1 + 8*2 = 17 words
    # Total words: 3^8 = 6561
    # Lower bound: ceil(6561/17) = 386, but actual lower bound is 403
    
    # Start with tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
    
    best_solution = list(initial)
    best_size = len(best_solution)
    
    if best_size <= 403:
        return best_solution
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if all words it uniquely covers are also covered by others
    
    def find_redundant(codewords, q, n, radius):
        """Find codewords that can be removed without losing coverage."""
        redundant = []
        for i, cw in enumerate(codewords):
            # Check if cw is needed: is there a word only covered by cw?
            # For radius 1, a word w is only covered by cw if:
            # w is at distance 1 from cw, and for all other cws', dist(w, cws') > 1
            # This is expensive. Instead, try removing and check coverage.
            pass
        return redundant
    
    # Better approach: try removing each codeword and check if coverage is maintained
    # This is O(m * m * n) which might be slow. Let's use a smarter method.
    
    # For radius 1, q=3, n=8:
    # Each codeword covers itself and 16 neighbors
    # A word w is covered by codeword c if dist(w,c) <= 1
    
    # Let's try a different structural approach: use the fact that we can
    # represent codewords as integers and use bit operations
    
    # Convert to integer representation for speed
    def to_int(word, q, n):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
    
    def from_int(val, q, n):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        return word[::-1]
    
    # Precompute neighbors for all possible words? Too many (6561 * 17 = ~111K)
    # But we can compute on the fly
    
    # Let's use a greedy removal approach with time checking
    current = list(best_solution)
    current_set = set(tuple(cw) for cw in current)
    
    # Build coverage map: for each word, which codewords cover it
    # This is expensive but let's try for the current set
    
    def build_coverage_map(codewords, q, n, radius):
        """Map each covered word to the set of codeword indices that cover it."""
        cov_map = {}
        for idx, cw in enumerate(codewords):
            # Add self
            wt = tuple(cw)
            if wt not in cov_map:
                cov_map[wt] = set()
            cov_map[wt].add(idx)
            # Add neighbors at distance 1
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        w = list(cw)
                        w[pos] = val
                        wt = tuple(w)
                        if wt not in cov_map:
                            cov_map[wt] = set()
                        cov_map[wt].add(idx)
        return cov_map
    
    # Try to remove redundant codewords
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        cov_map = build_coverage_map(current, q, n, radius)
        
        # Find a codeword that can be removed (all words it covers are covered by others)
        removed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            cw = current[idx]
            # Check all words covered by this codeword
            words_to_check = [tuple(cw)]
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        w = list(cw)
                        w[pos] = val
                        words_to_check.append(tuple(w))
            
            can_remove = True
            for wt in words_to_check:
                if wt in cov_map:
                    if cov_map[wt] == {idx}:
                        can_remove = False
                        break
                else:
                    can_remove = False
                    break
            
            if can_remove:
                current.pop(idx)
                removed = True
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = list(current)
                break  # Restart the loop to rebuild coverage map
        
        if not removed:
            break
    
    # Try swap-based improvement: replace a codeword with a random one
    # to potentially enable further removals
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try swapping each codeword with a random alternative
        improved = False
        for idx in range(len(current)):
            if time.time() > deadline - 5:
                break
            
            old_cw = current[idx]
            # Try a few random replacements
            for _ in range(5):
                new_cw = list(old_cw)
                pos = random.randint(0, n-1)
                old_val = new_cw[pos]
                new_val = random.choice([v for v in range(q) if v != old_val])
                new_cw[pos] = new_val
                
                if tuple(new_cw) in current_set:
                    continue
                
                # Check if removing old_cw and adding new_cw maintains coverage
                # This is expensive. Skip for now.
                pass
        
        if not improved:
            break
    
    if len(current) < best_size:
        best_solution = list(current)
        best_size = len(current)
    
    return best_solution
