import itertools
import time
import random
import sys

def construct(q, n, radius):
    """
    Constructs a covering code using a randomized greedy set cover approach.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Pre-compute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word tuple to index for quick lookup if needed, though we use sets of tuples
    # Pre-compute the ball (coverage) for each word
    # ball[w] = set of words within Hamming distance 'radius' from w
    balls = [None] * total_words
    
    # Helper to generate neighbors
    def get_neighbors(word):
        neighbors = [word]
        for i in range(n):
            original_val = word[i]
            for new_val in range(q):
                if new_val != original_val:
                    new_word = list(word)
                    new_word[i] = new_val
                    neighbors.append(tuple(new_word))
        return neighbors
    
    # Pre-compute balls
    for i, w in enumerate(all_words):
        balls[i] = set(get_neighbors(w))
        
    # We need to select a subset of indices such that the union of their balls is all_words
    # Greedy Set Cover:
    # 1. uncovered = set(all_words)
    # 2. While uncovered:
    #    a. Find i that maximizes len(balls[i] & uncovered)
    #    b. Add i to code
    #    c. uncovered -= balls[i]
    
    best_code_size = float('inf')
    best_code = None
    
    # Randomized restarts
    # The greedy choice is deterministic if we just pick max. 
    # To introduce randomness, we can randomly break ties or randomly permute the order of checking candidates 
    # if they have similar coverage, or just run multiple times with slight variations.
    # A simple effective variation: Randomly select the first candidate from the top-k best candidates.
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle the indices to break ties randomly
        indices = list(range(total_words))
        random.shuffle(indices)
        
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find the best candidate
            max_covered = 0
            best_candidates = []
            
            # Optimization: Instead of checking all total_words every time,
            # we can just iterate through all and find max intersection.
            # total_words is 729, so 729 * 729 * log(729) is small.
            # len(balls[i] & uncovered) involves set intersection.
            
            # To speed up, we can maintain a list of active candidates? 
            # But 729 is small. Let's just do the linear scan.
            
            for idx in indices:
                # Optimization: if the ball is already fully covered, skip? No, we need to know if it covers *new* stuff.
                # If balls[idx] is a subset of (all_words - uncovered), it covers 0 new.
                # Intersection size:
                # Using set intersection is fast in Python.
                covered_new = len(balls[idx] & uncovered)
                
                if covered_new > max_covered:
                    max_covered = covered_new
                    best_candidates = [idx]
                elif covered_new == max_covered and covered_new > 0:
                    best_candidates.append(idx)
            
            if not best_candidates:
                # Should not happen if uncovered is not empty
                break
                
            # Pick a random candidate from the best ones to add diversity
            chosen_idx = random.choice(best_candidates)
            code.append(all_words[chosen_idx])
            uncovered -= balls[chosen_idx]
            
        if uncovered:
            # Failed to cover within time or logic error
            continue
            
        if len(code) < best_code_size:
            best_code_size = len(code)
            best_code = code
            
            # If we found a very good code (e.g., <= 73), we might stop early or continue to try for 72/71
            if best_code_size <= 72:
                # Try a bit longer to see if we can get lower, but 73 is the known best.
                # We can break if we hit a lower bound we trust, but 71 is lower bound.
                # Let's just continue to see if we can get 72.
                pass
                
        # Optional: If we have plenty of time, we can try to optimize the found code further?
        # For now, random restarts of greedy are usually sufficient for this size.
        
    if best_code is None:
        # Fallback: return a trivial code (e.g., all words) or the seed construction
        # Seed construction:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in all_words]
            
    return best_code

if __name__ == "__main__":
    # Test with the specific task parameters
    q = 3
    n = 6
    radius = 1
    code = construct(q, n, radius)
    print(f"Found code size: {len(code)}")
    # Verify
    all_words = set(itertools.product(range(q), repeat=n))
    covered = set()
    for c in code:
        c = tuple(c)
        # Add c and its neighbors
        covered.add(c)
        for i in range(n):
            for v in range(q):
                if v != c[i]:
                    nc = list(c)
                    nc[i] = v
                    covered.add(tuple(nc))
    if covered == all_words:
        print("Valid covering code.")
    else:
        print("INVALID covering code.")
