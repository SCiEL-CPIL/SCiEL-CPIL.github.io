"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # For the specific case q=3, n=7, radius=1
    # Use a greedy construction with local search (swap-based) to minimize codeword count.
    # Deadline guard: return best result found before 140s wall-clock.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # All possible words
    all_words = list(itertools.product(range(3), repeat=7))
    all_words_set = set(all_words)
    
    # Precompute neighbors for each word (within Hamming distance 1)
    # For a word w, its neighbors are w itself and words differing in exactly one position
    def get_neighbors(w):
        w = list(w)
        neighbors = [tuple(w)]
        for i in range(7):
            old = w[i]
            for v in range(3):
                if v != old:
                    w[i] = v
                    neighbors.append(tuple(w))
            w[i] = old
        return neighbors
    
    # Precompute neighbor sets for all words
    print("Precomputing neighbors...")
    neighbor_sets = {}
    for w in all_words:
        neighbor_sets[w] = get_neighbors(w)
    print("Precomputation done.")
    
    # Greedy construction: start with empty set, greedily add words that cover most uncovered words
    # First, build initial code using greedy set cover
    print("Running greedy construction...")
    
    # Track which words are covered
    covered = set()
    code = []
    
    # For greedy, we want to pick codewords that cover the most uncovered words
    # Precompute for each potential codeword, which words it covers (i.e., its neighbor set)
    # But 2187 * 16 = ~35k entries, manageable
    
    # Actually, let's do a smarter greedy: 
    # Start with a random permutation, pick the word that covers the most uncovered words
    
    # For efficiency, track uncovered words
    uncovered = set(all_words)
    
    # Greedy set cover
    while uncovered:
        if time.time() > deadline:
            break
        
        # Find the codeword that covers the most uncovered words
        best_word = None
        best_count = -1
        
        # Sample a subset of candidates for speed, or iterate all
        # For 2187 words, iterating all is fine
        for w in all_words:
            # Count how many uncovered words are in w's neighbor set
            cnt = len(neighbor_sets[w] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
        
        if best_word is None:
            break
        
        code.append(list(best_word))
        covered.update(neighbor_sets[best_word])
        uncovered -= neighbor_sets[best_word]
        
        if len(code) % 10 == 0:
            print(f"Greedy: {len(code)} codewords, {len(uncovered)} uncovered")
    
    print(f"Greedy construction done: {len(code)} codewords")
    
    # Local search: try to remove codewords or swap them to reduce count
    # Strategy: for each codeword, check if removing it leaves all its uniquely covered words still covered
    # Then try swaps: replace one codeword with another to potentially allow removal of another
    
    def is_valid_code(code_list):
        """Check if code_list covers all words."""
        if not code_list:
            return len(all_words) == 0
        covered = set()
        for w in code_list:
            covered.update(neighbor_sets[tuple(w)])
        return covered == all_words_set
    
    # Verify initial code
    if not is_valid_code(code):
        print("Warning: initial code is invalid, using fallback")
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=7 - 4)]
    
    best_code = [w[:] for w in code]
    best_size = len(code)
    
    print(f"Starting local search from {best_size} codewords")
    
    # Local search: try removing codewords, then try swaps
    # First pass: try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline:
                return best_code
            
            # Try removing code[i]
            new_code = code[:i] + code[i+1:]
            if is_valid_code(new_code):
                code = new_code
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = [w[:] for w in code]
                    print(f"Removed codeword, now {len(code)}")
                break  # Restart after removal
    
    # Swap-based local search
    # For each pair (i, j), try replacing code[i] with some other word w such that
    # the new code is still valid and we can then remove code[j]
    
    # Precompute for each codeword, which words are ONLY covered by it (critical words)
    # A word is critical for codeword c if it's within distance 1 of c but not within distance 1 of any other codeword
    
    def get_critical_words(code_list):
        """For each codeword, find words only covered by it."""
        # For each word in the space, find which codewords cover it
        # critical[i] = set of words only covered by code[i]
        critical = [set() for _ in range(len(code_list))]
        
        for w in all_words:
            covering = []
            for i, c in enumerate(code_list):
                if w in neighbor_sets[tuple(c)]:
                    covering.append(i)
            if len(covering) == 1:
                critical[covering[0]].add(w)
        return critical
    
    # Swap search
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        no_improve_count += 1
        improvement = False
        
        # Try removing a codeword
        for i in range(len(code)):
            new_code = code[:i] + code[i+1:]
            if is_valid_code(new_code):
                code = new_code
                improvement = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = [w[:] for w in code]
                    print(f"Local search: removed codeword, now {len(code)}")
                no_improve_count = 0
                break
        
        if improvement:
            continue
        
        # Try swaps: replace code[i] with a random word, check validity
        # To be more efficient, only consider words that are neighbors of some uncovered-by-removal word
        for i in range(len(code)):
            if time.time() > deadline:
                break
            
            # What words become uncovered if we remove code[i]?
            removed_code = code[i]
            new_code_without_i = code[:i] + code[i+1:]
            
            # Find words only covered by code[i]
            only_covered_by_i = set()
            for w in neighbor_sets[tuple(removed_code)]:
                is_covered_by_other = False
                for j, c in enumerate(new_code_without_i):
                    if w in neighbor_sets[tuple(c)]:
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    only_covered_by_i.add(w)
            
            if not only_covered_by_i:
                # Can just remove it
                code = new_code_without_i
                improvement = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = [w[:] for w in code]
                    print(f"Local search: removed codeword (no critical words), now {len(code)}")
                no_improve_count = 0
                break
            
            # Try to find a replacement word that covers all words in only_covered_by_i
            # The replacement must be within distance 1 of all words in only_covered_by_i
            # This means the replacement must be in the intersection of their neighbor sets
            
            # Sample some candidate replacements
            # A good candidate is a word that is within distance 1 of many words in only_covered_by_i
            # For simplicity, try the original word and its neighbors
            
            candidates = list(neighbor_sets[tuple(removed_code)])
            random.shuffle(candidates)
            
            for cand in candidates[:20]:  # Try up to 20 candidates
                new_code = new_code_without_i + [list(cand)]
                if is_valid_code(new_code):
                    # Check if this swap allows further reduction
                    code = new_code
                    improvement = True
                    no_improve_count = 0
                    break
            
            if improvement:
                break
    
    if time.time() > deadline:
        return best_code
    
    return best_code

# EVOLVE-BLOCK-END
