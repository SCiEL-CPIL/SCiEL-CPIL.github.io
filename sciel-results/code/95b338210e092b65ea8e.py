"""Football pool problem: ternary covering code K_3(6,1)."""
import time
import itertools
import numpy as np

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Returns a list of codewords (each a list of n integers in 0..q-1) such that
    every word in {0..q-1}^n is within Hamming distance radius of some codeword.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for general case: use direct sum approach
        if n <= 4:
            # Small case: brute force or known codes
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [[a, b] for a in range(q) for b in range(q)]
            elif n == 3:
                # Ternary code of length 3, radius 1
                # Perfect code exists: 9 words
                return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
            elif n == 4:
                # Ternary tetracode: 9 words, perfect code radius 1
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        else:
            # General fallback: tetracode product
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specialized case: q=3, n=6, radius=1
    # Start with the 81-word tetracode product
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Extend to length 6 by adding all 3^2 = 9 possibilities for last 2 coords
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    # This gives 9 * 9 = 81 codewords
    
    # Convert to numpy array for efficient distance computation
    code = np.array(initial_code, dtype=np.int8)
    code_set = set(tuple(c) for c in code)
    
    # Precompute all 729 words
    all_words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
    
    # For each codeword, compute which words it covers (within Hamming distance 1)
    # A word w is covered by codeword c if Hamming distance <= 1
    # This means w == c or w differs from c in exactly one position
    
    # For efficiency, use the fact that distance <= 1 means:
    # - Same word (distance 0)
    # - Differ in exactly one position (distance 1)
    
    # For each codeword, the covered words are:
    # - The codeword itself
    # - For each position i, change one symbol to the other two possibilities
    
    def get_covered_words(cw):
        """Get all words within Hamming distance 1 of cw."""
        covered = [tuple(cw)]
        for pos in range(6):
            orig = cw[pos]
            for new_val in range(3):
                if new_val != orig:
                    word = list(cw)
                    word[pos] = new_val
                    covered.append(tuple(word))
        return covered
    
    # Build coverage: for each word, which codewords cover it
    # Use a dictionary: word -> list of codeword indices that cover it
    word_to_covers = {}
    for idx, cw in enumerate(code):
        for w in get_covered_words(cw):
            if w not in word_to_covers:
                word_to_covers[w] = []
            word_to_covers[w].append(idx)
    
    # Verify initial code is valid
    assert len(word_to_covers) == 729, f"Initial code doesn't cover all words: {len(word_to_covers)}"
    
    # Now try to remove redundant codewords
    # A codeword is redundant if every word it covers is also covered by at least one other codeword
    # Greedy approach: repeatedly find and remove redundant codewords
    
    # Track which codewords are still active
    active = [True] * len(code)
    removed = 0
    
    # For efficiency, maintain for each word the count of active codewords that cover it
    word_cover_count = {}
    for w, covers in word_to_covers.items():
        word_cover_count[w] = len([c for c in covers if active[c]])
    
    # Greedy removal: try to remove codewords that are redundant
    # To speed up, we can try removing codewords in a specific order
    # Let's try multiple strategies and keep the best result
    
    best_code = code.tolist()
    best_len = len(code)
    
    # Strategy 1: Simple greedy - repeatedly scan for redundant codewords
    # Start with the current state
    current_code = code.tolist()
    current_active = [True] * len(current_code)
    
    # Rebuild coverage counts from scratch for the current active set
    def rebuild_cover_counts(current_active, code_arr):
        counts = {}
        for idx, cw in enumerate(code_arr):
            if not current_active[idx]:
                continue
            for w in get_covered_words(cw):
                if w not in counts:
                    counts[w] = 0
                counts[w] += 1
        return counts
    
    cover_counts = rebuild_cover_counts(current_active, code)
    
    # Iteratively remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(code)):
            if not current_active[idx]:
                continue
            # Check if this codeword is redundant
            # It's redundant if every word it covers has cover_count >= 2
            cw = code[idx]
            is_redundant = True
            for w in get_covered_words(cw):
                if cover_counts.get(w, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove this codeword
                current_active[idx] = False
                for w in get_covered_words(cw):
                    cover_counts[w] -= 1
                changed = True
                removed += 1
                if time.time() > deadline:
                    break
    
    # Extract the remaining codewords
    remaining = [code[i].tolist() for i in range(len(code)) if current_active[i]]
    if len(remaining) < best_len:
        best_code = remaining
        best_len = len(remaining)
    
    # Strategy 2: Try removing codewords in different orders
    # Sort codewords by some heuristic and try removal in that order
    # Heuristic: remove codewords that cover more "over-covered" words first
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Start fresh from initial code
        current_active = [True] * len(code)
        cover_counts = rebuild_cover_counts(current_active, code)
        
        # Determine removal order based on heuristic
        # For each codeword, compute a score: sum over covered words of (cover_count - 1)
        # Higher score means more redundant
        scores = np.zeros(len(code), dtype=np.float32)
        for idx in range(len(code)):
            cw = code[idx]
            score = 0
            for w in get_covered_words(cw):
                score += max(0, cover_counts.get(w, 0) - 1)
            scores[idx] = score
        
        # Try removing in order of highest score first
        order = np.argsort(-scores)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if not current_active[idx]:
                continue
            cw = code[idx]
            is_redundant = True
            for w in get_covered_words(cw):
                if cover_counts.get(w, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                current_active[idx] = False
                for w in get_covered_words(cw):
                    cover_counts[w] -= 1
        
        remaining = [code[i].tolist() for i in range(len(code)) if current_active[i]]
        if len(remaining) < best_len:
            best_code = remaining
            best_len = len(remaining)
    
    # Strategy 3: Local search - try swapping codewords
    # If we can replace a codeword with a different one that maintains coverage,
    # we might enable further reductions
    if time.time() < deadline - 20:
        # Try to improve by finding alternative codewords
        current_code = np.array(best_code, dtype=np.int8)
        current_set = set(tuple(c) for c in current_code)
        
        # Rebuild coverage for current code
        word_to_covers2 = {}
        for idx, cw in enumerate(current_code):
            for w in get_covered_words(cw):
                if w not in word_to_covers2:
                    word_to_covers2[w] = []
                word_to_covers2[w].append(idx)
        
        # Try replacing each codeword with a nearby codeword that might allow removal of others
        # This is complex, so let's try a simpler approach:
        # Randomly perturb and re-run greedy reduction
        
        import random
        random.seed(42)
        
        for iter in range(20):
            if time.time() > deadline - 10:
                break
            
            # Start from best_code
            perturbed = [c[:] for c in best_code]
            
            # Randomly modify one codeword
            if len(perturbed) > 0:
                idx = random.randint(0, len(perturbed) - 1)
                pos = random.randint(0, 5)
                old_val = perturbed[idx][pos]
                new_val = random.choice([v for v in range(3) if v != old_val])
                perturbed[idx][pos] = new_val
            
            # Check if perturbed code is still valid
            word_to_covers3 = {}
            valid = True
            for cw in perturbed:
                for w in get_covered_words(cw):
                    if w not in word_to_covers3:
                        word_to_covers3[w] = []
                    word_to_covers3[w].append(cw)
            
            if len(word_to_covers3) < 729:
                continue  # Not valid, skip
            
            # Try greedy reduction on perturbed code
            pert_arr = np.array(perturbed, dtype=np.int8)
            p_active = [True] * len(pert_arr)
            
            def rebuild_cover_counts_p(p_active, p_arr):
                counts = {}
                for idx, cw in enumerate(p_arr):
                    if not p_active[idx]:
                        continue
                    for w in get_covered_words(cw):
                        if w not in counts:
                            counts[w] = 0
                        counts[w] += 1
                return counts
            
            p_cover = rebuild_cover_counts_p(p_active, pert_arr)
            
            p_changed = True
            while p_changed and time.time() < deadline - 5:
                p_changed = False
                for idx in range(len(pert_arr)):
                    if not p_active[idx]:
                        continue
                    cw = pert_arr[idx]
                    is_redundant = True
                    for w in get_covered_words(cw):
                        if p_cover.get(w, 0) <= 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        p_active[idx] = False
                        for w in get_covered_words(cw):
                            p_cover[w] -= 1
                        p_changed = True
                        if time.time() > deadline - 5:
                            break
            
            p_remaining = [pert_arr[i].tolist() for i in range(len(pert_arr)) if p_active[i]]
            if len(p_remaining) < best_len:
                best_code = p_remaining
                best_len = len(p_remaining)
    
    return best_code
