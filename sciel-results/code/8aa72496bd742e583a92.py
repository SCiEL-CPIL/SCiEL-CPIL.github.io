"""Improved program for K_3(8,1) ternary covering code."""
import time
import itertools
import random
import numpy as np

# Deadline guard
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    # The seed gives 9 * 3**(n-4) = 486 words for q=3, n=8.
    # We aim to do better (fewer words) or at least match.
    # Best known is 486, lower bound 403.
    # Let's try to find a code with fewer than 486 words.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct product construction
        # For general case, use a simple construction
        # Perfect code for q=3, n=4: tetracode has 9 words
        # For other parameters, use greedy or fallback
        return _fallback_construct(q, n, radius)
    
    # Specialized for q=3, n=8, radius=1
    # Space size: 3^8 = 6561
    # Each codeword covers 1 + 8*2 = 17 words
    # Lower bound: ceil(6561/17) = 386
    # Best known: 486
    
    # Strategy: Start with the 486-word code, then try to reduce via local search
    # or use a smarter construction.
    
    # First, generate the base 486-word code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Convert to numpy for faster operations
    code = np.array(base_code, dtype=np.int8)
    
    # Try to improve via local search: remove a codeword if all its covered words
    # are still covered by others, then add a new codeword that covers many uncovered words.
    
    # Build coverage structure
    # For each word in the space, track which codewords cover it
    # Space is 3^8 = 6561 words
    space_size = 3 ** 8
    covered_by = [0] * space_size  # count of codewords covering each word
    
    def word_to_int(w):
        """Convert word (list of ints 0-2) to integer 0..3^8-1"""
        val = 0
        for i, x in enumerate(w):
            val = val * 3 + x
        return val
    
    def int_to_word(val):
        """Convert integer to word"""
        w = []
        for _ in range(8):
            w.append(val % 3)
            val //= 3
        w.reverse()
        return w
    
    # Precompute neighbors: for each word, list of words at distance <= 1
    # This is 6561 * 17 = ~111k entries
    neighbors = []
    for val in range(space_size):
        w = int_to_word(val)
        nbrs = [val]
        for i in range(8):
            for d in range(1, 3):
                w2 = w[:]
                w2[i] = (w[i] + d) % 3
                nbrs.append(word_to_int(w2))
        neighbors.append(nbrs)
    
    # Initial coverage
    for ci, cw in enumerate(code):
        ci_int = word_to_int(cw)
        for nb in neighbors[ci_int]:
            covered_by[nb] += 1
    
    # Local search: try to remove codewords
    # A codeword can be removed if for all words it covers, covered_by[word] > 1
    improved = True
    while improved and _time_left() > 10:
        improved = False
        # Shuffle order for diversity
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if _time_left() < 15:
                break
            cw = code[idx]
            cw_int = word_to_int(cw)
            
            # Check if removing this codeword leaves all words covered
            removable = True
            for nb in neighbors[cw_int]:
                if covered_by[nb] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove this codeword
                for nb in neighbors[cw_int]:
                    covered_by[nb] -= 1
                code = np.delete(code, idx, axis=0)
                improved = True
                break  # Restart search after removal
    
    # Now try to add codewords to cover any uncovered words (shouldn't be any)
    # and potentially restructure to allow more removals
    
    # Try a different approach: greedy set cover with randomization
    # Actually, let's try to find a smaller code from scratch using greedy
    
    # Greedy set cover
    def greedy_cover():
        uncovered = set(range(space_size))
        code_list = []
        
        while uncovered and _time_left() > 5:
            # Find codeword that covers most uncovered words
            best_count = -1
            best_word = None
            
            # Sample a subset of candidate positions for speed
            # Try all 6561 positions
            for pos in range(space_size):
                if _time_left() < 10:
                    break
                count = 0
                for nb in neighbors[pos]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = pos
            
            if best_word is None:
                break
            
            uncovered.difference_update(neighbors[best_word])
            code_list.append(int_to_word(best_word))
        
        return code_list
    
    # Try greedy and compare
    greedy_code = greedy_cover()
    if len(greedy_code) < len(code):
        code = np.array(greedy_code, dtype=np.int8)
    else:
        # Use the locally searched version
        pass
    
    # Final local search on the best code
    # Rebuild covered_by
    covered_by = [0] * space_size
    for ci, cw in enumerate(code):
        ci_int = word_to_int(cw)
        for nb in neighbors[ci_int]:
            covered_by[nb] += 1
    
    # Try to remove more
    improved = True
    while improved and _time_left() > 5:
        improved = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if _time_left() < 10:
                break
            cw = code[idx]
            cw_int = word_to_int(cw)
            
            removable = True
            for nb in neighbors[cw_int]:
                if covered_by[nb] <= 1:
                    removable = False
                    break
            
            if removable:
                for nb in neighbors[cw_int]:
                    covered_by[nb] -= 1
                code = np.delete(code, idx, axis=0)
                improved = True
                break
    
    # Convert back to list of lists
    result = [list(cw) for cw in code]
    
    # Verify validity
    # Quick check: every word should be within distance 1 of some codeword
    # This is expensive, so only do a sample check if time permits
    if _time_left() > 20:
        # Full verification
        covered = set()
        for cw in result:
            ci_int = word_to_int(cw)
            for nb in neighbors[ci_int]:
                covered.add(nb)
        if len(covered) != space_size:
            # Invalid, fall back to base code
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    return result


def _fallback_construct(q, n, radius):
    """Fallback for non-standard parameters."""
    # Simple greedy set cover
    space_size = q ** n
    
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val
    
    def int_to_word(val):
        w = []
        for _ in range(n):
            w.append(val % q)
            val //= q
        w.reverse()
        return w
    
    # Precompute neighbors
    neighbors = []
    for val in range(space_size):
        w = int_to_word(val)
        nbrs = [val]
        for i in range(n):
            for d in range(1, q):
                if radius >= 1:
                    w2 = w[:]
                    w2[i] = (w[i] + d) % q
                    nbrs.append(word_to_int(w2))
        neighbors.append(nbrs)
    
    uncovered = set(range(space_size))
    code_list = []
    
    while uncovered and _time_left() > 5:
        best_count = -1
        best_word = None
        for pos in range(space_size):
            count = 0
            for nb in neighbors[pos]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = pos
        
        if best_word is None:
            break
        
        uncovered.difference_update(neighbors[best_word])
        code_list.append(int_to_word(best_word))
    
    return code_list
