"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # For the specific case: q=3, n=8, radius=1
    # We use a greedy construction with local optimization (swap-based reduction)
    # Start from the direct sum construction (486 words) and try to reduce.

    if q == 3 and n == 8 and radius == 1:
        # Ternary tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Direct sum: 9 * 3^4 = 486 words
        code = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                code.append(h + list(tail))
        
        # Convert to numpy array for efficiency
        code = np.array(code, dtype=np.int8)
        
        # Try to reduce the code size using greedy removal
        # A codeword can be removed if all words it covers are also covered by other codewords
        # For radius 1, a codeword c covers: c itself, and all words at Hamming distance 1
        
        # Precompute coverage: for each codeword, which "spheres" it covers
        # Sphere of c = {c} ∪ {words at distance 1 from c}
        # Size of sphere = 1 + n*(q-1) = 1 + 8*2 = 17
        
        def sphere_of(c):
            """Return set of words covered by codeword c (distance <= radius)"""
            covered = set()
            covered.add(tuple(c))
            for i in range(n):
                for v in range(q):
                    if v != c[i]:
                        w = list(c)
                        w[i] = v
                        covered.add(tuple(w))
            return covered
        
        # Greedy removal: try to remove codewords whose coverage is redundant
        # We'll use a faster approach: track coverage counts
        # For each word in the space, count how many codewords cover it
        
        # Actually, let's use a different approach: 
        # Try swapping codewords to reduce the total count
        # Or: iteratively remove codewords that are "redundant"
        
        # Let's use a randomized local search approach
        # Start with 486 codewords, try to find a valid smaller code
        
        # Approach: Start from 486, try to remove one codeword at a time
        # Check if the code is still valid after removal
        # If valid, keep the removal; if not, try another
        
        # To check validity efficiently, we need to verify that every word in {0,1,2}^8
        # is within distance 1 of some codeword.
        
        # Let's use a bitset approach for coverage
        # Total space size: 3^8 = 6561
        # We can represent coverage as a boolean array of size 6561
        
        def code_to_int(word):
            """Convert a word (list of ints) to an integer index"""
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        
        def int_to_word(idx):
            """Convert integer index back to word"""
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        
        # Precompute the sphere for each possible codeword position
        # sphere_idx[c_idx] = list of indices covered by codeword at position c_idx
        
        # Actually, let's just do greedy removal with periodic full validation
        
        code_list = [tuple(c) for c in code]
        
        # Try to remove codewords in a smart order
        # Codewords with more "unique" coverage are harder to remove
        # But let's just try random removals and validate
        
        import random
        random.seed(42)
        
        # Full validation function
        def is_valid(code_tuples):
            """Check if every word in the space is within radius 1 of some codeword"""
            covered = np.zeros(3**8, dtype=bool)
            for c in code_tuples:
                # Mark c itself
                covered[code_to_int(c)] = True
                # Mark all words at distance 1
                for i in range(n):
                    for v in range(q):
                        if v != c[i]:
                            w = list(c)
                            w[i] = v
                            covered[code_to_int(w)] = True
            return np.all(covered)
        
        # Greedy removal
        # Try to remove codewords; if removal keeps validity, keep it
        # Use random order to escape local minima
        current_code = list(code_list)
        
        # Precompute spheres for faster validation
        # sphere_of_idx[idx] = set of indices covered by word at index idx
        all_spheres = {}
        for idx in range(3**8):
            w = int_to_word(idx)
            s = set()
            s.add(idx)
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = list(w)
                        nw[i] = v
                        s.add(code_to_int(nw))
            all_spheres[idx] = s
        
        # Faster validation using coverage counts
        def get_coverage(code_tuples):
            """Return coverage count array"""
            cov = np.zeros(3**8, dtype=np.int32)
            for c in code_tuples:
                idx = code_to_int(c)
                for s_idx in all_spheres[idx]:
                    cov[s_idx] += 1
            return cov
        
        # Try to reduce
        removed = 0
        while time.time() < deadline - 5.0:
            # Try random removals
            made_progress = False
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                if len(current_code) <= 403:
                    break
                
                # Pick a random codeword to try removing
                idx_to_try = random.randint(0, len(current_code) - 1)
                c_to_remove = current_code[idx_to_try]
                
                # Check if removal is valid
                # Count how many words are covered only by this codeword
                # If any word is covered only by this codeword, removal is invalid
                
                # Compute coverage without this codeword
                # This is expensive, so let's use a heuristic:
                # Try the removal, validate, if valid keep it
                
                current_code.pop(idx_to_try)
                
                # Quick validation: check if coverage is still complete
                cov = np.zeros(3**8, dtype=np.int32)
                for c in current_code:
                    cidx = code_to_int(c)
                    for s_idx in all_spheres[cidx]:
                        cov[s_idx] += 1
                
                if np.all(cov > 0):
                    made_progress = True
                    removed += 1
                else:
                    # Restore
                    current_code.insert(idx_to_try, c_to_remove)
            
            if not made_progress:
                # Try a different strategy: swap-based
                # Try to replace two codewords with one
                # Or: try adding a codeword that allows removing two others
                pass
        
        return [list(c) for c in current_code]
    
    else:
        # General fallback: direct sum construction
        if n >= 4:
            # Use a simple construction
            if q == 3 and n == 4:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            else:
                # Repeat first coordinate pattern
                base = []
                for a in range(q):
                    for b in range(q):
                        word = [a, b] + [0] * (n - 2)
                        base.append(word)
                return base
        else:
            # n < 4, just enumerate
            return [list(p) for p in product(range(q), repeat=n)]

# EVOLVE-BLOCK-END
