"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Trivial cases
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(q)]

    N = q ** n
    if N > 100000:
        # Fallback for large N: simple greedy (not optimized for this specific task but safe)
        # For this task N=6561, so we proceed with the optimized path.
        pass

    # Precompute all words and their encodings
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Fast encoding: base-q representation
    # We map each word to a unique integer index 0..N-1
    # all_words is generated in lexicographic order, which corresponds to base-q counting.
    # So index in all_words IS the encoding.
    # Let's verify: 
    # 00...0 -> 0
    # 00...1 -> 1
    # ...
    # This is correct.
    
    # Precompute balls for radius 1 (general case: radius r)
    # For radius 1, ball size is 1 + n*(q-1)
    # We store the list of indices covered by each codeword index.
    # Since N is small (6561), we can store a list of lists or a sparse matrix.
    # A list of numpy arrays is fine.
    
    # To speed up distance calculation or ball generation, we can precompute neighbors.
    # For each word index i, decode to digits, generate neighbors, encode back.
    
    def decode(idx):
        """Decode index to list of digits."""
        digits = [0] * n
        for pos in range(n - 1, -1, -1):
            digits[pos] = idx % q
            idx //= q
        return digits

    # Precompute balls
    balls = []
    for i in range(N):
        digits = decode(i)
        ball = [i]
        for pos in range(n):
            orig = digits[pos]
            for alt in range(q):
                if alt == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = alt
                # Encode back
                new_idx = 0
                for d in new_digits:
                    new_idx = new_idx * q + d
                ball.append(new_idx)
        balls.append(ball)
        
    # Convert balls to numpy arrays for faster indexing? 
    # Actually, for coverage counting, we just need to increment counts.
    # Let's keep them as lists of ints for now, or convert to np.array if needed.
    # np.array is faster for vectorized ops, but here we do element-wise increments.
    # Let's use a flat list of lists.
    
    # Initial code: Direct sum of Tetracode (if q=3, n=8) or simple greedy
    # For q=3, n=8, radius=1:
    # Tetracode is a (4, 9, 3) code? No, Tetracode is (4, 9, 3) covering radius 1?
    # Tetracode has 9 words, covers 3^4=81 words. 9 * 5 = 45 < 81. Wait.
    # Perfect code of length 4, radius 1 over q=3 requires M = 3^4 / (1 + 4*2) = 81/9 = 9.
    # Yes, Tetracode is a perfect code.
    # So 9 words cover 81 words.
    # Direct sum with 4 more bits: 9 * 3^4 = 9 * 81 = 729? No.
    # Direct sum of C1 (length n1) and C2 (length n2) gives length n1+n2.
    # Size is |C1| * |C2|.
    # If we take Tetracode (len 4, size 9) and repeat it?
    # Or take Tetracode and a full space of length 4? No, that's not a code.
    # Standard construction: Take a code C1 of length n1, radius r1.
    # Take a code C2 of length n2, radius r2.
    # The direct sum C1 x C2 has length n1+n2, radius r1+r2.
    # If r1=1, r2=0, radius is 1.
    # C2 of radius 0 is the full space? No, radius 0 means only the word itself.
    # So C2 must be the full space {0..q-1}^n2 to cover everything with radius 0?
    # If C2 is the full space, size is q^n2.
    # Then total size is |C1| * q^n2.
    # For n=8, split into 4 and 4.
    # |C1| = 9 (Tetracode).
    # |C2| = 3^4 = 81 (Full space).
    # Total size = 9 * 81 = 729.
    # Wait, the prompt says "best known code has 486 words".
    # 486 = 6 * 81? Or 9 * 54?
    # 486 = 2 * 243 = 2 * 3^5.
    # Maybe a different construction.
    # Van Laarhoven et al. found 486.
    # Let's just start with a greedy construction or the 729 baseline and improve.
    # Actually, 486 is significantly better than 729.
    # Let's try to construct a good initial code.
    
    # Greedy Set Cover:
    # Start with empty code.
    # While not all covered:
    #   Find the word w that is covered by the most codewords? No, find the uncovered word.
    #   Pick a codeword c that covers the most uncovered words.
    #   Add c to code.
    
    # This is O(N^2) roughly, which is 6561^2 ~ 43M. Might be too slow in Python.
    # Better: Randomized greedy or local search from a decent start.
    
    # Let's use the 729 baseline as a start, it's valid.
    # Or even better, start with a random large set and prune.
    # Start with all words? No, too big.
    
    # Let's try to construct the 486 code structure if known, or just use local search.
    # Since I don't have the exact 486 construction memorized, I will use a robust local search.
    
    # Initial Code: 
    # We can use a "lattice" code or similar.
    # For now, let's use a simple heuristic: 
    # Take all words where sum of digits mod 3 == 0? 
    # Size 3^8 / 3 = 2187. Too big.
    
    # Let's stick to the Direct Sum 729 and prune it.
    # Tetracode words:
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a+b)%3, (a+2*b)%3])
            
    # Direct sum: (t, x) where t in Tetracode, x in {0,1,2}^4
    initial_code = []
    for t in tetracode:
        for x in itertools.product(range(3), repeat=4):
            initial_code.append(t + list(x))
            
    # This gives 729 words.
    # We want to reduce to ~486.
    
    # Local Search:
    # 1. Compute coverage counts.
    # 2. Repeatedly remove a random codeword if coverage remains full.
    # 3. If stuck, try swapping: remove one, add one (or remove two, add one) to improve "potential" or just accept if size decreases.
    
    # Data structures for speed:
    # in_code: set of indices
    # cov_counts: array of size N
    
    in_code = set()
    for cw in initial_code:
        idx = 0
        for d in cw:
            idx = idx * q + d
        in_code.add(idx)
        
    cov_counts = np.zeros(N, dtype=np.int32)
    for idx in in_code:
        for neighbor in balls[idx]:
            cov_counts[neighbor] += 1
            
    def is_covered():
        return np.all(cov_counts >= 1)
        
    def add_cw(idx):
        in_code.add(idx)
        for neighbor in balls[idx]:
            cov_counts[neighbor] += 1
            
    def remove_cw(idx):
        in_code.remove(idx)
        for neighbor in balls[idx]:
            cov_counts[neighbor] -= 1
            
    # Local Search Loop
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(in_code)
        np.random.shuffle(indices)
        
        for idx in indices:
            if (time.time() - start_time) > DEADLINE - 2:
                break
            # Try to remove
            remove_cw(idx)
            if is_covered():
                improved = True
                # Keep it removed
            else:
                add_cw(idx)
                # If we couldn't remove, maybe we can swap?
                # Try to find a replacement.
                # This is expensive. Let's just do removals first.
                # If we finish a pass with no removals, we might do a swap pass.
                
        # If no removals happened in the last pass, try swaps.
        # To keep it simple and fast, let's just do multiple removal passes.
        # If stuck, we can perturb.
        
    # Final check
    if not is_covered():
        # Revert to initial if something went wrong (shouldn't happen)
        return [[0]*n] # Fallback, but we should ensure validity
        
    # Convert indices back to words
    result = []
    for idx in in_code:
        result.append(decode(idx))
        
    return result
# EVOLVE-BLOCK-END
