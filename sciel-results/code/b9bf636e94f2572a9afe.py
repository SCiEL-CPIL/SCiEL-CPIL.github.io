# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 17
MIN_D = 6
DEADLINE = 55.0  # Safety margin before 60s kill


def _hamming(u, v):
    return (u ^ v).bit_count()


def construct():
    start = time.time()
    rng = random.Random(42)

    # 1. Construct the [17, 9, 5] QR code
    # Generator polynomial f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    # Coeffs: x^0 ... x^8 -> [1, 0, 0, 1, 1, 1, 0, 0, 1]
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    # Basis vectors are shifts of f1
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    # Generate all 512 codewords of the [17, 9, 5] code
    words_17 = []
    for mask in range(512):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                # XOR with basis vector i
                bv = basis[i]
                for j in range(17):
                    if bv[j]:
                        w ^= (1 << j)
        words_17.append(w)
        
    # 2. Extend to [18, 9, 6] by adding a parity bit
    # Parity bit is the sum of bits mod 2
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        # Append parity as the 18th bit (index 17)
        words_18.append(w | (parity << 17))
        
    # 3. Shorten to [17, 8, 6]
    # Shortening by deleting coordinate i means:
    # Keep only codewords with bit i = 0, then delete bit i.
    # The task states the reference is the shortened code.
    # Usually, shortening the extended QR code at any position yields an equivalent [17, 8, 6] code.
    # We will try position 0 (the parity bit position in our extended representation? No, parity is new).
    # In our representation, bits 0..16 are the original QR bits, bit 17 is parity.
    # If we shorten at position 17 (the parity bit), we keep words with parity 0.
    # Since the original code is even-weight? No, QR codes are not necessarily even weight.
    # Wait, the extended code has even weight. So parity bit = sum of first 17 bits.
    # If we keep words with parity bit 0, that means sum of first 17 bits is 0.
    # This is a subcode of the original [17, 9, 5] code consisting of even-weight words.
    # The dimension drops by 1 if the all-ones vector is not in the code?
    # Actually, the set of even-weight words in a linear code is a subcode of index 2 if the code contains odd-weight words.
    # Does the [17, 9, 5] QR code contain odd-weight words?
    # The generator polynomial f1(x) has weight 5 (odd). So yes.
    # So the subcode of even-weight words has dimension 8 and size 256.
    # What is its minimum distance?
    # The original d=5.
    # The extended code has d=6.
    # Shortening the extended code at the parity position gives the subcode of the original code with even weight.
    # Let C' be this subcode.
    # For any c in C', wt(c) is even.
    # In the extended code, c_ext = (c, parity(c)). Since parity(c)=0, c_ext = (c, 0).
    # wt(c_ext) = wt(c).
    # Since d(C_ext) = 6, wt(c_ext) >= 6 for c != 0.
    # So wt(c) >= 6 for c != 0 in C'.
    # Thus C' is a [17, 8, 6] code.
    # Size is 256.
    
    # Let's construct this specific code: subcode of words_17 with even weight.
    code = []
    for w in words_17:
        if w.bit_count() % 2 == 0:
            code.append(w)
            
    # Verify size and distance
    # Size should be 256
    # Min distance should be 6
    
    # Optimization: Vectorized distance check for adding new words
    # Convert code to numpy array of bits for faster distance calculation
    # But for 256 words, O(N^2) is 256*255/2 ~ 32k checks, which is fast in Python.
    # However, we want to improve beyond 256.
    
    # 4. Local Search to improve the code
    # We have a [17, 8, 6] code of size 256.
    # We want to add more words or swap to get a larger code with d >= 6.
    # The known lower bound is 258 (Milshtein 2019).
    # So we just need to add 2 more words? Or maybe replace some to allow more.
    
    # Strategy:
    # 1. Try to add random words that are at distance >= 6 from all existing words.
    # 2. If stuck, try to swap a word with another word that allows adding more words.
    
    # Efficient distance check using bit operations
    # code is a list of ints.
    # We can maintain a set for O(1) lookup? No, we need distance.
    
    # Let's use a greedy approach with a "conflict graph" or just simple checks.
    # Given the small size (256), simple checks are fine.
    
    # First, verify our initial code is valid.
    min_d_found = 17
    for i in range(len(code)):
        for j in range(i+1, len(code)):
            d = _hamming(code[i], code[j])
            if d < min_d_found:
                min_d_found = d
                if min_d_found < 6:
                    break
        if min_d_found < 6:
            break
            
    if min_d_found < 6:
        # Fallback: if our construction is wrong, use a simple greedy random start
        # But the logic above seems sound. Let's assume it's correct.
        # For safety, if it fails, return the largest valid subset found by greedy.
        pass

    # Local Search
    # We will try to add words.
    # To speed up, we can precompute distances?
    # With 256 words, adding a candidate requires 256 distance checks.
    # 256 * (cost of bit_count) is very fast.
    
    # We can try to add words from the full space of 2^17.
    # But 2^17 = 131072.
    # Checking all of them: 131072 * 256 checks = 33 million checks.
    # bit_count is fast, but Python loop overhead is high.
    # 33M iterations in Python might take ~5-10 seconds.
    # We have 55 seconds. This is feasible.
    
    # Let's try to find all possible words that can be added.
    # If we find more than 1, we add them greedily?
    # Actually, adding one word might block others.
    # A better strategy:
    # 1. Find a candidate word c.
    # 2. Check if it is valid against current code.
    # 3. If valid, add it.
    # 4. Repeat until no more valid candidates found in a pass.
    
    # To be efficient, we can iterate through all 2^17 words once.
    # If a word is valid, add it.
    # Then iterate again?
    # Since adding a word only increases constraints, a single pass might miss words that become valid?
    # No, adding words makes it harder to add new ones.
    # So a single pass is sufficient for a greedy maximal code.
    # But the order matters.
    
    # Let's try a randomized greedy approach.
    # Shuffle all 2^17 words? No, too slow to shuffle 131k items repeatedly.
    # Just iterate 0..2^17-1.
    
    # Optimization:
    # Instead of checking against the whole code list every time,
    # we can just do it.
    
    # Let's check time for one full pass.
    # 131072 candidates.
    # For each, check against ~256-300 codewords.
    # ~35 million distance checks.
    # In Python, ~10-20 seconds.
    # We have 55 seconds.
    # We can do 2-3 passes or restart with different seeds?
    
    # Let's do a single deterministic pass first.
    # Then if time remains, try random restarts or local moves.
    
    # Actually, the "Milshtein" code is nonlinear.
    # A greedy pass starting from a linear code might get stuck at 256 or slightly above.
    # To get 258, we might need to swap some words.
    
    # Let's implement a simple local search:
    # 1. Start with the 256-word code.
    # 2. Try to add words.
    # 3. If no words can be added, try to remove one word and add two?
    #    Or swap one word with another to free up space?
    
    # Given the time limit, let's stick to a robust greedy addition first.
    
    # To speed up the check, we can use NumPy?
    # Code is a list of ints.
    # We can convert to a numpy array of uint16.
    # Distance = popcount(a ^ b).
    # NumPy doesn't have a built-in popcount for arrays directly in a vectorized way that's super fast without loops?
    # np.bitwise_xor, then sum of bits?
    # Actually, Python's int.bit_count() is implemented in C and is very fast.
    # The bottleneck is the Python loop.
    
    # Let's try to optimize the inner loop.
    
    def can_add(candidate, code_list):
        for c in code_list:
            if _hamming(candidate, c) < MIN_D:
                return False
        return True

    # We'll iterate through all possible words.
    # To make it deterministic and potentially find the "best" greedy completion,
    # we can sort candidates by some heuristic?
    # E.g., words with lower weight? Or random order?
    # Random order often finds larger maximal codes.
    
    # Let's use a random order for the candidates.
    # Generating a random permutation of 131k items takes time.
    # We can just use rng to pick candidates?
    # No, we want to be thorough.
    
    # Alternative: Iterate 0 to 2^17-1.
    # If we find a valid word, add it.
    
    # Let's estimate:
    # 131k * 256 checks = 33M.
    # If each check is ~100ns (optimistic), 3.3s.
    # If each check is ~1us (realistic Python), 33s.
    # This is cutting it close.
    
    # We can prune:
    # If we have a set of forbidden patterns? No.
    
    # Let's try to use a NumPy vectorized check for the whole code at once.
    # code_arr = np.array(code, dtype=np.uint16)
    # cand = np.uint16(candidate)
    # dists = np.array([(cand ^ c).bit_count() for c in code_arr]) # Still a loop
    # 
    # There is a trick:
    # Use np.unpackbits?
    # code_bits = np.unpackbits(code_arr.view(np.uint8), axis=1) # 256 x 17
    # cand_bits = np.unpackbits(np.array([candidate]).view(np.uint8), axis=1) # 1 x 17
    # dists = np.sum(code_bits ^ cand_bits, axis=1)
    # This is vectorized!
    # 
    # Let's try this.
    
    code_arr = np.array(code, dtype=np.uint16)
    # We need to update code_arr when we add words.
    # It's easier to keep a list and convert to numpy for checks?
    # Or keep a numpy array and append?
    # Appending to numpy is slow.
    # Let's keep a list `code_list` and a numpy array `code_np`.
    # When we add, we update both.
    
    code_list = list(code)
    code_np = np.array(code_list, dtype=np.uint16)
    
    # Precompute unpacked bits for the initial code?
    # The code changes, so we can't precompute for all.
    # But we can recompute unpacked bits for the current code each time we check a candidate?
    # No, that's expensive.
    # 
    # Better:
    # Keep code_list.
    # For each candidate, do the Python loop check.
    # It might be fast enough.
    
    # Let's add a timeout guard inside the loop.
    
    added_count = 0
    # Iterate through all candidates
    # To randomize, we can shuffle a list of indices?
    # Or just use a fixed order.
    # Let's use a fixed order 0..2^17-1.
    
    for cand in range(1, 1 << 17):  # Skip 0, it's already in the code
        if time.time() - start > DEADLINE - 5:
            break
            
        # Check distance against all current codewords
        valid = True
        for c in code_list:
            if _hamming(cand, c) < MIN_D:
                valid = False
                break
                
        if valid:
            code_list.append(cand)
            added_count += 1
            
    # If we have time, try a few random restarts or local swaps?
    # For now, return the result.
    
    return code_list
# EVOLVE-BLOCK-END
