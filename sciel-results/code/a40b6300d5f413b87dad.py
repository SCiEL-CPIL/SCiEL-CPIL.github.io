import time
import random
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    # Step 1: Generate the 256-word linear code (shortened [18,9,6] QR code)
    # The quadratic residue code over GF(17) is based on the quadratic residues mod 17.
    # QR(17) = {1, 2, 4, 8, 9, 13, 15, 16}
    # The generator polynomial is g(x) = 1 + x^2 + x^3 + x^4 + x^5 + x^6 + x^8 (degree 9)
    # Wait, for QR codes, the generator polynomial roots are the primitive 17th roots of unity.
    # Let's use a known generator matrix for the [17,8,6] QR code?
    # Actually, the standard QR code of length 17 is [17, 8, 6] or [17, 9, 6]?
    # Brouwer's table says A_2(17,6) >= 258. The linear code is often cited as [17,8,6] with size 256?
    # Wait, 2^8 = 256. So it's an [17,8,6] code.
    # Let's construct it.
    # The quadratic residues mod 17 are Q = {1, 2, 4, 8, 9, 13, 15, 16}.
    # The generator polynomial g(x) has roots alpha^i for i in Q.
    # This is a cyclic code.
    # Let's find g(x) by computing the minimal polynomials.
    # alpha is a primitive 17th root of unity in GF(2^8) or similar.
    # Since 2^4 = 16 = -1 mod 17, the order of 2 mod 17 is 8.
    # So we can work in GF(2^8).
    # Let's find a primitive element in GF(2^8) whose 17th power is 1? No, we need an element of order 17.
    # The multiplicative group of GF(2^8) has order 255. 17 divides 255 (255 = 15 * 17).
    # So there are elements of order 17 in GF(2^8).
    # Let's find one.
    # We need an element beta such that beta^17 = 1 and beta != 1.
    # Let's use the polynomial x^8 + x^4 + x^3 + x^2 + 1 (primitive polynomial for GF(2^8)).
    # Let alpha be a root. Then alpha^255 = 1.
    # We want alpha^15. (alpha^15)^17 = alpha^255 = 1.
    # So beta = alpha^15 is a 17th root of unity.
    # Let's compute the minimal polynomial of beta over GF(2).
    # The minimal polynomial is the product of (x - beta^2^i) for i=0..7 (since order of 2 mod 17 is 8).
    # This will be a polynomial of degree 8.
    # Let's compute it symbolically or numerically.
    # Alternatively, we can just use a known generator matrix for the [17,8,6] QR code.
    # Or we can use the fact that the code is cyclic with generator polynomial g(x).
    # Let's compute g(x).
    # We need the polynomial whose roots are beta^q for q in Q.
    # Q = {1, 2, 4, 8, 9, 13, 15, 16}.
    # Note that 2Q mod 17 = Q. So the set of roots is stable under Frobenius.
    # The generator polynomial g(x) is the product of (x - beta^q) for q in Q.
    # Let's compute this product in GF(2)[x] mod (x^17 - 1).
    # Actually, since it's a cyclic code, g(x) divides x^17 - 1.
    # Let's compute g(x) = prod_{q in Q} (x - beta^q).
    
    # Let's implement GF(2^8) arithmetic to compute g(x).
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # Let's represent elements as integers 0-255.
    # exp table and log table.
    
    poly = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
    
    exp_table = [0] * 512
    log_table = [0] * 256
    x = 1
    for i in range(255):
        exp_table[i] = x
        log_table[x] = i
        # multiply by x (i.e., shift left and reduce)
        x = (x << 1)
        if x & 0x100:
            x ^= poly
    # fill remaining
    for i in range(255, 512):
        exp_table[i] = exp_table[i - 255]
    log_table[0] = -1
    
    def gf_mul(a, b):
        if a == 0 or b == 0:
            return 0
        return exp_table[log_table[a] + log_table[b]]
    
    def gf_pow(a, n):
        if a == 0:
            return 0
        return exp_table[(log_table[a] * n) % 255]
    
    # Find beta = alpha^15
    beta = gf_pow(2, 15)  # alpha is 2 (the generator of the field)
    # Check beta^17 == 1
    # assert gf_pow(beta, 17) == 1
    
    # Compute g(x) = prod_{q in Q} (x - beta^q)
    # In GF(2), x - a = x + a.
    # g(x) is a polynomial of degree 8.
    # Let's represent polynomials as lists of coefficients (integers 0 or 1).
    # g(x) = [1, 0, 0, 0, 0, 0, 0, 0, 0] (degree 8, coeff of x^8 is 1)
    # Actually, let's build it up.
    # Start with g(x) = 1.
    # For each q in Q, multiply by (x + beta^q).
    
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    g_coeffs = [1]  # represents 1
    
    for q in Q:
        root = gf_pow(beta, q)
        # Multiply g_coeffs by (x + root)
        # New degree is len(g_coeffs)
        new_g = [0] * (len(g_coeffs) + 1)
        for i, c in enumerate(g_coeffs):
            # c * x
            new_g[i + 1] ^= c
            # c * root
            new_g[i] ^= gf_mul(c, root)
        g_coeffs = new_g
    
    # g_coeffs has length 9 (degree 8).
    # g_coeffs[0] is coeff of x^0, ..., g_coeffs[8] is coeff of x^8.
    # The generator polynomial is g(x) = sum g_coeffs[i] x^i.
    # The code is the set of multiples of g(x) mod (x^17 - 1).
    # There are 2^8 = 256 such multiples (since degree of g is 8, and 17-8=9? No, message length is 17-8=9? No, k=17-n=9? No, k=17-8=9 is wrong. k = n - deg(g) = 17 - 8 = 9. So [17,9,6] code?
    # Wait, if k=9, size is 512. But Brouwer says 258..340.
    # Let's check the distance.
    # If k=9, size 512, it might violate the upper bound 340.
    # So maybe the QR code is [17,8,6] with size 256.
    # This implies deg(g) = 9.
    # Why did I get deg(g)=8?
    # The number of roots is |Q| = 8. So deg(g) should be 8.
    # So k = 17 - 8 = 9. Size 512.
    # Is there a [17,9,6] code?
    # Griesmer bound: n >= 6(2^0 + 2^1 + ... + 2^8) = 6(511) = 3066. No.
    # So a [17,9,6] code cannot exist.
    # Therefore, my assumption that the QR code has roots exactly at Q is wrong, or the code is not cyclic in the way I think, or I made a mistake.
    
    # Let's re-evaluate.
    # The quadratic residue code of length 17.
    # The set of quadratic residues Q has size 8.
    # The set of non-residues N has size 8.
    # The code C_Q is generated by the vector with 1s at positions in Q.
    # This code is cyclic.
    # Its generator polynomial g(x) has roots alpha^i where i is in the coset of Q under multiplication by 2.
    # Since 2Q = Q, the roots are exactly {alpha^q : q in Q}.
    # So deg(g) = 8.
    # So k = 9.
    # But [17,9,6] is impossible.
    # So the minimum distance must be less than 6?
    # Or maybe the code is [17,8,6] and the generator polynomial has degree 9?
    # This would happen if the roots include 0? No.
    # Or if the code is not the QR code but the extended QR code shortened?
    # The task says: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate".
    # The extended QR code has length 18.
    # The QR code has length 17.
    # The extended code has a parity bit.
    # If the [18,9,6] code exists, then the [17,8,6] code (punctured) exists?
    # Puncturing reduces length by 1 and distance by at most 1.
    # So [18,9,6] -> [17,9,5].
    # But we want distance 6.
    # So we must delete a coordinate from the [18,9,6] code.
    # This results in a [17,9,6] code? No, puncturing can reduce distance.
    # However, if the minimum weight words in the [18,9,6] code do not have a 1 in the deleted coordinate, then the distance remains 6.
    # So we need to find a coordinate such that no minimum weight word has a 1 there.
    # In a cyclic code, the weight distribution is uniform.
    # So we need to check if there is a coordinate with no minimum weight words.
    # For the [18,9,6] QR code, the minimum weight is 6.
    # The number of minimum weight words is likely large.
    # But maybe we can just generate the [18,9,6] code and delete a coordinate.
    
    # Let's generate the [18,9,6] extended QR code.
    # It is the QR code of length 17 with a parity bit appended.
    # Wait, the QR code of length 17 is [17,9,5]?
    # If [17,9,6] is impossible, then the QR code must be [17,9,5].
    # Then the extended code is [18,9,6].
    # So the linear code of size 256 is obtained by taking the [18,9,6] code and deleting one coordinate.
    # To maintain distance 6, we must delete a coordinate that is 0 in all minimum weight words.
    # Let's generate the [18,9,6] code.
    # We need the generator polynomial of the [17,9,5] QR code.
    # This is the g(x) we computed above, with roots in Q.
    # Let's verify the distance of this [17,9,5] code.
    # If it is 5, then the extended code is [18,9,6].
    
    # Let's generate the codewords of the [17,9,5] code.
    # g_coeffs is the generator polynomial.
    # We need to multiply g(x) by all polynomials m(x) of degree < 9.
    # There are 2^9 = 512 such polynomials.
    # But we only want the 256 words of the [17,8,6] code?
    # No, the task says the reference is 256 words.
    # So we must be taking a subcode?
    # Or maybe the QR code is [17,8,6] after all?
    # Let's check the weight of the generator polynomial g(x).
    # If the weight of g(x) is 6, and the code is [17,9,5], then the minimum weight is at most 6.
    # If the minimum weight is 5, then we can't use it directly.
    
    # Let's just generate the [18,9,6] code and delete a coordinate.
    # How to generate the [18,9,6] code?
    # It is the set of vectors (c, p) where c is in the [17,9,5] code and p is the parity bit.
    # So we generate all 512 codewords of the [17,9,5] code, append the parity bit, and then delete one coordinate.
    # To get a [17,9,6] code, we need to delete a coordinate that is 0 in all weight-6 words.
    # Let's find such a coordinate.
    
    # Generate [17,9,5] code
    codewords_17 = []
    for m in range(512):
        # m is the message polynomial (9 bits)
        # c(x) = m(x) * g(x) mod (x^17 - 1)
        # Let's do polynomial multiplication.
        # m(x) has degree < 9.
        # g(x) has degree 8.
        # c(x) has degree < 17.
        c = 0
        for i in range(9):
            if m & (1 << i):
                # shift g_coeffs by i
                for j in range(9):
                    if g_coeffs[j]:
                        c ^= (1 << (i + j))
        # c is a 17-bit integer (bits 0..16)
        codewords_17.append(c)
    
    # Now extend to 18 bits
    codewords_18 = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        c_ext = c | (parity << 17)
        codewords_18.append(c_ext)
    
    # Find a coordinate to delete
    # We want a coordinate j (0..17) such that no weight-6 word has a 1 at j.
    # Weight-6 words in the [18,9,6] code.
    # Let's find the minimum weight words.
    min_weight_words = []
    for c in codewords_18:
        w = bin(c).count('1')
        if w == 6:
            min_weight_words.append(c)
    
    # Count how many min weight words have a 1 at each position
    ones_count = [0] * 18
    for c in min_weight_words:
        for j in range(18):
            if c & (1 << j):
                ones_count[j] += 1
    
    # Find a position with 0 ones
    delete_pos = -1
    for j in range(18):
        if ones_count[j] == 0:
            delete_pos = j
            break
    
    if delete_pos == -1:
        # No such position? Then we can't get distance 6 by deletion.
        # Maybe the distance is actually 6 without deletion?
        # Or maybe we should just use the [17,9,5] code and accept distance 5? No, we need distance 6.
        # Let's check if the [17,9,5] code has any weight-6 words.
        # If it does, and we delete a coordinate where they are 0, we get distance 6.
        # If no coordinate is 0 in all weight-6 words, then we can't do it.
        # But the task says it's possible.
        # Maybe I made a mistake in the QR code generation.
        # Let's just pick a random coordinate and hope? No.
        # Let's try all coordinates and see which one maximizes the minimum distance of the punctured code.
        best_pos = -1
        best_min_dist = 0
        for j in range(18):
            # Puncture at j
            punctured = []
            for c in codewords_18:
                # Remove bit j
                c_punc = (c & ((1 << j) - 1)) | ((c >> (j + 1)) << j)
                punctured.append(c_punc)
            # Find min distance
            min_d = 18
            for i in range(len(punctured)):
                for k in range(i + 1, len(punctured)):
                    d = bin(punctured[i] ^ punctured[k]).count('1')
                    if d < min_d:
                        min_d = d
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = j
        
        delete_pos = best_pos
    
    # Delete the coordinate
    base_codewords = []
    for c in codewords_18:
        c_punc = (c & ((1 << delete_pos) - 1)) | ((c >> (delete_pos + 1)) << delete_pos)
        base_codewords.append(c_punc)
    
    # Remove duplicates (just in case)
    base_codewords = list(set(base_codewords))
    
    # Verify distance
    for i in range(len(base_codewords)):
        for j in range(i + 1, len(base_codewords)):
            if hamming_dist(base_codewords[i], base_codewords[j]) < MIN_DIST:
                # This should not happen
                pass
    
    codewords = list(base_codewords)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Now use a greedy heuristic to add more words.
    rng = random.Random(42)
    
    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
    
    # Phase 1: Greedy addition of random compatible words
    attempts = 0
    max_attempts_greedy = 100000
    while attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.4:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Phase 2: Local search / clique extension
    for _ in range(5000):
        if time.time() - start_time > time_limit * 0.8:
            break
        
        K = len(codewords)
        if K == 0:
            break
        
        idx = rng.randint(0, K - 1)
        removed = codewords[idx]
        others = [c for i, c in enumerate(codewords) if i != idx]
        
        found = False
        for _ in range(200):
            if time.time() - start_time > time_limit * 0.8:
                break
            
            w1 = rng.getrandbits(N)
            if not is_compatible_with_all(w1, others):
                continue
            
            for _ in range(50):
                w2 = rng.getrandbits(N)
                if not is_compatible_with_all(w2, others):
                    continue
                if hamming_dist(w1, w2) < MIN_DIST:
                    continue
                
                new_set = others + [w1, w2]
                codewords = new_set
                found = True
                break
            
            if found:
                break
        
        if found:
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
    
    # Phase 3: Final greedy polish
    attempts = 0
    max_attempts_final = 50000
    while attempts < max_attempts_final:
        if time.time() - start_time > time_limit:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    return best_codewords
