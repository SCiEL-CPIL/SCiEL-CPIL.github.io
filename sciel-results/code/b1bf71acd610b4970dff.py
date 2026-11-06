import time
import random
from itertools import combinations


def construct():
    N = 18
    D = 6
    bc = int.bit_count

    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # --- Reed-Muller [16,5,8] code, extended by 2 zero bits -> 32 words, dist 8 ---
    def reed_muller_words():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # --- Extended QR(17) code construction ---
    def construct_qr17_extended():
        """Construct the extended quadratic residue code QR(17).
        
        QR(17) over GF(2) is a [17,11,5] BCH code.
        Extended by one parity bit -> [18,11,6] with 2048 codewords.
        """
        # GF(2^8) arithmetic with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D)
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                a <<= 1
                if a & 256:
                    a ^= 0x11D
                b >>= 1
            return res

        def gf_pow(a, e):
            res = 1
            while e:
                if e & 1:
                    res = gf_mul(res, a)
                a = gf_mul(a, a)
                e >>= 1
            return res

        # Find a primitive 17th root of unity in GF(2^8)
        # 2^8 - 1 = 255 = 17 * 15, so elements of order 17 exist
        alpha = None
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and gf_pow(candidate, 1) != 1:
                # Check it's truly order 17 (not a divisor)
                # 17 is prime, so if candidate^17=1 and candidate!=1, order is 17
                alpha = candidate
                break
        
        if alpha is None:
            return []  # Fallback
        
        # Quadratic residues mod 17
        residues = [1, 2, 4, 8, 16, 15, 13, 9]
        
        # Compute generator polynomial g(x) = prod_{q in residues} (x + alpha^q)
        poly = [1]  # Start with 1
        for q in residues:
            root = gf_pow(alpha, q)
            new_poly = [0] * (len(poly) + 1)
            for i, c in enumerate(poly):
                new_poly[i] ^= c
                new_poly[i+1] ^= gf_mul(c, root)
            poly = new_poly
        
        # The generator polynomial should have coefficients in GF(2)
        g_coeffs = []
        valid = True
        for c in poly:
            if c in (0, 1):
                g_coeffs.append(c)
            else:
                valid = False
                break
        
        if not valid or len(g_coeffs) != 7:
            return []  # Fallback if construction is invalid
        
        # Build generator matrix for [17,11] code
        gen_matrix = []
        for i in range(11):
            row = 0
            for j, c in enumerate(g_coeffs):
                if c:
                    row |= 1 << (i + j)
            gen_matrix.append(row)
        
        # Generate all 2^11 codewords and extend by parity bit
        words = []
        for mask in range(1 << 11):
            word = 0
            for i in range(11):
                if mask & (1 << i):
                    word ^= gen_matrix[i]
            parity = bc(word) % 2
            extended_word = word | (parity << 17)
            words.append(extended_word)
        
        return words

    def is_valid_candidate(w, code):
        for c in code:
            if bc(w ^ c) < D:
                return False
        return True

    def greedy_from_seed(seed, order, dl):
        code = list(seed)
        code_set = set(code)
        for w in order:
            if time.time() > dl:
                break
            if w in code_set:
                continue
            if is_valid_candidate(w, code):
                code.append(w)
                code_set.add(w)
        return code

    def generate_by_weight(w):
        """Generate all N-bit words with exactly w bits set."""
        result = []
        for positions in combinations(range(N), w):
            word = 0
            for p in positions:
                word |= 1 << p
            result.append(word)
        return result

    best_code = []

    # --- Try QR(17) extended code first (should give ~2048 words with dist >= 6) ---
    qr_seed = construct_qr17_extended()
    qr_valid = False
    if len(qr_seed) >= 512:
        # Verify a sample of pairs to ensure distance >= 6
        qr_valid = True
        for i in range(0, min(100, len(qr_seed)), 10):
            for j in range(i+1, min(i+10, len(qr_seed))):
                if bc(qr_seed[i] ^ qr_seed[j]) < D:
                    qr_valid = False
                    break
            if not qr_valid:
                break
    
    if qr_valid:
        best_code = qr_seed
    else:
        # Fallback to Reed-Muller
        best_code = reed_muller_words()

    seed_set = set(best_code)

    # --- Weight-ordered greedy expansion ---
    weight_order = []
    for w in range(4, 15):
        if time.time() > deadline:
            break
        for candidate in generate_by_weight(w):
            if candidate not in seed_set:
                weight_order.append(candidate)

    code1 = greedy_from_seed(best_code, weight_order, deadline)
    if len(code1) > len(best_code):
        best_code = code1

    # --- Local search improvement ---
    if time.time() < deadline - 3.0 and len(best_code) > 0:
        code = list(best_code)
        code_set = set(code)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for _ in range(20000):
                if time.time() > deadline - 1.0:
                    break
                w = random.randint(0, (1 << N) - 1)
                if w in code_set:
                    continue
                if is_valid_candidate(w, code):
                    code.append(w)
                    code_set.add(w)
                    improved = True

        if len(code) > len(best_code):
            best_code = code

    return best_code
