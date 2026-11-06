import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    # ---- Build the 256-word linear code: shortened [18,9,6] extended QR code ----
    # Quadratic residues mod 19: {1,4,5,6,7,9,11,16,17}
    # Extended QR code of length 18: add a parity coordinate.
    # We construct the generator matrix for the [18,9,6] code, then shorten
    # by deleting one coordinate (the parity coordinate) to get [17,9,5]... 
    # Actually shortening reduces distance by at most 1, so [17,9,5].
    # But the task says the 256-word linear reference has distance 6.
    # Let me reconsider: the [18,9,6] code shortened gives [17,9,5] or [17,9,6]?
    # Shortening: delete a coordinate and keep only codewords with 0 in that position.
    # This preserves distance >= d-1. For the QR code, if we shorten at a 
    # "good" position we might keep distance 6.
    # 
    # Actually, let me just build the QR code generator matrix directly.
    # The quadratic residue code of length 18 (extended) over GF(2):
    # Generator: G = [I_9 | P] where P is 9x9 from the QR construction.
    # 
    # A simpler approach: the [18,9,6] code is the extended QR(19) code.
    # Its generator matrix can be built from the quadratic residues.
    # 
    # Let me use a different approach: construct the code via the 
    # quadratic residues mod 19.
    
    # The quadratic residues mod 19 are: 1, 4, 5, 6, 7, 9, 11, 16, 17
    # The non-residues are: 2, 3, 8, 10, 12, 13, 14, 15, 18
    # 
    # For the extended QR code of length 18:
    # The generator matrix G has rows that are the "shifts" of the 
    # residue pattern. But constructing this correctly is tricky.
    #
    # Alternative: I'll use the fact that the [18,9,6] QR code has a 
    # known generator matrix. Let me build it from the idempotent.
    #
    # Actually, let me try a more direct approach. The binary QR code 
    # of length n=18 (n-1=19 prime) has generator polynomial related 
    # to the quadratic residues.
    #
    # Let me just try to construct it and verify.
    
    # Quadratic residues mod 19
    qres = set()
    for x in range(1, 19):
        qres.add((x*x) % 19)
    # qres = {1, 4, 5, 6, 7, 9, 11, 16, 17}
    
    # The generator matrix for the [18,9,6] extended QR code:
    # Row i (0-indexed) of G: the i-th cyclic shift of the base row.
    # The base row for the QR code: position j (0 to 17) is 1 if 
    # (j+1) mod 19 is a quadratic residue, plus a parity bit.
    #
    # Actually, for the extended QR code, the generator matrix is:
    # G[i, j] = 1 if (j - i) mod 19 is in Q, for j = 0..17
    # And G[i, 17] is the parity bit to make each row even weight.
    #
    # Wait, the extended QR code of length 18: the first 17 coordinates 
    # are from the QR code of length 17 (which has the "hole" at position 18=0),
    # and the 18th is a parity check.
    #
    # Let me use the standard construction:
    # The [17,8,5] QR code (punctured) has generator from the idempotent.
    # The [18,9,6] extended code adds a parity coordinate.
    #
    # For the [18,9,6] code, generator matrix G (9 x 18):
    # G[i, j] = 1 if (j - i) mod 19 is a quadratic residue mod 19, for j=0..16
    # G[i, 17] = parity bit (to make row even)
    #
    # Then shortening: delete coordinate 17 (the parity coordinate) and 
    # keep only codewords with 0 in that position.
    # This gives a [17,8,5] or [17,8,6] code with 256 words.
    #
    # But we want 256 words with distance >= 6. If shortening gives distance 5,
    # that's not good. Let me check: the [18,9,6] code has minimum weight 6.
    # Shortening at any coordinate gives minimum weight >= 5.
    # 
    # Hmm, but the task says the 256-word linear reference has distance 6.
    # Maybe I'm confusing things. Let me try a different approach.
    #
    # Actually, maybe the construction is: take the [18,9,6] code and 
    # shorten at a specific coordinate to get [17,9,6] with 256 words.
    # Shortening preserves the dimension if the shortened coordinate 
    # is not always 0 (which it isn't for a non-trivial code).
    # And distance: shortening can reduce distance by at most 1.
    # If the original has distance 6, shortened has distance >= 5.
    #
    # But wait - if we shorten by keeping only codewords with 0 in position k,
    # the resulting code has dimension k-1 (if position k is not identically 0)
    # and minimum distance >= d-1.
    #
    # For [18,9,6] shortened: dimension 8, distance >= 5. That gives 256 words.
    # But distance might be 5, not 6.
    #
    # Let me just construct it and check.

    def build_linear_code():
        # Build [18,9,6] extended QR code generator matrix
        # G[i, j] for i in 0..8, j in 0..17
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            for j in range(17):
                # (j - i) mod 19
                pos = (j - i) % 19
                if pos in qres:
                    G[i, j] = 1
            # Parity bit
            G[i, 17] = G[i, :17].sum() % 2
        
        # Generate all 256 codewords
        codewords = set()
        for mask in range(256):
            cw = np.zeros(18, dtype=np.uint8)
            for i in range(9):
                if (mask >> i) & 1:
                    cw = (cw + G[i]) % 2
            # Convert to integer
            val = 0
            for j in range(18):
                val = (val << 1) | int(cw[j])
            codewords.add(val)
        
        return codewords, G
    
    all_cw_18, G = build_linear_code()
    
    # Shorten: delete coordinate 17 (parity), keep codewords with 0 there
    # Coordinate 17 is the least significant bit in our encoding (j=17 is LSB)
    # Actually, let me re-examine: in my encoding, j=0 is MSB, j=17 is LSB.
    # So coordinate 17 (parity) is the LSB.
    
    cw_17 = set()
    for val in all_cw_18:
        if (val & 1) == 0:  # parity bit (LSB) is 0
            cw_17.add(val >> 1)  # remove the parity bit
    
    # Check minimum distance
    cw_list = list(cw_17)
    min_d = 999
    for i in range(len(cw_list)):
        for j in range(i+1, len(cw_list)):
            d = (cw_list[i] ^ cw_list[j]).bit_count()
            if d < min_d:
                min_d = d
    
    # If min distance is 5, we need a different approach
    # Let me try shortening at a different coordinate
    if min_d < 6:
        # Try all 18 coordinates for shortening
        best_shortened = None
        best_min_d = 0
        for coord in range(18):
            cw_short = set()
            for val in all_cw_18:
                if ((val >> (17 - coord)) & 1) == 0:
                    # Remove coordinate 'coord'
                    upper = val >> (18 - coord)
                    lower = val & ((1 << (17 - coord)) - 1)
                    cw_short.add((upper << (17 - coord)) | lower)
            if len(cw_short) == 256:
                cl = list(cw_short)
                md = 999
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        d = (cl[i] ^ cl[j]).bit_count()
                        if d < md:
                            md = d
                if md > best_min_d:
                    best_min_d = md
                    best_shortened = cw_short
        
        if best_shortened is not None:
            cw_list = list(best_shortened)
            min_d = best_min_d
        else:
            cw_list = list(cw_17)
    
    # If still not distance 6, fall back to greedy
    if min_d < 6:
        # Use greedy to build a code
        cw_set = set()
        for w in range(1 << 17):
            if all((w ^ cw).bit_count() >= 6 for cw in cw_set):
                cw_set.add(w)
        cw_list = list(cw_set)
    
    # Now we have a starting code. Convert to numpy for SA.
    # Use a list of integers.
    code_list = list(cw_list)
    code_set = set(cw_list)
    
    # Precompute bit representations for SA
    # We'll work with integers and use bit_count for distances.
    
    # Simulated annealing:
    # 1. Try to add a random word
    # 2. If it conflicts, try to repair by flipping bits in conflicting codewords
    # 3. Accept with probability based on "energy" (number of conflicts)
    
    best_code = list(code_list)
    best_size = len(code_list)
    
    # Precompute all 17-bit words for random selection
    all_words = np.arange(1 << 17, dtype=np.int64)
    
    rng = random.Random(42)
    
    def get_conflicts(new_word, code):
        """Return list of (index, codeword) that conflict with new_word (dist < 6)."""
        conflicts = []
        for idx, cw in enumerate(code):
            if (new_word ^ cw).bit_count() < MIN_DIST:
                conflicts.append((idx, cw))
        return conflicts
    
    def try_repair(new_word, code, code_set):
        """
        Try to add new_word to code by flipping bits in conflicting codewords.
        Returns (success, new_code_list, new_code_set, energy_change)
        """
        conflicts = get_conflicts(new_word, code)
        if not conflicts:
            return True, code + [new_word], code_set | {new_word}, 0
        
        new_code = list(code)
        new_set = set(code_set)
        n_conflicts = len(conflicts)
        
        for idx, cw in conflicts:
            fixed = False
            # Try flipping each single bit in cw
            for bit in range(17):
                new_cw = cw ^ (1 << bit)
                if new_cw == new_word:
                    continue
                if (new_cw ^ new_word).bit_count() < MIN_DIST:
                    continue
                # Check distance to all other codewords (skip self)
                ok = True
                for j, other in enumerate(new_code):
                    if j == idx:
                        continue
                    if (new_cw ^ other).bit_count() < MIN_DIST:
                        ok = False
                        break
                if ok:
                    new_code[idx] = new_cw
                    new_set.discard(cw)
                    new_set.add(new_cw)
                    fixed = True
                    break
            
            if not fixed:
                # Try flipping two bits
                for bit1 in range(17):
                    if not fixed:
                        for bit2 in range(bit1 + 1, 17):
                            new_cw = cw ^ (1 << bit1) ^ (1 << bit2)
                            if new_cw == new_word:
                                continue
                            if (new_cw ^ new_word).bit_count() < MIN_DIST:
                                continue
                            ok = True
                            for j, other in enumerate(new_code):
                                if j == idx:
                                    continue
                                if (new_cw ^ other).bit_count() < MIN_DIST:
                                    ok = False
                                    break
                            if ok:
                                new_code[idx] = new_cw
                                new_set.discard(cw)
                                new_set.add(new_cw)
                                fixed = True
                                break
            
            if not fixed:
                # Remove the conflicting codeword
                new_code.pop(idx)
                new_set.discard(cw)
                # Re-check for remaining conflicts
                remaining = get_conflicts(new_word, new_code)
                if remaining:
                    return False, code, code_set, len(remaining)
        
        # Verify new_word can be added
        final_conflicts = get_conflicts(new_word, new_code)
        if final_conflicts:
            return False, code, code_set, len(final_conflicts)
        
        new_code.append(new_word)
        new_set.add(new_word)
        return True, new_code, new_set, 0
    
    # SA parameters
    T0 = 10.0  # initial temperature
    T_final = 0.01
    cooling = 0.999
    
    current_code = list(code_list)
    current_set = set(code_list)
    T = T0
    
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Pick a random word
        new_word = int(rng.choice(all_words))
        
        if new_word in current_set:
            no_improve_count += 1
            continue
        
        conflicts = get_conflicts(new_word, current_code)
        
        if not conflicts:
            # Can add directly!
            current_code.append(new_word)
            current_set.add(new_word)
            no_improve_count = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)
            continue
        
        # Try to repair
        success, new_code, new_set, energy = try_repair(new_word, current_code, current_set)
        
        if success:
            current_code = new_code
            current_set = new_set
            no_improve_count = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)
        else:
            # Accept with probability exp(-energy/T)
            if energy > 0 and T > T_final:
                log_accept = -energy / T
                if log_accept > -50:  # Avoid underflow
                    if rng.random() < np.exp(log_accept):
                        current_code = new_code
                        current_set = new_set
            no_improve_count += 1
        
        T = max(T * cooling, T_final)
        
        # Periodic restart if stuck
        if no_improve_count > 3000:
            # Revert to best and restart SA
            current_code = list(best_code)
            current_set = set(best_code)
            T = T0
            no_improve_count = 0
    
    # Return the best code found
    result = best_code
    
    # Verify
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST}"
    
    return result
