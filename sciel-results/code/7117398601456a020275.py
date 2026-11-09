import time
import itertools

def construct():
    N = 18
    MIN_DIST = 6
    
    # 1. Construct the Extended Quadratic Residue Code of length 18
    # This is based on the QR code for p=17.
    # The QR code has parameters [17, 9, 6].
    # Extending by a parity bit (which is 0 for all even-weight codewords) gives [18, 9, 6].
    # Total words: 2^9 = 512.
    
    p = 17
    # Compute quadratic residues modulo 17
    Q = set()
    for x in range(1, p):
        Q.add((x * x) % p)
    
    # Create the base vector u of length 17
    u = [0] * p
    for i in range(p):
        if i in Q:
            u[i] = 1
            
    # Generate cyclic shifts of u
    shifts = []
    for k in range(p):
        shift = [0] * p
        for i in range(p):
            shift[i] = u[(i + k) % p]
        shifts.append(shift)
        
    # Find a basis of dimension 9 from these shifts using Gaussian Elimination
    basis = []
    for s in shifts:
        temp = s[:]
        # Reduce temp using current basis
        for b in basis:
            # Find first 1 in temp
            first_one = -1
            for i in range(p):
                if temp[i] == 1:
                    first_one = i
                    break
            if first_one == -1:
                break # temp is zero, dependent
            
            # Find basis vector with pivot at first_one
            # We maintain basis in row echelon form (pivot positions are increasing)
            # Actually, simpler: just check if we can reduce to zero
            # Let's do standard reduction:
            # Find b in basis that has a 1 in the first_one position
            pivot_found = False
            for j in range(len(basis)):
                if basis[j][first_one] == 1:
                    for i in range(p):
                        temp[i] ^= basis[j][i]
                    pivot_found = True
                    break
            
            if not pivot_found:
                # This means no basis vector has a 1 in the position where temp has its first 1.
                # This implies temp is independent of the current basis.
                # We add temp to the basis.
                # Note: We don't need to fully reduce temp against *all* basis vectors 
                # to prove independence, just that its leading term isn't covered.
                # However, for the basis to be useful for generating all combinations, 
                # it just needs to be a linearly independent set.
                basis.append(temp)
                break
        
    # The dimension should be 9. If not, something is wrong, but for p=17 it is 9.
    # Generate all 2^9 linear combinations
    words_17 = []
    num_basis = len(basis)
    # If basis size is not 9, we might have a bug, but let's proceed with what we have.
    # Ideally num_basis == 9.
    
    for mask in range(1 << num_basis):
        word = [0] * p
        for i in range(num_basis):
            if (mask >> i) & 1:
                for j in range(p):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    # Extend to length 18
    # Since all weights in the QR code are even, the parity bit is 0.
    # We can just append 0.
    words_18 = []
    for w in words_17:
        # parity = sum(w) % 2
        # extended = w + [parity]
        # Since weight is even, parity is 0.
        extended = w + [0]
        
        # Convert to integer
        val = 0
        for i, bit in enumerate(extended):
            if bit:
                val |= (1 << i)
        words_18.append(val)
        
    # Verify minimum distance
    # 512 words is small enough for full check
    min_dist_found = N
    L = len(words_18)
    for i in range(L):
        for j in range(i + 1, L):
            d = (words_18[i] ^ words_18[j]).bit_count()
            if d < min_dist_found:
                min_dist_found = d
            if min_dist_found < MIN_DIST:
                break
        if min_dist_found < MIN_DIST:
            break
            
    if min_dist_found >= MIN_DIST:
        return words_18
    else:
        # Fallback: If the QR construction failed (should not happen), 
        # try a simple greedy approach to get *some* valid code, 
        # though 512 is the target.
        # For robustness, return an empty list or a small known valid code if verification fails.
        # But QR code verification should pass.
        return []

if __name__ == "__main__":
    # Quick self-test
    words = construct()
    print(f"Constructed {len(words)} words.")
    if len(words) > 0:
        # Verify again
        min_d = 18
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        print(f"Min distance: {min_d}")
