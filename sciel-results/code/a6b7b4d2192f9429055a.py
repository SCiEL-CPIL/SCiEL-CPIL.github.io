import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # 1. Compute Quadratic Residues modulo 17
    p = 17
    Q = set()
    for x in range(1, p):
        Q.add((x * x) % p)
    
    # 2. Create the base vector u of length 17
    # u[i] = 1 if (i % p) is in Q, else 0
    # Note: 0 is not in Q.
    u = [0] * p
    for i in range(p):
        if i in Q:
            u[i] = 1
            
    # 3. Generate cyclic shifts and find a basis
    # We need a basis of dimension 9.
    # Generate all 17 shifts
    shifts = []
    for k in range(p):
        shift = [0] * p
        for i in range(p):
            shift[i] = u[(i + k) % p]
        shifts.append(shift)
        
    # Gaussian Elimination to find independent vectors
    basis = []
    for s in shifts:
        # Check if s is in the span of current basis
        # Represent vectors as tuples or lists
        dependent = False
        # Try to reduce s using basis
        temp = s[:]
        for b in basis:
            # Find first 1 in temp
            first_one = -1
            for i in range(p):
                if temp[i] == 1:
                    first_one = i
                    break
            if first_one == -1:
                break # temp is zero
            
            # Find basis vector with first 1 at same position or earlier?
            # Standard row echelon form: pivot positions
            # Let's just do simple reduction:
            # Find b in basis that has a 1 in the first_one position of temp
            found_pivot = False
            for j in range(len(basis)):
                if basis[j][first_one] == 1:
                    # XOR temp with basis[j]
                    for i in range(p):
                        temp[i] ^= basis[j][i]
                    found_pivot = True
                    break
            if not found_pivot:
                # No basis vector covers this pivot, so s is independent
                # Add s to basis
                basis.append(s)
                dependent = False
                break
            else:
                dependent = True # Actually, we need to finish reduction to see if it becomes zero
        # Re-evaluate dependency properly
        # Better approach: maintain basis in row echelon form
        pass

    # Let's use a simpler method: 
    # The QR code is cyclic. The generator polynomial g(x) has degree p - dim = 17 - 9 = 8.
    # We can find the generator polynomial by finding the minimal polynomial of the shifts?
    # Or just use the fact that the dimension is 9.
    # Let's just collect all 512 words by assuming the first 9 shifts are independent?
    # They are not necessarily independent in that order.
    
    # Robust Basis Finding:
    basis = []
    for s in shifts:
        # Check if s is linearly independent from basis
        is_independent = True
        # Reduce s by basis
        temp = s[:]
        for b in basis:
            # Find first 1 in temp
            first_one = -1
            for i in range(p):
                if temp[i] == 1:
                    first_one = i
                    break
            if first_one == -1:
                break
            
            # Find basis vector with pivot at first_one
            pivot_found = False
            for j in range(len(basis)):
                # Find pivot of basis[j]
                pivot_b = -1
                for i in range(p):
                    if basis[j][i] == 1:
                        pivot_b = i
                        break
                if pivot_b == first_one:
                    for i in range(p):
                        temp[i] ^= basis[j][i]
                    pivot_found = True
                    break
            
            if not pivot_found:
                # This s has a pivot not covered by basis, so it's independent
                # But we need to add it to basis and break
                # Wait, the logic above is slightly flawed for standard GE.
                # Let's use a standard GE function.
                pass

    def add_to_basis(basis, vec):
        # Add vec to basis if independent
        # Returns True if added
        temp = vec[:]
        for b in basis:
            # Find first 1 in temp
            first_one = -1
            for i in range(p):
                if temp[i] == 1:
                    first_one = i
                    break
            if first_one == -1:
                break # temp is zero, dependent
            
            # Find b with pivot at first_one
            pivot_found = False
            for j in range(len(basis)):
                pivot_b = -1
                for i in range(p):
                    if basis[j][i] == 1:
                        pivot_b = i
                        break
                if pivot_b == first_one:
                    for i in range(p):
                        temp[i] ^= basis[j][i]
                    pivot_found = True
                    break
            if not pivot_found:
                # temp has a pivot not in basis, so it's independent
                # But we need to reduce it fully?
                # Actually, if no basis vector has a 1 in the first_one position of temp,
                # then temp is independent of the current basis.
                # We add temp to basis.
                basis.append(temp)
                return True
        
        return False

    basis = []
    for s in shifts:
        add_to_basis(basis, s)
        
    # Now basis should have size 9
    # Generate all 2^9 linear combinations
    words_17 = []
    num_basis = len(basis)
    for mask in range(1 << num_basis):
        word = [0] * p
        for i in range(num_basis):
            if (mask >> i) & 1:
                for j in range(p):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    # Extend to length 18 with parity bit
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        # Append parity bit to make total weight even
        # If sum is odd, append 1. If even, append 0.
        # New weight = sum(w) + parity.
        # If sum(w) is odd, parity=1, new weight even.
        # If sum(w) is even, parity=0, new weight even.
        extended = w + [parity]
        
        # Convert to integer
        val = 0
        for i, bit in enumerate(extended):
            if bit:
                val |= (1 << i)
        words_18.append(val)
        
    # Verify minimum distance (sample check or full check if fast enough)
    # 512 words is small enough to check all pairs?
    # 512 * 511 / 2 ~ 130k pairs.
    # Each check is XOR and bit_count. Very fast in Python.
    
    min_dist_found = N
    for i in range(len(words_18)):
        for j in range(i + 1, len(words_18)):
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
        # Fallback to greedy if construction failed (should not happen)
        return []
