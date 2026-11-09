def construct():
    n = 10
    m = 5
    
    # Generate two MOLS of order 5
    # Standard construction for prime power m:
    # L_k(i, j) = (i + k*j) % m for k=1..m-1
    # Here m=5, so we can get 4 MOLS of order 5.
    # We only need 2 to lift to order 10?
    # Actually, we can lift any number of MOLS of order m to order 2m?
    # The construction in the current program lifts MOLS of order 5 to order 10.
    # Let's verify the lifting construction.
    # i = 2*i1 + i2, j = 2*j1 + j2 where i1,j1 in {0,1}, i2,j2 in {0..4}?
    # No, i in 0..9. i = 5*i1 + i2?
    # Current code: i1 = i // 5, i2 = i % 5.
    # So i1 in {0,1}, i2 in {0..4}.
    # val = 2 * sq[i2][j2] + ((i1 + j1) % 2)
    # This maps values 0..4 to 0..9 by doubling and adding parity.
    # This is a valid "doubling" construction for Latin squares.
    # If sq is a Latin square of order 5, is the new square a Latin square of order 10?
    # Yes, this is a standard construction.
    # Are they orthogonal?
    # If sq_a and sq_b are orthogonal, are the lifted squares orthogonal?
    # The pairs (val_a, val_b) are (2*a + p, 2*b + p) where p = (i1+j1)%2.
    # Since a,b are distinct pairs in the base square, and p is determined by i,j,
    # the mapping is injective on the pairs if the base pairs are distinct.
    # So yes, orthogonality is preserved.
    
    # We can generate 4 MOLS of order 5, so we can generate 4 MOLS of order 10?
    # Wait, if we can generate 4, why does the current program only return 2?
    # Let's check the current program's logic.
    # It generates m-1 = 4 squares of order 5.
    # It lifts all 4 to order 10.
    # Then it validates them.
    # Then it tries to find a 5th.
    # Why does it only return 2?
    # Maybe the lifted squares are NOT all mutually orthogonal?
    # Let's check.
    # Base squares: L_k(i,j) = (i + k*j) % 5.
    # Lifted: V_k(i,j) = 2 * ((i2 + k*j2) % 5) + ((i1 + j1) % 2).
    # Consider V_k and V_l.
    # Pair: (2*(i2 + k*j2)%5 + p, 2*(i2 + l*j2)%5 + p).
    # If (i,j) != (i',j'), do we get distinct pairs?
    # This depends on whether the base pairs are distinct.
    # Since L_k and L_l are orthogonal for k != l, the pairs ((i2+k*j2)%5, (i2+l*j2)%5) are distinct for all (i2,j2).
    # However, the lifted pair also includes p.
    # If we fix i2,j2, p can be 0 or 1 depending on i1,j1.
    # So for a fixed (i2,j2), we have 4 positions (i1,j1) in {0,1}x{0,1}.
    # The base symbol pair is fixed for (i2,j2).
    # The lifted symbols are (2*a+p, 2*b+p).
    # For p=0: (2a, 2b).
    # For p=1: (2a+1, 2b+1).
    # These are distinct pairs.
    # So for each (i2,j2), we get 4 distinct pairs in the 2x2 block.
    # Since the base pairs are distinct for different (i2,j2), the sets of pairs from different blocks are disjoint?
    # The values in the first component are 2*a or 2*a+1.
    # The values in the second component are 2*b or 2*b+1.
    # So the pairs are in {0..9}x{0..9}.
    # Are they all distinct across the whole 10x10 grid?
    # Yes, because the base pairs (a,b) are distinct for each (i2,j2), and the parity p just splits them into even/odd.
    # So we should get 4 mutually orthogonal Latin squares of order 10!
    
    # Why does the current program return only 2?
    # Let's look at the validation in the current program.
    # It checks is_latin and is_orthogonal.
    # Maybe my analysis is wrong?
    # Let's test with a small example.
    # Order 2 MOLS? N(2)=1.
    # Order 4 MOLS? N(4)=3.
    # Order 10 MOLS?
    # If the construction works, we should get 4.
    # Let's re-read the current program output.
    # "inspiration 3 ... The current program returns 2 orthogonal Latin squares".
    # This is an observation from the prompt.
    # If the construction yields 4, why does it return 2?
    # Maybe the validation fails for the 3rd and 4th?
    # Or maybe the "lift" is not valid for all 4?
    # Let's check the Latin property.
    # Row i: i1 fixed. i2 varies 0..4.
    # Val = 2 * ((i2 + k*j2)%5) + ((i1+j1)%2).
    # As j varies 0..9, j1 varies 0,1,0,1.. and j2 varies 0..4,0..4.
    # For fixed i1, j1=0: j2=0..4. Val = 2*((i2+k*j2)%5) + i1.
    # j2=0..4 gives distinct residues mod 5. So 2*res + i1 gives 5 distinct values of parity i1.
    # For j1=1: j2=0..4. Val = 2*((i2+k*j2)%5) + (i1+1)%2.
    # This gives 5 distinct values of parity 1-i1.
    # So the row contains 5 values of parity i1 and 5 of parity 1-i1.
    # Are they all distinct?
    # The sets of values are {2r+i1} and {2r+1-i1}.
    # These sets are disjoint.
    # So the row has 10 distinct values.
    # So it IS a Latin square.
    
    # So why only 2?
    # Maybe the orthogonality check fails?
    # Let's check orthogonality of V_1 and V_2.
    # Pairs (2a+p, 2b+p).
    # If (i,j) != (i',j'), do we have distinct pairs?
    # If (i2,j2) != (i2',j2'), then (a,b) != (a',b') because L1, L2 are orthogonal.
    # So the base pairs are distinct.
    # The lifted pairs are determined by (a,b,p).
    # Since (a,b) are distinct, the lifted pairs are distinct.
    # So V_1 and V_2 are orthogonal.
    # Same for V_1, V_3.
    # So we should have 4 MOLS.
    
    # If the current program returns 2, maybe there is a bug in the validation or the generation?
    # Or maybe the "inspiration" text is just wrong/outdated?
    # Or maybe the seed program returns 2, and the current program (which is the seed) returns 2.
    # If I can return 4, the score would be 4/2 = 2.0!
    # This would be a huge improvement.
    
    # Let's implement the generation of 4 squares and validate them.
    # If they are valid, I return 4.
    # If not, I return whatever is valid.
    
    m = 5
    mols_m = []
    for k in range(1, m): # k=1,2,3,4
        sq = [[(i + k * j) % m for j in range(m)] for i in range(m)]
        mols_m.append(sq)
        
    mols_n = []
    for sq in mols_m:
        new_sq = []
        for i in range(10):
            row = []
            for j in range(10):
                i1 = i // m
                i2 = i % m
                j1 = j // m
                j2 = j % m
                val = 2 * sq[i2][j2] + ((i1 + j1) % 2)
                row.append(val)
            new_sq.append(row)
        mols_n.append(new_sq)
        
    # Validate
    def is_latin(sq):
        for i in range(10):
            if set(sq[i]) != set(range(10)):
                return False
            col = [sq[k][i] for k in range(10)]
            if set(col) != set(range(10)):
                return False
        return True

    def is_orthogonal(sq_a, sq_b):
        seen_pairs = set()
        for i in range(10):
            for j in range(10):
                pair = (sq_a[i][j], sq_b[i][j])
                if pair in seen_pairs:
                    return False
                seen_pairs.add(pair)
        return True
        
    valid_sqs = []
    for sq in mols_n:
        if not is_latin(sq):
            continue
        orthogonal = True
        for v_sq in valid_sqs:
            if not is_orthogonal(sq, v_sq):
                orthogonal = False
                break
        if orthogonal:
            valid_sqs.append(sq)
            
    # If we get 4, great.
    # If we get less, we return what we have.
    # I suspect we will get 4.
    
    result = [
        [[int(val) for val in row] for row in sq]
        for sq in valid_sqs
    ]
    
    return result
