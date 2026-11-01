<<<<<<< SEARCH
def _try_alternative_constructions():
    """
    Try alternative structural constructions that might yield rank < 28.
    Returns (U, V, W) if a valid lower-rank solution is found, else None.
    """
    if time.time() > DEADLINE:
        return None
    
    # Try a construction based on a different block decomposition.
    # Instead of 4 blocks of 2x2, try 2 blocks of 2x4.
    # For 2x2x4, we can try to find a rank-13 or rank-14 decomposition.
    # If we find rank-14 for each, total is 28 (same).
    # If we find rank-13 for each, total is 26 (improvement!).
    
    # Known: minimal rank for 2x2x4 is at most 14.
    # Let's try to construct a rank-14 decomposition for 2x2x4.
    
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4.
    # Tensor entries: T[a,b,c] = 1 if a=i*2+j, b=j*4+k, c=k*2+i
    # for i,j in 0..1, k in 0..3.
    
    # Try a Strassen-like construction for 2x2x4.
    # Split B into two 2x2 blocks: B = [B1, B2]
    # Then C = A*B1 + A*B2 (but with the specific index structure).
    
    # Actually, the structure is different. Let me think...
    # C[k*2+i] = sum_j A[i,j] * B[j, k]
    # So C is 2x4, with C[i, k] = sum_j A[i,j] * B[j, k]
    
    # This is just standard 2x2 times 2x4 matrix multiplication.
    # The minimal rank for this is known to be at most 14 (from literature).
    
    # For now, return None. Constructing an explicit rank-14 solution
    # for 2x2x4 and combining them is complex and may not yield improvement
    # over the existing rank-28 solution.
    
    return None
=======
def _try_alternative_constructions():
    """
    Try alternative structural constructions that might yield rank < 28.
    Returns (U, V, W) if a valid lower-rank solution is found, else None.
    """
    if time.time() > DEADLINE:
        return None
    
    # The 2x2x8 problem decomposes into two independent 2x2x4 sub-problems:
    # Block 0: B columns 0,1 ; C columns 0,1
    # Block 1: B columns 2,3 ; C columns 2,3
    # Block 2: B columns 4,5 ; C columns 4,5
    # Block 3: B columns 6,7 ; C columns 6,7
    #
    # Actually it's 4 independent 2x2x2 sub-problems (each rank 7 via Strassen).
    # To get rank < 28, we need to find a decomposition of 2x2x4 with rank <= 13.
    #
    # For 2x2x4: A is 2x2 (4 entries), B is 2x4 (8 entries), C is 2x4 (8 entries).
    # Tensor: T[a,b,c] = 1 if a = i*2+j, b = j*4+k, c = k*2+i, for i,j in 0..1, k in 0..3.
    #
    # We'll do a bounded random search for a rank-13 decomposition of 2x2x4.
    # Strategy: use integer/half-integer entries in {-1, -1/2, 0, 1/2, 1}.
    
    if time.time() > DEADLINE - 15:
        return None
    
    # Define the 2x2x4 tensor
    n2, m2, p2 = 2, 2, 4
    nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2  # 4, 8, 8
    valid_224 = set()
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                valid_224.add((i*m2+j, j*p2+k, k*n2+i))
    
    # Candidate values for entries
    candidates = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    
    # Try random search for rank-13 decomposition of 2x2x4
    # We'll parameterize U (4x13), V (8x13), W (8x13) with entries from candidates.
    # That's 4*13 + 8*13 + 8*13 = 52 + 104 + 104 = 260 entries, each 5 choices.
    # Too many for exhaustive. Use random sampling.
    
    rng = random.Random(42)
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - 10:
            break
        
        # Randomly assign values
        U224 = [[rng.choice(candidates) for _ in range(13)] for _ in range(nm2)]
        V224 = [[rng.choice(candidates) for _ in range(13)] for _ in range(mp2)]
        W224 = [[rng.choice(candidates) for _ in range(13)] for _ in range(pn2)]
        
        # Verify
        valid = True
        for a in range(nm2):
            for b in range(mp2):
                for c in range(pn2):
                    s = sum(U224[a][r] * V224[b][r] * W224[c][r] for r in range(13))
                    expected = Fraction(1) if (a, b, c) in valid_224 else Fraction(0)
                    if s != expected:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a rank-13 decomposition of 2x2x4!
            # Now construct the full 2x2x8 by using this for each of the 4 blocks.
            # But wait - if we use the same decomposition for all 4 blocks, total rank = 4*13 = 52.
            # That's worse than 28. We need to find a way to share factors.
            #
            # Actually, the 4 blocks share the same A matrix. So we can potentially
            # use the same U for all blocks. Let me reconsider.
            #
            # In the full 2x2x8:
            # U is 4xR (A indices)
            # V is 16xR (B indices: j*8+k for j in 0..1, k in 0..7)
            # W is 16xR (C indices: k*2+i for k in 0..7, i in 0..1)
            #
            # The tensor is: T[a, b, c] = 1 if a=i*2+j, b=j*8+k, c=k*2+i
            # for i,j in 0..1, k in 0..7.
            #
            # This factors as: for each k, the sub-tensor over (i,j) is the standard 2x2 multiplication.
            # So the full tensor is a "direct sum" of 8 copies of 2x2 multiplication, but they
            # share the same A indices.
            #
            # Actually no - each k gives a separate "slice". The tensor is:
            # T[i*2+j, j*8+k, k*2+i] = 1
            #
            # This means for fixed k, we have T[i*2+j, j*8+k, k*2+i] = 1 for all i,j.
            # So for each k, the (i,j) substructure is 2x2 matrix multiplication.
            #
            # If we use Strassen for each k, we get 8*7 = 56 rank. But the blocks share U.
            # In the Strassen construction for 2x2x2, U is 4x7. If we use the same U for all 8 k's,
            # we need V and W to be 16x56... no that doesn't work either.
            #
            # Let me think differently. The standard way to decompose 2x2x8 is:
            # C = A * B where A is 2x2, B is 2x8.
            # Split B into 4 blocks of 2x2: B = [B0, B1, B2, B3]
            # Then C = A*B0 + A*B1 + A*B2 + A*B3 (each A*Bk is 2x2)
            # C = [C0, C1, C2, C3] where Ck = A*Bk
            #
            # Each Ck = A*Bk is a 2x2x2 multiplication, rank 7 via Strassen.
            # Total rank = 4*7 = 28. This is what the current solution does.
            #
            # To get rank < 28, we need a smarter construction. One approach:
            # Use a rank-13 decomposition of 2x2x4 (multiplying 2x2 by 2x4).
            # Then 2x2x8 = 2x2x4 + 2x2x4 (two independent blocks), total rank = 2*13 = 26.
            #
            # But the two 2x2x4 blocks share the same A. So we need:
            # U shared (4x13), V1 (8x13) for first block, V2 (8x13) for second block,
            # W1 (8x13) for first block, W2 (8x13) for second block.
            # Total rank = 26.
            #
            # But this requires the same U to work for both 2x2x4 decompositions.
            # That's a strong constraint. Let's just try random search for the full 2x2x8
            # with rank 26 directly.
            break
    
    # Direct random search for rank-26 decomposition of 2x2x8
    # U: 4x26, V: 16x26, W: 16x26
    # Entries from {-1, -1/2, 0, 1/2, 1}
    # That's 4*26 + 16*26 + 16*26 = 104 + 416 + 416 = 936 entries.
    # Too many for random search with 5 choices each.
    #
    # Alternative: use the block structure. Assume the decomposition has the form:
    # For block 0 (k=0,1): U0 (4x13), V0 (8x13), W0 (8x13)
    # For block 1 (k=2,3): U1 (4x13), V1 (8x13), W1 (8x13)
    # For block 2 (k=4,5): U2 (4x13), V2 (8x13), W2 (8x13)
    # For block 3 (k=6,7): U3 (4x13), V3 (8x13), W3 (8x13)
    # Total rank = 52. Too high.
    #
    # Better: assume all 4 blocks share the same U (4x13).
    # Then we need V (16x13) and W (16x13) such that the full tensor is decomposed.
    # Total rank = 13.
    #
    # This is very constrained. Let's try a structured search.
    #
    # Actually, let's try a different approach: use the known rank-14 decomposition
    # of 2x2x4 (if it exists) and see if we can combine two such decompositions
    # with shared U to get rank 26.
    #
    # For now, return None. The random search is unlikely to succeed quickly.
    
    return None
>>>>>>> REPLACE

