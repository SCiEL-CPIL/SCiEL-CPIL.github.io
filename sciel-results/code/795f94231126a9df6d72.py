"""
Construct Mutually Orthogonal Latin Squares (MOLS) of order 10.

Approach:
1. Use the known fact that for prime power q, there exist q-1 MOLS of order q.
2. For order 10, we cannot directly use prime power construction.
3. However, we can use the Bruck-Ryser-Chowla theorem and known constructions.
4. The maximum number of MOLS of order 10 is known to be 2 (from Euler's 36 officers problem, 
   it was proven that you cannot have 3 MOLS of order 6, but for order 10, we can have up to 2).
   
   Actually, let me reconsider. For order n, the maximum number of MOLS is at most n-1.
   For n=10, we need to find how many MOLS we can construct.
   
   A better approach: Use the construction based on finite fields where possible, 
   and for order 10, use a direct construction or search.
   
   Known result: The maximum number of MOLS of order 10 is 2. This was established 
   through computational search and theoretical bounds.
   
   Let me construct 2 MOLS of order 10 explicitly.
   
   Actually, I recall that for order 10, we can construct exactly 2 MOLS. Let me use 
   a systematic construction.

   One approach: Use the fact that 10 = 2 * 5. We can use a tensor/Kronecker product 
   approach or a direct construction.

   Let me use a direct construction based on modular arithmetic with careful design.

   For a Latin square of order 10, we can use:
   L1[i][j] = (i + j) % 10
   L2[i][j] = (i + 2*j) % 10

   But we need to check orthogonality. The pair (L1[i][j], L2[i][j]) must be distinct 
   for all (i,j).

   L1[i][j] = (i+j) % 10
   L2[i][j] = (i+2j) % 10

   The pair is ((i+j) mod 10, (i+2j) mod 10). For two cells (i1,j1) and (i2,j2) to 
   have the same pair:
   i1+j1 ≡ i2+j2 (mod 10)
   i1+2j1 ≡ i2+2j2 (mod 10)

   Subtracting: j1 - j2 ≡ 0 (mod 10), so j1 ≡ j2 (mod 10).
   Then i1 ≡ i2 (mod 10). So the pairs are all distinct. 

   So L1 and L2 are orthogonal. Can we find a third?

   L3[i][j] = (i + 3*j) % 10
   Check (L1, L3): (i+j, i+3j). If equal: j1≡j2, i1≡i2. Distinct.
   Check (L2, L3): (i+2j, i+3j). If equal: j1≡j2, i1≡i2. Distinct.

   So L1, L2, L3 are pairwise orthogonal! That gives us 3 MOLS.

   Let's try L4[i][j] = (i + 4*j) % 10
   Check with L1: (i+j, i+4j) -> distinct
   Check with L2: (i+2j, i+4j) -> distinct  
   Check with L3: (i+3j, i+4j) -> distinct

   So L1, L2, L3, L4 are pairwise orthogonal. That's 4 MOLS!

   Let's try L5[i][j] = (i + 5*j) % 10
   Check with L1: (i+j, i+5j). If i1+j1≡i2+j2 and i1+5j1≡i2+5j2:
   4(j1-j2)≡0 mod 10, so j1-j2≡0 mod 10/gcd(4,10)=gcd(4,10)=2, so j1-j2≡0 mod 2.
   This doesn't force j1≡j2 mod 10. So NOT orthogonal!

   So L5 doesn't work with L1. Let me try other multipliers.

   The condition for (i+a*j) and (i+b*j) to be orthogonal:
   (a-b)*j ≡ 0 mod 10 must imply j≡0 mod 10, i.e., gcd(a-b, 10) = 1.

   So we need multipliers a1, a2, ..., ak such that all pairwise differences are coprime to 10.
   gcd(a_i - a_j, 10) = 1 for all i≠j.

   Possible multipliers: 0, 1, 3, 7, 9 (these are the units mod 10... wait, no)
   Actually, we need gcd(a_i - a_j, 10) = 1, meaning a_i - a_j is odd and not divisible by 5.

   Let's list numbers 0-9 and check which pairs have difference coprime to 10:
   Coprime to 10 means the difference is in {1,3,7,9} mod 10 (i.e., odd and not multiple of 5).

   Multipliers: 0, 1, 3, 7, 9
   Differences:
   1-0=1 ✓, 3-0=3 ✓, 7-0=7 ✓, 9-0=9 ✓
   3-1=2 ✗ (gcd(2,10)=2)

   So 0 and 1 can't both be used with 3. Let me reconsider.

   We need a set S ⊆ {0,...,9} such that for all a,b ∈ S, a≠b, gcd(a-b, 10)=1.

   Let's try: {0, 1, 3, 7, 9}
   1-0=1✓, 3-0=3✓, 7-0=7✓, 9-0=9✓
   3-1=2✗

   Try {0, 1, 7, 9}:
   1-0=1✓, 7-0=7✓, 9-0=9✓
   7-1=6✗

   Try {0, 3, 7, 9}:
   3-0=3✓, 7-0=7✓, 9-0=9✓
   7-3=4✗

   Try {0, 1, 9}:
   1-0=1✓, 9-0=9✓, 9-1=8✗

   Try {0, 3, 9}:
   3-0=3✓, 9-0=9✓, 9-3=6✗

   Try {0, 7, 9}:
   7-0=7✓, 9-0=9✓, 9-7=2✗

   Try {1, 3, 7}:
   3-1=2✗

   Try {1, 7, 9}:
   7-1=6✗

   Try {3, 7, 9}:
   7-3=4✗

   Hmm, it seems hard to get more than 2 or 3. Let me try {0, 1}:
   1-0=1✓. So {0,1} works. That gives L[i][j]=(i+0*j)%10=i and L[i][j]=(i+1*j)%10=(i+j)%10.

   Wait, L[i][j]=i is not a Latin square (each row is constant). We need proper Latin squares.

   Let me reconsider. A Latin square of form L[i][j]=(i+a*j)%n is a Latin square iff gcd(a,n)=1.
   For n=10, a must be coprime to 10, so a ∈ {1,3,7,9}.

   So valid multipliers are {1,3,7,9}.
   Check pairwise:
   3-1=2, gcd(2,10)=2 ✗
   7-1=6, gcd(6,10)=2 ✗
   9-1=8, gcd(8,10)=2 ✗
   7-3=4, gcd(4,10)=2 ✗
   9-3=6, gcd(6,10)=2 ✗
   9-7=2, gcd(2,10)=2 ✗

   None of the valid multipliers are pairwise coprime differences! So this linear construction 
   only gives us 1 Latin square at a time from this family, and no two are orthogonal.

   I need a different approach. Let me use a direct construction or a known result.

   Known: The maximum number of MOLS of order 10 is 2. Let me construct 2 explicitly.
   """

def construct():
    n = 10
    
    # Construct two orthogonal Latin squares of order 10
    # L1: standard cyclic
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # L2: need to be orthogonal to L1
    # Try L2[i][j] = (i + 2*j) % n -- but gcd(2,10)=2, so not a Latin square!
    # Each row would have repeated values.
    
    # Let me verify: L2[i][j] = (i + 2*j) % 10
    # Row 0: 0,2,4,6,8,0,2,4,6,8 -- not a permutation! Not a Latin square.
    
    # So I need a different construction for L2.
    
    # Let me use a known pair of orthogonal Latin squares of order 10.
    # One construction uses the fact that 10 = 2*5 and builds from smaller squares.
    
    # Alternative: Use a direct table.
    # Let me construct L2 manually to be orthogonal to L1.
    
    # L1[i][j] = (i+j) % 10
    # For orthogonality, the map (i,j) -> (L1[i][j], L2[i][j]) must be bijective.
    
    # Let me try L2[i][j] = (2*i + j) % 10
    # Check if L2 is a Latin square:
    # Row i: (2i+j) % 10 for j=0..9: 2i, 2i+1, ..., 2i+9 mod 10 -- this is a permutation! ✓
    # Col j: (2i+j) % 10 for i=0..9: j, j+2, j+4, ..., j+18 mod 10
    # This gives j, j+2, j+4, j+6, j+8, j, j+2, j+4, j+6, j+8 -- NOT a permutation! ✗
    
    # So that doesn't work either.
    
    # Let me think differently. I'll use a known construction.
    # For order 10, one can use the following pair:
    
    # Actually, let me just hardcode a known valid pair of orthogonal Latin squares of order 10.
    
    # L1 (cyclic):
    # 0 1 2 3 4 5 6 7 8 9
    # 1 2 3 4 5 6 7 8 9 0
    # ...
    
    # L2: I need to construct this carefully.
    # Let me use a different approach: define L2 by a permutation of columns in each row.
    
    # For L2 to be orthogonal to L1 where L1[i][j]=(i+j)%10:
    # The pair (L1[i][j], L2[i][j]) must cover all 100 pairs exactly once.
    # L1[i][j] = (i+j)%10 = k means j = (k-i)%10.
    # So for each (k, l) pair, there should be exactly one (i,j) with L1[i][j]=k and L2[i][j]=l.
    # j = (k-i)%10, and L2[i][(k-i)%10] = l.
    # So for each i and l, L2[i][(k-i)%10] = l for exactly one k.
    # This means that for each row i, the values L2[i][j] as j varies must be a permutation 
    # (which is already required for Latin square), AND the specific mapping must work out.
    
    # Let me try: L2[i][j] = (i + 3*j) % 10
    # Row check: i, i+3, i+6, i+9, i+2, i+5, i+8, i+1, i+4, i+7 mod 10 -- permutation ✓
    # Col check: For col j, values are i+3j mod 10 for i=0..9 -- permutation ✓
    # So L2 is a Latin square.
    
    # Orthogonality with L1: (i+j, i+3j) mod 10
    # If (i1+j1, i1+3j1) ≡ (i2+j2, i2+3j2) mod 10:
    # j1-j2 ≡ 0 (from subtracting), so j1≡j2, then i1≡i2. ✓
    
    # Great! So L1[i][j]=(i+j)%10 and L2[i][j]=(i+3j)%10 are orthogonal.
    
    L2 = [[(i + 3*j) % n for j in range(n)] for i in range(n)]
    
    # Can we find a third? L3[i][j] = (i + a*j) % 10 where gcd(a,10)=1.
    # a ∈ {1,3,7,9}. We used a=1 and a=3.
    # Try a=7: L3[i][j] = (i+7j)%10
    # Orthogonal to L1? (i+j, i+7j): j1≡j2 from subtracting, i1≡i2. ✓
    # Orthogonal to L2? (i+3j, i+7j): 4(j1-j2)≡0 mod 10. gcd(4,10)=2, so j1-j2≡0 mod 5.
    # Not necessarily j1≡j2 mod 10. So NOT orthogonal to L2. ✗
    
    # Try a=9: L3[i][j] = (i+9j)%10
    # Orthogonal to L1? (i+j, i+9j): 8(j1-j2)≡0 mod 10. gcd(8,10)=2, j1-j2≡0 mod 5. ✗
    
    # So the linear family only gives 2 mutually orthogonal squares.
    
    # Let me verify our 2 squares are correct and return them.
    
    # Verify L1 is Latin:
    for i in range(n):
        row = L1[i]
        col = [L1[r][i] for r in range(n)]
        assert sorted(row) == list(range(n)), f"L1 row {i} not Latin"
        assert sorted(col) == list(range(n)), f"L1 col {i} not Latin"
    
    # Verify L2 is Latin:
    for i in range(n):
        row = L2[i]
        col = [L2[r][i] for r in range(n)]
        assert sorted(row) == list(range(n)), f"L2 row {i} not Latin"
        assert sorted(col) == list(range(n)), f"L2 col {i} not Latin"
    
    # Verify orthogonality:
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            assert pair not in pairs, f"Duplicate pair {pair} at ({i},{j})"
            pairs.add(pair)
    assert len(pairs) == 100, f"Only {len(pairs)} distinct pairs"
    
    # Return as list of lists of ints
    result = [L1, L2]
    return result
