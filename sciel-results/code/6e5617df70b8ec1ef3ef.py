"""Construct mutually orthogonal Latin squares of order 10."""
# EVOLVE-BLOCK-START
import time
import random

N = 10
# Map (a,b) in Z5 x Z2 to integer 0..9
def enc(a, b):
    return a * 2 + b

def dec(x):
    return (x // 2, x % 2)

# Precompute the 10 symbols as (a,b) pairs
SYMBOLS = [(a, b) for a in range(5) for b in range(2)]
SYM_TO_INT = {s: i for i, s in enumerate(SYMBOLS)}

# We'll use the group Z5 x Z2.
# A Latin square can be defined by L(i,j) = i + j (group addition) but we need to index rows/cols by 0..9.
# Let's represent row i as (i//2, i%2) and col j as (j//2, j%2)? No, that's not a group action on the indices directly.
# Actually, the standard construction for order 10 (which is 5*2) uses the fact that Z10 is not a field.
# However, we can construct a set of MOLS using the structure of Z5 x Z2.
# Let the set of symbols be G = Z5 x Z2.
# Let rows be indexed by x in G and columns by y in G.
# A Latin square is a function L: G x G -> G such that for each x, L(x, .) is a bijection and for each y, L(., y) is a bijection.
# The simplest is L(x,y) = x + y. This is a Latin square.
# Another is L(x,y) = x + c*y for some constant c?
# In an abelian group, L_c(x,y) = x + c*y is a Latin square if the map y -> c*y is a bijection.
# In Z5 x Z2, the multiplication by c=(c1,c2) maps (a,b) to (c1*a mod 5, c2*b mod 2).
# This is a bijection iff c1 is not 0 mod 5 and c2 is not 0 mod 2.
# So c1 in {1,2,3,4} and c2 in {1}.
# So possible c values are (1,1), (2,1), (3,1), (4,1).
# Let's check orthogonality of L_c1 and L_c2.
# L_c1(x,y) = x + c1*y
# L_c2(x,y) = x + c2*y
# The pair (x + c1*y, x + c2*y) must be distinct for all (x,y).
# Let x + c1*y = x' + c1*y' and x + c2*y = x' + c2*y'.
# Subtracting: c1*(y-y') = x-x' and c2*(y-y') = x-x'.
# So (c1-c2)*(y-y') = 0.
# Since the group is Z5 x Z2, and c1, c2 are in Z5 x Z2.
# c1-c2 is in Z5 x Z2.
# If c1-c2 is not zero, does it have to be a unit (bijective multiplier)?
# In Z5 x Z2, an element (d1, d2) is a unit (i.e., multiplication by it is a bijection) iff d1 != 0 and d2 != 0.
# Here c1=(k1,1) and c2=(k2,1).
# c1-c2 = (k1-k2, 0).
# The second component is 0. So multiplication by (k1-k2, 0) is NOT a bijection (it maps everything to the subgroup Z5 x {0}).
# Wait, if (c1-c2)*(y-y') = 0, and c1-c2 = (d, 0), then (d*(y1-y1'), 0) = (0,0).
# So d*(y1-y1') = 0 in Z5. If d != 0, then y1-y1'=0.
# But what about y2-y2'? The equation gives no constraint on y2-y2'.
# So we can have y1=y1' but y2 != y2'.
# Then y != y'. But x = x'.
# Let's check:
# x + c1*y = x + (k1*y1, y2)
# x' + c1*y' = x + (k1*y1, y2')
# These are equal iff y2 = y2'.
# So if y2 != y2', the values are different.
# Wait, I made a mistake.
# L_c(x,y) = x + c*y.
# If y=(y1,y2), c=(k,1), then c*y = (k*y1 mod 5, 1*y2 mod 2) = (k*y1, y2).
# So L_c(x,y) = (x1 + k*y1, x2 + y2).
# Let's check if L_c is a Latin square.
# For fixed x, y -> (x1+k*y1, x2+y2). Since k!=0 and 1!=0, this is a bijection.
# For fixed y, x -> (x1+k*y1, x2+y2). Since 1!=0, this is a bijection.
# So L_c is a Latin square for c=(k,1) with k in {1,2,3,4}.
# Now orthogonality of L_{k1} and L_{k2}.
# Pair: (x1+k1*y1, x2+y2) and (x1+k2*y1, x2+y2).
# Wait, the second component is the same: x2+y2.
# So the pair is (A, B) where A = x1+k1*y1 and B = x1+k2*y1? No.
# L_{k1}(x,y) = (x1+k1*y1, x2+y2)
# L_{k2}(x,y) = (x1+k2*y1, x2+y2)
# The ordered pair of symbols is ((x1+k1*y1, x2+y2), (x1+k2*y1, x2+y2)).
# Let's see if this is unique for each (x,y).
# Suppose ((x1+k1*y1, x2+y2), (x1+k2*y1, x2+y2)) = ((x1'+k1*y1', x2'+y2'), (x1'+k2*y1', x2'+y2')).
# From the second components of the symbols: x2+y2 = x2'+y2' and x2+y2 = x2'+y2'. This is the same equation.
# From the first components of the symbols: x1+k1*y1 = x1'+k1*y1' and x1+k2*y1 = x1'+k2*y1'.
# Subtracting: (k1-k2)*(y1-y1') = 0.
# Since k1 != k2, k1-k2 is non-zero in Z5, so y1=y1'.
# Then x1 = x1'.
# From x2+y2 = x2'+y2', we have x2-x2' = y2'-y2.
# We need to determine if (x2, y2) is uniquely determined.
# We have one equation x2+y2 = const. There are 4 solutions for (x2,y2) in Z2 x Z2.
# So the pair of symbols is NOT unique.
# Thus L_{k1} and L_{k2} are NOT orthogonal.
#
# We need a different construction.
# Let's use the standard construction for MOLS of order 10.
# It is known that there are at most 9 MOLS of order 10.
# A known set of MOLS can be constructed using the finite field GF(5) and the parity.
# Let's define the symbols as (a,b) where a in Z5, b in Z2.
# Let rows be (u,v) in Z5 x Z2 and cols be (s,t) in Z5 x Z2.
# Define L(u,v; s,t) = (u + s, v + t + f(s,t)) ? No.
#
# Let's try L(u,v; s,t) = (u + s, v + t + s*t) ? No, s*t is in Z5, t is in Z2.
#
# Let's look at the known MOLS(10).
# They can be constructed from the 5x5 MOLS of GF(5) by "doubling".
# Actually, a simpler way:
# Let the symbols be 0..9.
# Let the square be S[i][j].
# We can use the fact that Z10 is isomorphic to Z5 x Z2.
# Let i = 2*a + b, j = 2*c + d, where a,c in Z5, b,d in Z2.
# Let the symbol be 2*e + f.
#
# A known construction:
# S[i][j] = ( (a + c) * 2 + (b + d) ) ? This is just addition in Z10.
# Let's try S[i][j] = ( (a + c) * 2 + (b + d + a*c) ) ? No, a*c is in Z5.
#
# Let's use the group Z5 x Z2.
# Let the square be L(u,v; s,t) = (u + s, v + t + u*t) ? No.
#
# Let's try:
# L(u,v; s,t) = (u + s, v + t + s*t) is not well-defined because s*t is in Z5 and t is in Z2.
#
# Let's use the following:
# L(u,v; s,t) = (u + s, v + t) is one square.
# L(u,v; s,t) = (u + 2*s, v + t) is another?
# Check orthogonality:
# (u+s, v+t) and (u+2s, v+t).
# Second component is the same. So not orthogonal.
#
# We need the second component to depend on s and t differently.
# Let L(u,v; s,t) = (u + s, v + t + s*t) ? No.
#
# Let's try:
# L(u,v; s,t) = (u + s, v + t + (s mod 2)*(t mod 2)) ? No, s is in Z5.
#
# Let's use the polynomial method over GF(5).
# Let the symbols be (a,b) in GF(5) x Z2.
# Let rows be (x,y) in GF(5) x Z2 and cols be (u,v) in GF(5) x Z2.
# Define L_k(x,y; u,v) = (x + k*u, y + v + k*u*v) ? No, k*u*v is in GF(5), but y+v is in Z2.
#
# Let's try:
# L_k(x,y; u,v) = (x + k*u, y + v)
# This doesn't work as before.
#
# Let's try:
# L_k(x,y; u,v) = (x + k*u, y + v + (k*u mod 2)) ? No.
#
# Actually, there is a known construction using the fact that 10 = 5 * 2.
# The MOLS of order 10 can be derived from the MOLS of order 5 by "doubling" each symbol.
# But we need to be careful.
#
# Let's use the following construction:
# Let the symbols be (a,b) where a in Z5, b in Z2.
# Let the square be S[i][j] where i = 2*a + b, j = 2*c + d.
# Define S[i][j] = ( (a + c) mod 5, (b + d + a*c) mod 2 ).
# Let's check if this is a Latin square.
# For fixed i=(a,b), as j=(c,d) varies, (a+c, b+d+a*c) varies.
# a+c covers all Z5.
# For a fixed c, b+d+a*c covers all Z2 as d varies.
# So for each a+c, there are 2 values of b+d+a*c.
# So the map (c,d) -> (a+c, b+d+a*c) is a bijection.
# Similarly for fixed j.
# So it is a Latin square.
#
# Now let's check orthogonality with another square.
# Let S2[i][j] = ( (a + 2*c) mod 5, (b + d + 2*a*c) mod 2 ).
# Pair: ( (a+c, b+d+a*c), (a+2c, b+d+2ac) ).
# Let's see if this is unique.
# Let (a+c, b+d+a*c) = (a'+c', b'+d'+a'*c') and (a+2c, b+d+2ac) = (a'+2c', b'+d'+2a'*c').
# From the first components: a+c = a'+c' and a+2c = a'+2c'.
# Subtracting: c = c'. Then a = a'.
# From the second components: b+d+a*c = b'+d'+a'*c' and b+d+2ac = b'+d'+2a'*c'.
# Since a=a', c=c', we have a*c = a'*c'.
# So b+d = b'+d' and b+d = b'+d'.
# So b+d = b'+d'.
# But we need to determine (b,d) uniquely.
# We have b+d = const. There are 2 solutions.
# So not orthogonal.
#
# Let's try S2[i][j] = ( (a + c) mod 5, (b + d + c) mod 2 ).
# Pair: ( (a+c, b+d+a*c), (a+c, b+d+c) ).
# First components are the same. So not orthogonal.
#
# Let's try S2[i][j] = ( (2*a + c) mod 5, (b + d + a*c) mod 2 ).
# Pair: ( (a+c, b+d+a*c), (2a+c, b+d+ac) ).
# Let (a+c, b+d+ac) = (a'+c', b'+d'+a'c') and (2a+c, b+d+ac) = (2a'+c', b'+d'+a'c').
# From the second components: b+d+ac = b'+d'+a'c'.
# From the first components: a+c = a'+c' and 2a+c = 2a'+c'.
# Subtracting: a = a'. Then c = c'.
# Then b+d = b'+d'.
# So not orthogonal.
#
# It seems hard to find two orthogonal squares this way.
#
# Let's use the known result that there are 9 MOLS of order 10.
# A known set is given by:
# S_k(i,j) = ( (i + k*j) mod 5, (i//2 + j//2 + k*(i%2)*(j%2)) mod 2 ) ? No.
#
# Let's use the following construction from the literature:
# Let the symbols be (a,b) in Z5 x Z2.
# Let the square be S_k(i,j) where i = 2*a + b, j = 2*c + d.
# S_k(i,j) = ( (a + k*c) mod 5, (b + d + (k*c) mod 2) ) ? No.
#
# Let's try:
# S_k(i,j) = ( (a + k*c) mod 5, (b + d + (a*d + c*b) mod 2) ) ?
#
# Actually, let's use the fact that Z5 x Z2 is a group.
# Let the square be L_k(x,y) = x + k*y, where x,y are in Z5 x Z2.
# We saw that L_k and L_m are not orthogonal if k,m are in Z5 x Z2 with k-m not a unit.
# But if we choose k,m such that k-m is a unit, they are orthogonal.
# In Z5 x Z2, the units are (k1,k2) with k1!=0, k2!=0.
# So k1 in {1,2,3,4}, k2 in {1}.
# So the units are (1,1), (2,1), (3,1), (4,1).
# Let's take k=(1,1) and m=(2,1).
# k-m = (-1,0) = (4,0). This is not a unit.
# So L_k and L_m are not orthogonal.
#
# Wait, my earlier analysis was wrong.
# Let's re-check.
# L_k(x,y) = x + k*y.
# L_m(x,y) = x + m*y.
# Pair: (x + k*y, x + m*y).
# Let x + k*y = x' + k*y' and x + m*y = x' + m*y'.
# Subtracting: (k-m)*y = (k-m)*y'.
# So (k-m)*(y-y') = 0.
# If k-m is a unit, then y=y', so x=x'.
# So if k-m is a unit, the squares are orthogonal.
# In Z5 x Z2, the units are (k1,k2) with k1!=0, k2!=0.
# So we need k-m to be a unit.
# Let k=(1,1), m=(1,0)? No, m must be such that L_m is a Latin square, so m must be a unit.
# So m=(k1,k2) with k1!=0, k2!=0.
# So m=(1,1), (2,1), (3,1), (4,1).
# k-m for k=(1,1), m=(2,1) is (-1,0) = (4,0), not a unit.
# So no two of these are orthogonal.
#
# So we need a different construction.
#
# Let's use the following:
# Let the symbols be (a,b) in Z5 x Z2.
# Let the square be S_k(i,j) where i = 2*a + b, j = 2*c + d.
# S_k(i,j) = ( (a + k*c) mod 5, (b + d + k*(a*d + c*b)) mod 2 ) ?
#
# Let's try a simpler approach.
# We know that there are 9 MOLS of order 10.
# We can hardcode a known set of MOLS.
#
# Let's generate them using the following method:
# Let the symbols be 0..9.
# Let S_k[i][j] = ( (i + k*j) % 10 ) ? No, this is not a Latin square for k not coprime to 10.
#
# Let's use the group Z5 x Z2.
# Let the square be S_k(i,j) = ( (i//2 + k*(j//2)) % 5, (i%2 + j%2 + k*(i//2)*(j//2) % 2) ) ?
#
# Let's try:
# S_k(i,j) = ( (i//2 + k*(j//2)) % 5, (i%2 + j%2) )
# This is not orthogonal.
#
# Let's use the following construction:
# S_k(i,j) = ( (i//2 + k*(j//2)) % 5, (i%2 + j%2 + (k*(i//2)*(j//2)) % 2) )
#
# Let's check if this is a Latin square.
# For fixed i, as j varies, j//2 covers 0..4 and j%2 covers 0,1.
# So (i//2 + k*(j//2)) % 5 covers all Z5 if k!=0.
# And (i%2 + j%2 + k*(i//2)*(j//2) % 2) covers all Z2.
# So it is a Latin square.
#
# Let's check orthogonality of S_k and S_m.
# This is complex. Let's just generate them and check.
#
# Let's write a function to generate S_k and check orthogonality.

def generate_square(k):
    S = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            a = i // 2
            b = i % 2
            c = j // 2
            d = j % 2
            # Symbol is ( (a + k*c) % 5, (b + d + (k*a*c) % 2) )
            sym_a = (a + k * c) % 5
            sym_b = (b + d + (k * a * c) % 2) % 2
            S[i][j] = sym_a * 2 + sym_b
    return S

def check_latin(S):
    for i in range(N):
        if sorted(S[i]) != list(range(N)):
            return False
    for j in range(N):
        col = [S[i][j] for i in range(N)]
        if sorted(col) != list(range(N)):
            return False
    return True

def check_orthogonal(S1, S2):
    pairs = set()
    for i in range(N):
        for j in range(N):
            p = (S1[i][j], S2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    start = time.time()
    deadline = start + 80  # 80s deadline, 10s margin
    
    # Generate squares for k = 1, 2, 3, 4
    squares = []
    for k in [1, 2, 3, 4]:
        S = generate_square(k)
        if not check_latin(S):
            continue
        # Check orthogonality with existing squares
        orthogonal = True
        for S2 in squares:
            if not check_orthogonal(S, S2):
                orthogonal = False
                break
        if orthogonal:
            squares.append(S)
        
        if time.time() > deadline:
            break
    
    # If we have at least 1 square, return it.
    # If we have more, return all.
    # We need to return a list of lists of lists of ints.
    result = []
    for S in squares:
        result.append([list(row) for row in S])
    
    # If no squares found (should not happen), return a trivial one.
    if not result:
        result = [[[(i+j)%N for j in range(N)] for i in range(N)]]
    
    return result
# EVOLVE-BLOCK-END
