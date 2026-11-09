"""Program to find a set of distinct integers in 2..40 with all subset products distinct."""

def construct():
    # The key insight: for all subset products to be distinct, the prime exponent vectors
    # of the selected numbers must be linearly independent over the integers.
    # This means no non-trivial integer combination of the exponent vectors sums to zero.
    #
    # The maximum set of size 16 is:
    # {2, 3, 4, 5, 7, 9, 11, 13, 14, 16, 17, 19, 23, 25, 29, 31}
    #
    # Let's verify this is valid:
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 (11 primes)
    # Powers: 4=2^2, 9=3^2, 16=2^4, 25=5^2
    # Mixed: 14=2*7
    #
    # The exponent vectors (in order of primes 2,3,5,7,11,13,17,19,23,29,31):
    # 2:  (1,0,0,0,0,0,0,0,0,0,0)
    # 3:  (0,1,0,0,0,0,0,0,0,0,0)
    # 4:  (2,0,0,0,0,0,0,0,0,0,0)
    # 5:  (0,0,1,0,0,0,0,0,0,0,0)
    # 7:  (0,0,0,1,0,0,0,0,0,0,0)
    # 9:  (0,2,0,0,0,0,0,0,0,0,0)
    # 11: (0,0,0,0,1,0,0,0,0,0,0)
    # 13: (0,0,0,0,0,1,0,0,0,0,0)
    # 14: (1,0,0,1,0,0,0,0,0,0,0)
    # 16: (4,0,0,0,0,0,0,0,0,0,0)
    # 17: (0,0,0,0,0,0,1,0,0,0,0)
    # 19: (0,0,0,0,0,0,0,1,0,0,0)
    # 23: (0,0,0,0,0,0,0,0,1,0,0)
    # 25: (0,0,2,0,0,0,0,0,0,0,0)
    # 29: (0,0,0,0,0,0,0,0,0,1,0)
    # 31: (0,0,0,0,0,0,0,0,0,0,1)
    #
    # We need to check linear independence. The vectors for primes 11,13,17,19,23,29,31
    # are standard basis vectors, so they're independent and don't interact with others.
    # The remaining vectors (for 2,3,4,5,7,9,14,16,25) involve primes 2,3,5,7.
    #
    # Let's check if these 9 vectors are independent:
    # v2=(1,0,0,0), v3=(0,1,0,0), v4=(2,0,0,0), v5=(0,0,1,0), v7=(0,0,0,1),
    # v9=(0,2,0,0), v14=(1,0,0,1), v16=(4,0,0,0), v25=(0,0,2,0)
    #
    # These span a 4D space. We have 9 vectors in 4D, so they can't be independent!
    # Wait, that's a problem. Let me reconsider.
    #
    # Actually, the full set has 16 vectors in 11D (one dimension per prime up to 31).
    # The 7 pure prime vectors (11,13,17,19,23,29,31) give 7 independent dimensions.
    # The remaining 9 vectors use primes 2,3,5,7, so they live in a 4D subspace.
    # So we can have at most 4 independent vectors from that group.
    # 7 + 4 = 11, but we have 16 total. So we need the 4D subspace vectors to be
    # independent among themselves AND independent from the pure prime vectors.
    # The pure prime vectors are in disjoint dimensions, so they're automatically
    # independent from the 2,3,5,7 vectors.
    # But the 9 vectors in the 4D space can't be independent (max rank 4).
    # So my set of 16 is INVALID!
    #
    # Let me reconsider. I need at most 4 vectors from {2,3,4,5,7,8,9,14,15,16,18,20,21,25,27,28,30,32,35,36}
    # that are linearly independent in the 4D space of primes {2,3,5,7}.
    #
    # Good choices: 2=(1,0,0,0), 3=(0,1,0,0), 5=(0,0,1,0), 7=(0,0,0,1)
    # These are the standard basis. Then I can add the 7 larger primes.
    # Total: 4 + 7 = 11.
    #
    # Can I do better? What if I use 4,9,25,49 instead? No, 49>40.
    # What about 2,3,5,7,11,13,17,19,23,29,31 = 11 primes.
    # Plus I can add at most 3 more from the composite numbers, but they must be
    # independent from the prime vectors. Since composites use the same primes,
    # they're in the same 4D space. With 4 primes already using all 4 dimensions,
    # no additional composite can be independent.
    #
    # So max is 11? But the current program gets 16 with score 1.3333 = 16/12.
    # Wait, score = value/seed. If seed=12, then 16/12=1.3333. So the seed returns 12.
    #
    # Let me re-examine. The issue is that subset products must be distinct.
    # This is equivalent to: for any two distinct subsets A and B, prod(A) != prod(B).
    # This is equivalent to: the multiplicative semigroup generated is free, which
    # means the exponent vectors are linearly independent over the non-negative integers...
    # No, it means that the only integer linear combination that gives zero is the trivial one.
    # Actually, it means the vectors are linearly independent over Z (integers).
    #
    # For the set to have all subset products distinct, we need: if sum_{i in A} v_i = sum_{j in B} v_j
    # for disjoint subsets A, B, then A = B = empty.
    # This is equivalent to: the only integer linear combination sum c_i v_i = 0 with c_i in {-1,0,1}
    # is the trivial one.
    # This is weaker than linear independence over Z, which allows any integer coefficients.
    #
    # So we need the vectors to be "sum-free" or to have no non-trivial {-1,0,1}-combination equal to zero.
    # This is different from linear independence!
    #
    # Example: v1=(1,0), v2=(2,0), v3=(3,0). These are linearly dependent (all in 1D).
    # But is there a {-1,0,1} combination equal to zero? 
    # c1*v1 + c2*v2 + c3*v3 = 0 => c1 + 2*c2 + 3*c3 = 0 with c_i in {-1,0,1}.
    # Possible: c1=1, c2=0, c3=0 => 1 != 0. c1=-1,c2=0,c3=0 => -1 != 0.
    # c1=1,c2=-1,c3=0 => 1-2=-1 != 0. c1=-1,c2=1,c3=0 => -1+2=1 != 0.
    # c1=1,c2=0,c3=-1 => 1-3=-2 != 0. c1=-1,c2=0,c3=1 => -1+3=2 != 0.
    # c1=1,c2=-1,c3=0 => -1. c1=-1,c2=1,c3=0 => 1.
    # c1=1,c2=0,c3=-1 => -2. c1=-1,c2=0,c3=1 => 2.
    # c1=1,c2=-1,c3=0, etc.
    # Actually c1=1, c2=0, c3=0 gives 1. No combination gives 0 except trivial.
    # Wait: c1=1, c2=0, c3=0: sum=1. c1=-1,c2=0,c3=0: sum=-1.
    # c1=1,c2=-1,c3=0: 1-2=-1. c1=-1,c2=1,c3=0: -1+2=1.
    # c1=1,c2=0,c3=-1: 1-3=-2. c1=-1,c2=0,c3=1: -1+3=2.
    # c1=1,c2=-1,c3=0, c1=1,c2=0,c3=-1, etc.
    # What about c1=1, c2=0, c3=0, c4=... only 3 vectors.
    # I don't see a non-trivial {-1,0,1} combination giving 0.
    # So {2,4,8} (vectors (1,0),(2,0),(3,0)) might be valid!
    # Check: subset products of {2,4,8}: 1,2,4,8,8,16,32,64. Wait, 2*4=8 and 8=8. Collision!
    # So {2,4,8} is invalid because 2*4=8.
    #
    # So the condition is stronger. Let me think again.
    # prod(A) = prod(B) means prod(A \ B) = prod(B \ A) where A\B and B\A are disjoint.
    # So we need: no two disjoint non-empty subsets have the same product.
    #
    # For {2,4,8}: {2,4} and {8} are disjoint, both product 8. Invalid.
    #
    # So the condition is: the set is "product-distinct" or "multiplicatively free".
    #
    # Back to the original construction: using prime powers.
    # If we pick 2^a, 3^b, 5^c, 7^d, 11, 13, 17, 19, 23, 29, 31,
    # then each number has a unique "signature" in terms of which primes it uses.
    # The subset product is determined by the sum of exponents for each prime.
    # For distinctness, we need: no two subsets give the same exponent sums for all primes.
    #
    # If we use: 2, 4, 16 (2^1, 2^2, 2^4), 3, 9 (3^1, 3^2), 5, 25 (5^1, 5^2),
    # and 7, 11, 13, 17, 19, 23, 29, 31.
    # Total: 3+2+2+1+1+1+1+1+1+1 = 14 numbers.
    #
    # The exponent for prime 2 comes from {2,4,16} with exponents {1,2,4}.
    # Subset sums of {1,2,4} are all distinct (0-7).
    # Exponent for prime 3 comes from {3,9} with exponents {1,2}. Subset sums: 0,1,2,3.
    # Exponent for prime 5 comes from {5,25} with exponents {1,2}. Subset sums: 0,1,2,3.
    # Exponents for primes 7,11,13,17,19,23,29,31 are 0 or 1.
    #
    # So the full exponent vector is a tuple of (e2, e3, e5, e7, e11, e13, e17, e19, e23, e29, e31)
    # where e2 in {0..7}, e3 in {0..3}, e5 in {0..3}, others in {0,1}.
    # All combinations are distinct, so all subset products are distinct.
    #
    # That gives 14. Can we get 15 or 16?
    #
    # The current program returns 16. Let me see what it returns:
    # 2,4,16, 3,9, 5,25, 7,11,13,17,19,23,29,31 = 14 numbers.
    # Wait, the code says:
    # selections = {2:[1,2,4], 3:[1,2], 5:[1,2]}
    # So: 2,4,16, 3,9, 5,25, then 7,11,13,17,19,23,29,31.
    # That's 3+2+2+8 = 15 numbers.
    #
    # Wait, primes list has 12 primes: 2,3,5,7,11,13,17,19,23,29,31,37.
    # So it also includes 37. Total: 3+2+2+1+1+1+1+1+1+1+1 = 15.
    #
    # Hmm, 15 not 16. But score is 1.3333 = 16/12. So the seed must return 12.
    #
    # Let me just return the 15-element set and verify it's valid.

    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return result
