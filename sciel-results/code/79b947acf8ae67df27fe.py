import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 3, 3, 3
    rank = 21
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Laser method construction for 3x3 matrix multiplication
    # Using G = Z_3 x Z_3, |G| = 9
    # omega = exp(2*pi*i/3) but we use algebraic: omega^3 = 1, 1+omega+omega^2 = 0
    # We'll use the explicit coefficient structure
    
    # The 21 terms:
    # Terms 0-8: "core" terms indexed by (a,b) in Z_3 x Z_3
    # Terms 9-20: "correction" terms
    
    omega = Fraction(1)  # placeholder, we'll use explicit values
    
    # For the laser method, we need the Fourier transform of Z_3 x Z_3
    # Characters: chi_{a,b}(x,y) = omega^(a*x + b*y) where omega is primitive 3rd root of unity
    # Since we need exact rational/algebraic numbers, let's use the structure directly
    
    # Actually, let me use the explicit integer/half-integer construction
    # The rank-21 decomposition for 3x3 matmul:
    
    # Core terms (9): for each (s,t) in Z_3 x Z_3
    # U[i*3+j, core(s,t)] = 1 if i-j ≡ s (mod 3)
    # V[j*3+k, core(s,t)] = 1 if j-k ≡ t (mod 3)  
    # W[k*3+i, core(s,t)] = 1 if ... 
    
    # This is getting complex. Let me use a verified construction.
    
    # Known rank-21 construction:
    # We'll build it and verify
    
    col = 0
    # 9 "diagonal" terms
    for s in range(3):
        for t in range(3):
            for i in range(3):
                for j in range(3):
                    if (i - j) % 3 == s:
                        U[i * 3 + j][col] = Fraction(1)
                for k in range(3):
                    if (j - k) % 3 == t:
                        V[j * 3 + k][col] = Fraction(1)
                for k in range(3):
                    if (k - i) % 3 == (s + t) % 3:
                        W[k * 3 + i][col] = Fraction(1)
            col += 1
    
    # 12 "off-diagonal" correction terms
    # These handle the cases where the simple diagonal doesn't work
    # For each pair of distinct residues
    
    # Actually, let me verify this works first before adding more terms
    # Let me check if 9 terms suffice (they don't for 3x3)
    
    # The issue: with only 9 terms, we can't get rank 9 (lower bound is 19)
    # We need more terms
    
    # Let me restart with a cleaner approach using the actual laser method
    
    # Reset
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Use the explicit rank-21 construction from literature
    # The decomposition uses the following structure:
    
    # For the 3x3 case, we can use the following 21 rank-1 tensors:
    
    # Let me use a systematic approach:
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Construction using Z_3 x Z_3 laser method:
    # Let omega be a primitive cube root of unity (we'll use exact algebraic)
    # But since we need rational/half-integer entries, let's use a different approach
    
    # Actually, the laser method can give rational coefficients!
    # Let me use the explicit construction:
    
    # Term types:
    # Type 1 (3 terms): "pure diagonal" 
    # Type 2 (18 terms): "laser" terms
    
    # Let me just implement a verified rank-21 construction
    
    # Here's a known construction:
    # We use the fact that 3x3 matmul has rank <= 21
    
    # Let me build it carefully:
    
    # The 21 terms are indexed by (a, b) where a in {0,1,2}, b in {0,1,2} 
    # plus additional terms
    
    # I'll use the following verified approach:
    # Generate candidate decompositions and verify
    
    # For now, let me use a simple but correct rank-21 construction:
    
    # Method: Use the "laser" with G = Z_3 x Z_3
    # The 9 group elements give 9 base terms
    # The remaining 12 terms correct the off-diagonal entries
    
    # Base terms (r = 0..8), indexed by (g1, g2) in Z_3 x Z_3:
    # U[i*3+j, r] = 1 if (i-j) mod 3 == g1
    # V[j*3+k, r] = 1 if (j-k) mod 3 == g2  
    # W[k*3+i, r] = 1 if (k-i) mod 3 == (g1+g2) mod 3
    # But this gives sum = 1 only when conditions align, not always 1
    
    # The problem: for a given (i,j,k), the product is 1 only if all three conditions hold
    # Otherwise it's 0. We need the SUM over r to be 1.
    
    # For fixed (i,j,k), let d1 = (i-j)%3, d2 = (j-k)%3, d3 = (k-i)%3
    # Note d1 + d2 + d3 ≡ 0 (mod 3)
    # The term r=(g1,g2) contributes 1 iff g1=d1 AND g2=d2 AND (g1+g2)%3=d3
    # Since d1+d2 ≡ -d3 (mod 3), this is consistent
    # So exactly ONE base term contributes 1 for each (i,j,k)
    # Wait, that means the 9 base terms ALREADY give the correct decomposition!
    # But rank 9 is below the known lower bound of 19...
    
    # Let me re-examine. The condition for W is (k-i)%3 == (g1+g2)%3
    # But (k-i)%3 = -(i-k)%3 = -((i-j)+(j-k))%3 = -(d1+d2)%3
    # And (g1+g2)%3 should equal (k-i)%3 = -(d1+d2)%3
    # So we need g1+g2 ≡ -(d1+d2) (mod 3)
    # But we also need g1 = d1 and g2 = d2
    # So we need d1+d2 ≡ -(d1+d2) (mod 3), i.e., 2(d1+d2) ≡ 0 (mod 3)
    # This is only true when d1+d2 ≡ 0 (mod 3)
    
    # So the base terms only work when d1+d2 ≡ 0 (mod 3)!
    # For other cases, we need correction terms.
    
    # Let me fix the W condition:
    # W[k*3+i, r] should be 1 if (k-i)%3 == some function of g1,g2
    
    # Actually, the laser method uses:
    # U[i,j,r] depends on i-j (the "A" index difference)
    # V[j,k,r] depends on j-k (the "B" index difference)
    # W[k,i,r] depends on k-i (the "C" index difference)
    
    # And the character condition is: chi(g1 * g2) = chi(k-i)
    # In additive notation: g1 + g2 ≡ k-i (mod 3)
    
    # But k-i = (k-j) + (j-i) = -(j-k) + (i-j) = -d2 + d1
    # So we need g1 + g2 ≡ d1 - d2 (mod 3)
    # With g1 = d1, g2 = d2: d1 + d2 ≡ d1 - d2 (mod 3) => 2*d2 ≡ 0 => d2 ≡ 0
    
    # This still doesn't work for all cases. The laser method is more subtle.
    
    # Let me use a completely different, verified construction.
    
    # I'll implement the rank-21 decomposition using explicit coefficients.
    
    # After careful consideration, here's a working rank-21 construction:
    
    # Reset all to zero
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Use the following construction:
    # 21 terms, each with specific U, V, W patterns
    
    # Let me use numpy to verify and build this
    
    # For the 3x3 case, I'll use the explicit known decomposition
    
    # Terms 0-2: "row" terms
    for a in range(3):
        r = a
        for i in range(3):
            for j in range(3):
                if i == a:
                    U[i*3+j][r] = Fraction(1)
        for j in range(3):
            for k in range(3):
                if j == a:
                    V[j*3+k][r] = Fraction(1)
        for k in range(3):
            for i in range(3):
                if k == a:
                    W[k*3+i][r] = Fraction(1)
    
    # Terms 3-5: "column" terms  
    for a in range(3):
        r = 3 + a
        for i in range(3):
            for j in range(3):
                if j == a:
                    U[i*3+j][r] = Fraction(1)
        for j in range(3):
            for k in range(3):
                if k == a:
                    V[j*3+k][r] = Fraction(1)
        for k in range(3):
            for i in range(3):
                if i == a:
                    W[k*3+i][r] = Fraction(1)
    
    # This is getting too complex without verification. Let me use a different strategy.
    # I'll implement a rank-21 decomposition and verify it numerically.
    
    # Let me use the explicit construction from the laser method paper.
    
    # For 3x3 matrix multiplication, the laser method with G=Z_3 x Z_3 gives:
    # Rank = 9 + 3*3 = 18? No...
    
    # Let me just use a verified rank-21 construction by solving the system.
    
    # Actually, let me try a simpler approach: use the known result that
    # 3x3 matmul has rank 21 and construct it explicitly.
    
    # Here's a working construction:
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # 9 terms: for each (s,t) in Z_3 x Z_3
    for s in range(3):
        for t in range(3):
            r = s * 3 + t
            for i in range(3):
                for j in range(3):
                    if (i - j) % 3 == s:
                        U[i*3+j][r] = Fraction(1)
            for j in range(3):
                for k in range(3):
                    if (j - k) % 3 == t:
                        V[j*3+k][r] = Fraction(1)
            for k in range(3):
                for i in range(3):
                    if (k - i) % 3 == (s + t) % 3:
                        W[k*3+i][r] = Fraction(1)
    
    # 12 terms: correction terms for the off-diagonal cases
    # For each pair (a,b) with a != b in Z_3, and for each "type"
    # These handle the cases where s+t != (k-i)%3
    
    # The 12 correction terms:
    # For each a in {0,1,2}, b in {0,1,2}, a != b:
    #   4 terms per (a,b) pair? That's 6*4=24, too many.
    
    # Let me use 12 terms differently.
    
    # Actually, let me verify the 9-term construction first:
    # For (i,j,k), the sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
    # = sum over (s,t) where (i-j)%3=s, (j-k)%3=t, (k-i)%3=(s+t)%3
    # Let d1=(i-j)%3, d2=(j-k)%3, d3=(k-i)%3
    # Note: d1+d2+d3 ≡ 0 (mod 3)
    # The term (s,t) contributes 1 iff s=d1, t=d2, and (s+t)%3=d3
    # i.e., (d1+d2)%3 = d3
    # But d3 = -d1-d2 (mod 3)
    # So we need d1+d2 ≡ -d1-d2 (mod 3) => 2(d1+d2) ≡ 0 (mod 3)
    # => d1+d2 ≡ 0 (mod 3)
    
    # So the 9 terms give 1 when d1+d2 ≡ 0 (mod 3), else 0.
    # We need to add terms that give 1 when d1+d2 != 0 (mod 3).
    
    # When d1+d2 ≡ 1 (mod 3): need contribution of 1
    # When d1+d2 ≡ 2 (mod 3): need contribution of 1
    
    # Let me add 12 more terms to handle these cases.
    
    # For d1+d2 ≡ 1: (d1,d2) can be (0,1),(1,0),(2,2)
    # For d1+d2 ≡ 2: (d1,d2) can be (0,2),(1,1),(2,0)
    
    # I'll create 6 terms for d1+d2≡1 and 6 for d1+d2≡2, but that's 12 total.
    
    # Actually, let me use a different structure for the correction terms.
    
    # For the correction terms, I'll use:
    # For each c in {1,2} (the value of d1+d2 mod 3):
    #   For each d1 in {0,1,2}:
    #     d2 = (c - d1) % 3
    #     Create a term that contributes 1 when (d1,d2) is this specific pair
    #     But we need the product U*V*W to be 1 for ALL (i,j,k) with this (d1,d2)
    
    # This is tricky. Let me use a different approach.
    
    # I'll use the following 12 correction terms:
    # For a in {0,1,2}, b in {0,1,2}:
    #   Term 9 + a*3 + b:
    #   U[i,j] = 1 if i-j ≡ a (mod 3) AND i+j ≡ b (mod 3)  [this selects specific (i,j)]
    #   V[j,k] = 1 if j-k ≡ something
    #   W[k,i] = 1 if k-i ≡ something
    
    # This is getting too complicated. Let me use a numerical approach to find the decomposition.
    
    # For now, let me return the schoolbook rank-27 as a fallback and add the laser method
    # terms that I can verify work.
    
    # Let me try a completely different, simpler rank-21 construction:
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # I'll use the following explicit 21-term construction:
    
    # Terms 0-8: "block diagonal" terms
    # For each a in {0,1,2}:
    #   Term 3a: U has 1s where i==a, V has 1s where j==a, W has 1s where k==a
    #   Term 3a+1: U has 1s where j==a, V has 1s where k==a, W has 1s where i==a  
    #   Term 3a+2: U has 1s where i==j, V has 1s where j==k, W has 1s where k==i (for this a)
    
    # This is still not clean. Let me just implement a verified construction.
    
    # After much deliberation, here's a clean rank-21 construction:
    
    # I'll use the fact that the 3x3 matmul tensor can be written as a sum of 21 rank-1 tensors.
    # The construction is based on the following:
    
    # Let's define the 21 terms explicitly:
    
    # Group 1: 9 terms (the "laser" core)
    for s in range(3):
        for t in range(3):
            r = s * 3 + t
            for i in range(3):
                for j in range(3):
                    if (i - j) % 3 == s:
                        U[i*3+j][r] = Fraction(1)
            for j in range(3):
                for k in range(3):
                    if (j - k) % 3 == t:
                        V[j*3+k][r] = Fraction(1)
            for k in range(3):
                for i in range(3):
                    if (k - i) % 3 == (s + t) % 3:
                        W[k*3+i][r] = Fraction(1)
    
    # Group 2: 12 terms (corrections)
    # For each c in {1, 2} (target value of d1+d2 mod 3):
    #   For each d1 in {0, 1, 2}:
    #     d2 = (c - d1) % 3
    #     d3 = (3 - d1 - d2) % 3  # which is (d1+d2) % 3 = c... wait
    #     Actually d3 = (k-i)%3 = -(d1+d2)%3 = (3-c)%3
    #     
    #     Create a term that gives 1 for all (i,j,k) with (i-j)%3=d1, (j-k)%3=d2
    #     But we need U[i,j]*V[j,k]*W[k,i] = 1 for these, and 0 otherwise
    #     
    #     U[i,j] = 1 if (i-j)%3 == d1
    #     V[j,k] = 1 if (j-k)%3 == d2  
    #     W[k,i] = 1 if (k-i)%3 == (3-c)%3
    #     
    #     But this would give 1 for ALL (i,j,k) with those differences, which is what we want!
    #     However, this term would ALSO contribute to cases where d1+d2 == 0 (mod 3) if
    #     (3-c)%3 happens to equal the required value... but c is 1 or 2, so (3-c)%3 is 2 or 1,
    #     while for d1+d2==0, we need d3=0. So no overlap!
    
    for c in [1, 2]:
        for d1 in range(3):
            d2 = (c - d1) % 3
            d3 = (3 - c) % 3  # = (3-c)%3, which is the required (k-i)%3
            r = 9 + (c-1)*3 + d1
            for i in range(3):
                for j in range(3):
                    if (i - j) % 3 == d1:
                        U[i*3+j][r] = Fraction(1)
            for j in range(3):
                for k in range(3):
                    if (j - k) % 3 == d2:
                        V[j*3+k][r] = Fraction(1)
            for k in range(3):
                for i in range(3):
                    if (k - i) % 3 == d3:
                        W[k*3+i][r] = Fraction(1)
    
    # Now verify: for each (i,j,k), compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # and check it equals 1
    
    # Let me verify numerically
    U_np = np.array([[float(x) for x in row] for row in U])
    V_np = np.array([[float(x) for x in row] for row in V])
    W_np = np.array([[float(x) for x in row] for row in W])
    
    max_err = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for r in range(rank):
                    val += U_np[i*3+j, r] * V_np[j*3+k, r] * W_np[k*3+i, r]
                if abs(val - 1) > 1e-10:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook rank 27
        rank = 27
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if time.time() - start_time > DEADLINE - 5.0:
                        break
                    U[i * m + j][column] = Fraction(1)
                    V[j * p + k][column] = Fraction(1)
                    W[k * n + i][column] = Fraction(1)
                    column += 1
    else:
        # Convert Fractions to int where possible for cleaner output
        U = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
        V = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
        W = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
