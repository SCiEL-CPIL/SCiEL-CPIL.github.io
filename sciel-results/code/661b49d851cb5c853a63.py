"""
Construct two Sidon sets with disjoint nonzero differences using a Singer difference set.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Construct Set A using a Singer difference set
    # We look for a prime power q such that q+1 divides 256, or close to it.
    # Singer difference sets exist for parameters (v, k, lambda) where v = q^2 + q + 1, k = q + 1, lambda = 1.
    # Here, we want a Sidon set in [1, 256]. A Sidon set is a (n, k, 1) Sidon set, which is equivalent to a difference set with lambda=1 if it's "perfect" in the cyclic group, but we are in integers.
    # Actually, a Singer difference set in Z_v has the property that every nonzero difference appears exactly once (up to sign? No, in cyclic groups, differences are directed).
    # In integers, if we take a Singer difference set D in Z_v and map it to [1, v], the differences might wrap around.
    # However, if v is small enough, we can fit it in [1, 256].
    
    # Let's try q=15 (prime power? No, 15 is not prime power).
    # q=16 (2^4). v = 16^2 + 16 + 1 = 256 + 16 + 1 = 273. Too big.
    # q=9 (3^2). v = 9^2 + 9 + 1 = 81 + 9 + 1 = 91. k = 10.
    # q=8 (2^3). v = 8^2 + 8 + 1 = 64 + 8 + 1 = 73. k = 9.
    # q=7 (prime). v = 49 + 7 + 1 = 57. k = 8.
    # q=5 (prime). v = 25 + 5 + 1 = 31. k = 6.
    # q=4 (2^2). v = 16 + 4 + 1 = 21. k = 5.
    # q=3 (prime). v = 9 + 3 + 1 = 13. k = 4.
    
    # Let's use q=9, v=91, k=10.
    # The Singer difference set in Z_91 can be constructed using finite fields.
    # We need a primitive element in GF(9^2) or similar? No, Singer difference set of order q is in Z_{q^2+q+1}.
    # Construction: Let F_q be the field. The Singer cycle is a cyclic subgroup of GL(2, q) of order q^2+q+1.
    # Alternatively, we can just use a known construction or search for a small Sidon set that is "dense".
    
    # Actually, a simpler approach for Set A:
    # Use a known optimal Sidon set. For n=91, max size is ~sqrt(91) ~ 9.5. So 10 is optimal.
    # Let's try to generate a Singer difference set for q=9.
    # The elements are {x * g^i} for i=0..k-1? No.
    # The Singer difference set D is a subset of Z_v such that every nonzero element of Z_v is a difference of exactly one pair (d1, d2) in D.
    # This implies |D| = k = q+1, and the number of differences is k(k-1) = (q+1)q = q^2+q = v-1.
    # So all v-1 nonzero residues in Z_v are covered exactly once as differences.
    # When mapped to integers [1, v], the differences a-b will be in {-v+1, ..., v-1}.
    # Specifically, if we take D subset of {0, ..., v-1}, the differences a-b mod v are all distinct and nonzero.
    # The actual integer differences a-b will be in (-v, v).
    # Since v=91, the differences are in [-90, 90] excluding 0.
    # This is a very "tight" set of differences. It uses the minimum possible number of difference values for a set of size k?
    # No, any Sidon set of size k has k(k-1) distinct nonzero differences.
    # The Singer set has exactly k(k-1) differences, and they are all distinct mod v.
    # In integers, they might collide if a-b and c-d differ by a multiple of v.
    # But since |a-b| < v, the only way a-b = c-d + mv is if m=0 (since |a-b| < v) or if the values wrap.
    # Wait, if a-b = 90 and c-d = -1, they are distinct.
    # The set of integer differences from a Singer set in Z_v (mapped to 0..v-1) is exactly the set of all nonzero residues mod v, represented as integers in (-v, v).
    # So Diff(A) will be { -90, ..., -1, 1, ..., 90 } minus some collisions? No, if they are distinct mod v, and the range is (-v, v), they are distinct integers.
    # So Diff(A) will have size k(k-1) = 90.
    # And the values are exactly the integers from -90 to 90 excluding 0?
    # Let's check: if D is a Singer difference set, then {a-b | a,b in D, a!=b} = Z_v \ {0}.
    # So the set of differences is {1, 2, ..., v-1} in Z_v.
    # Mapped to integers, these are {1, ..., 90} and {-90, ..., -1} (since -1 = 90 mod 91, etc).
    # So Diff(A) = { -90, ..., -1, 1, ..., 90 }.
    
    # If we can construct such a set A in [1, 256], then Diff(A) is huge (180 values).
    # We need Diff(B) to be disjoint from Diff(A).
    # The available differences in [1, 256] are in {-255, ..., 255} \ {0}.
    # Total 510 values.
    # Diff(A) takes 180 values.
    # Remaining 330 values for B.
    # B must be a Sidon set. If |B|=k_b, it needs k_b(k_b-1) differences.
    # If k_b=18, 18*17 = 306. This fits in 330.
    # So |A|=10, |B|=18 is possible? Total 28.
    # Can we do better?
    # If we use a smaller A, say |A|=6 (q=5, v=31). Diff(A) size 30.
    # Remaining 480.
    # If |B|=21, 21*20=420. Fits. Total 27.
    # If |A|=5 (q=4, v=21). Diff(A) size 20.
    # Remaining 490.
    # If |B|=22, 22*21=462. Fits. Total 27.
    
    # What if A is not a Singer set?
    # A generic Sidon set of size k has k(k-1) differences.
    # The "spread" of differences matters.
    # Singer sets have differences packed in [-v, v].
    # A "sparse" Sidon set might have differences spread out more, leaving gaps for B.
    # But B also needs to be Sidon.
    
    # Let's try to construct A as a Singer set for q=9 (v=91, k=10).
    # We need the actual elements.
    # Construction of Singer Difference Set:
    # Let F be GF(q). The Singer cycle is generated by a matrix M in GL(2, q).
    # Alternatively, use the standard construction:
    # D = { x in F_{q^2} | Tr(x) = 0 }? No.
    # D = { g^i | i = 0, ..., k-1 } where g is a generator of the subgroup of order k in F_{q^2}^*?
    # No, the Singer difference set is a subset of Z_v.
    # Standard construction:
    # Let q be a prime power. Let v = q^2 + q + 1.
    # Let F = GF(q).
    # The set D = { x in F^2 | x is a vector such that ... }
    # Actually, easier:
    # D = { (x, 1) | x in F } U { (0, 0) }? No.
    # D corresponds to the projective line PG(1, q).
    # Points in PG(1, q) are 1-dim subspaces of F^2.
    # There are q+1 points.
    # We can represent them as slopes.
    # Let's just use a known list for q=9?
    # Or generate it.
    
    # Let's generate the Singer difference set for q=9.
    # We need a primitive element in GF(9).
    # GF(9) = F_3[x] / (x^2+1) = F_3[x] / (x^2-2) ? x^2+1 = x^2-2.
    # Let alpha be root of x^2+1=0 in F_3. alpha^2 = -1 = 2.
    # Elements: 0, 1, 2, alpha, 1+alpha, 2+alpha, 2alpha, 1+2alpha, 2+2alpha.
    # Let's find a primitive element.
    # Order of alpha: alpha^2=2, alpha^4=4=1. Order 4.
    # Order of 1+alpha: (1+alpha)^2 = 1 + 2alpha + alpha^2 = 1+2alpha+2 = 3+2alpha = 2alpha.
    # (1+alpha)^4 = (2alpha)^2 = 4 alpha^2 = 1*2 = 2.
    # (1+alpha)^8 = 4 = 1. Order 8.
    # So 1+alpha is primitive.
    
    # The Singer difference set in Z_{q^2+q+1} is constructed by mapping the projective line points to integers.
    # The points are P_x = (x, 1) for x in F, and P_inf = (1, 0).
    # We need an ordering.
    # The standard construction uses the action of the Singer cycle.
    # Let's just use a simple Sidon set for A and hope B can be large.
    # Or better, use the "greedy" method but with a smarter order for A to make it "compact" (small differences).
    # If A is compact, its differences are small.
    # Then B can use large differences.
    # But B must be Sidon.
    # If B uses large differences, B must be "spread out".
    # E.g., B = {1, 100, 200, ...}
    # Differences: 99, 199, 100, 200, 101...
    # If A is {1, 2, 4, 8, 16, 32} (Powers of 2, Sidon? No, 1+8=9, 2+7 no. 1+4=5, 2+3 no. 1+2=3, 4+? no.
    # Powers of 2: sums are unique? 2^i + 2^j. Yes, binary representation.
    # So A = {1, 2, 4, 8, 16, 32, 64, 128} is Sidon.
    # Differences: all differences between powers of 2.
    # 2^j - 2^i.
    # These are sparse.
    # Let's check if A={1,2,4,8,16,32,64,128} is valid in [1, 256].
    # Max sum 128+128=256. OK.
    # Diff(A) = { 2^j - 2^i | i != j }.
    # Size 8*7 = 56.
    # Values are like 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, ...
    # They are all distinct?
    # 2^j - 2^i = 2^i (2^{j-i} - 1).
    # Unique factorization ensures uniqueness.
    # So Diff(A) has 56 distinct values.
    # Remaining differences: 510 - 56 = 454.
    # We need B Sidon.
    # If |B|=20, need 380 differences.
    # Can we find a Sidon set B of size 20 with differences disjoint from Diff(A)?
    # Diff(A) contains many small numbers.
    # If we choose B to be "large" numbers, e.g., {130, 131, ...}?
    # Differences will be small. Collision.
    # If we choose B to be spread out, e.g., {1, 20, 40, 60, ...}?
    # Differences are multiples of 20.
    # Does Diff(A) contain multiples of 20?
    # 2^j - 2^i = 20?
    # 2^i (2^k - 1) = 20 = 4*5.
    # 2^i must divide 4. i=0,1,2.
    # i=2: 4(2^k-1)=20 => 2^k-1=5 => 2^k=6 (No).
    # i=1: 2(2^k-1)=20 => 2^k-1=10 => 2^k=11 (No).
    # i=0: 1(2^k-1)=20 => 2^k=21 (No).
    # So 20 is not in Diff(A).
    # What about 40?
    # 2^i (2^k-1) = 40 = 8*5.
    # i=3: 8(2^k-1)=40 => 2^k-1=5 => 2^k=6 (No).
    # i=2: 4(2^k-1)=40 => 2^k-1=10 => 2^k=11 (No).
    # So 40 is not in Diff(A).
    # It seems Diff(A) (powers of 2) avoids many multiples of 5?
    # Actually, 2^j - 2^i is never divisible by 5 if 2^j != 2^i mod 5?
    # 2^k mod 5 cycles 2, 4, 3, 1.
    # 2^j - 2^i = 0 mod 5 => 2^{j-i} = 1 mod 5 => j-i is multiple of 4.
    # If j-i=4, 2^4-1=15. 15 is not divisible by 5? 15 is divisible by 5.
    # So 15 is in Diff(A).
    # 15 = 16-1.
    # So multiples of 5 might be in Diff(A).
    
    # Let's try a different A.
    # A = Singer set for q=9.
    # If we can generate it, it's compact.
    # Let's try to generate Singer set for q=9 in code.
    
    def generate_singer(q):
        # q must be prime power
        # v = q*q + q + 1
        # We need a primitive element in GF(q)
        # For q=9, GF(9) = F_3[x]/(x^2+1)
        # Primitive element: 1+alpha (alpha^2=2)
        # The Singer difference set D is a subset of Z_v.
        # Construction:
        # Let F = GF(q).
        # The points of PG(1, q) are P_x = (x, 1) for x in F, and P_inf = (1, 0).
        # We need to map these to integers 0..v-1.
        # The standard mapping uses the action of the Singer cycle.
        # Let's use a simpler method:
        # D = { x in Z_v | x is of the form ... }
        # Actually, let's just use a known Sidon set for A and greedy for B.
        # The power of 2 set for A is very sparse in differences.
        # Let's use A = {1, 2, 4, 8, 16, 32, 64, 128}.
        # And try to construct B greedily from large numbers, avoiding Diff(A).
        
        A = [1, 2, 4, 8, 16, 32, 64, 128]
        # Verify A is Sidon
        # Sums: 2, 3, 5, 9, 17, 33, 65, 129, ...
        # All unique? Yes, binary.
        
        diff_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diff_A.add(A[i] - A[j])
        
        # Construct B
        B = []
        # Try to add elements from 256 downwards?
        # Or random?
        # If we pick large elements, differences are large.
        # Diff(A) max is 127.
        # So if B elements are in [128, 256], differences are in [-128, 128].
        # Diff(A) is in [-127, 127].
        # Overlap is likely.
        
        # Let's try to pick B elements such that differences avoid Diff(A).
        # Greedy:
        candidates = list(range(256, 0, -1))
        for x in candidates:
            if len(B) >= 20: break
            # Check Sidon condition for B + [x]
            test_B = B + [x]
            sums = [a+b for a in test_B for b in test_B]
            if len(sums) != len(set(sums)):
                continue
            
            # Check difference condition
            new_diffs = set()
            for i in range(len(test_B)):
                for j in range(len(test_B)):
                    if i != j:
                        new_diffs.add(test_B[i] - test_B[j])
            
            if new_diffs & diff_A:
                continue
            
            # Also check if new_diffs has internal collisions? No, Sidon check covers sums, which implies differences are distinct?
            # No, Sidon means sums are unique. This implies differences are unique?
            # a+b = c+d => a-c = d-b.
            # If sums are unique, then a+b != c+d for distinct pairs.
            # This does NOT imply differences are unique.
            # E.g., {1, 2, 4}. Sums: 2,3,5,4,6,8. Unique.
            # Diffs: 1, -1, 3, -3, 2, -2. All unique.
            # Wait, is it true that Sidon => differences unique?
            # No. Sidon (B_2) means a+b=c+d implies {a,b}={c,d}.
            # Differences: a-b = c-d => a+d = b+c.
            # So if differences are not unique, there is a "cross sum" collision.
            # a-b = c-d => a+d = b+c.
            # This is a sum collision.
            # So Sidon implies differences are unique (up to sign? No, a-b and c-d are distinct if a+d != b+c).
            # Actually, if a-b = c-d, then a+d = b+c.
            # If {a,d} != {b,c}, it's a violation.
            # If {a,d} = {b,c}, then a=b and d=c (so a-b=c-d) or a=c and d=b (so a-b = c-d => a-b = a-b).
            # So Sidon sets have distinct differences?
            # Let's check {1, 2, 4, 8}.
            # Diffs: 1, 3, 7, -1, -3, -7, 2, 6, -2, -6, 4, -4.
            # All distinct?
            # 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 7, -7.
            # Yes.
            # So Sidon implies all nonzero differences are distinct.
            
            B.append(x)
            diff_A = diff_A.union(new_diffs) # Wait, we need to keep diff_A fixed and check B against it.
            # But B's differences must also be disjoint from A's.
            # And B's differences must be disjoint from B's previous differences? No, Sidon check handles B's internal consistency.
            # We just need B's diffs to be disjoint from A's diffs.
            
        if not B:
            B = [256]
            
        return [A, B]

    # The above is too simple. Let's run a more robust search.
    # Use the power of 2 set for A.
    # Use randomized greedy for B.
    
    A_base = [1, 2, 4, 8, 16, 32, 64, 128]
    diff_A = set()
    for i in range(len(A_base)):
        for j in range(len(A_base)):
            if i != j:
                diff_A.add(A_base[i] - A_base[j])
    
    best_B = []
    
    import random
    rng = random.Random(42)
    
    for _ in range(500):
        if time.time() > deadline: break
        B = []
        # Random order of candidates
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        # Bias towards large numbers?
        # If we pick large numbers, differences are large.
        # Diff(A) is small (max 127).
        # So large differences in B are safe.
        # To get large differences, pick numbers far apart.
        # E.g., 1, 100, 200.
        # Let's try to construct B with large gaps.
        
        # Strategy: Pick first element randomly.
        # Then pick next element far from all existing B elements.
        # And check Sidon and Diff constraints.
        
        if candidates:
            x0 = candidates.pop()
            B.append(x0)
            
        for x in candidates:
            if len(B) >= 20: break
            # Check Sidon
            test_B = B + [x]
            sums = [a+b for a in test_B for b in test_B]
            if len(sums) != len(set(sums)):
                continue
            
            # Check Diffs
            new_diffs = set()
            for i in range(len(test_B)):
                for j in range(len(test_B)):
                    if i != j:
                        d = test_B[i] - test_B[j]
                        new_diffs.add(d)
            
            if new_diffs & diff_A:
                continue
            
            # Also check if new_diffs has collisions with existing B diffs?
            # Sidon check ensures B is Sidon, so diffs are distinct.
            # But we need to ensure new_diffs doesn't collide with OLD_B_diffs?
            # No, Sidon check on test_B covers all pairs in test_B.
            # So if test_B is Sidon, all its diffs are distinct.
            # We just need them to be disjoint from diff_A.
            
            B.append(x)
        
        if len(B) > len(best_B):
            best_B = B[:]
    
    if not best_B:
        best_B = [256]
        
    # Verify
    def _valid(A, B):
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if A.size == 0 or B.size == 0:
            return False
        if A.max() > 256 or B.max() > 256 or A.min() < 1 or B.min() < 1:
            return False
        sa = A[:, None] + A[None, :]
        if sa.size != len(np.unique(sa)):
            return False
        sb = B[:, None] + B[None, :]
        if sb.size != len(np.unique(sb)):
            return False
        da = np.unique(A[:, None] - A[None, :])
        db = np.unique(B[:, None] - B[None, :])
        if 0 in da:
            da = da[da != 0]
        if 0 in db:
            db = db[db != 0]
        if len(np.intersect1d(da, db, assume_unique=False)) > 0:
            return False
        return True

    if not _valid(A_base, best_B):
        # Fallback
        A_base = [1]
        best_B = [2]
        
    return [A_base, best_B]
