"""B_3[1] construction via base-4 digit separation + greedy augmentation."""
import time
import numpy as np

def construct():
    deadline = time.time() + 85.0
    N = 4096
    
    # Phase 1: Build guaranteed-valid skeleton using base-4 digits.
    # Numbers of form sum(d_i * 4^i) where d_i in {0,1,2} give unique triple sums
    # because in base 4, each digit of a triple sum is determined uniquely.
    # With k digits, we get 3^k numbers. 3^5=243 > 4096? No, 4^5=1024, 3^5=243.
    # Max value with k digits: sum(2*4^i for i=0..k-1) = 2*(4^k-1)/3.
    # For k=5: 2*(1024-1)/3 = 682 < 4096. Good, 243 numbers.
    # For k=6: 2*(4096-1)/3 = 2730 < 4096. 3^6=729 numbers.
    # For k=7: 2*(16384-1)/3 = 10922 > 4096. Too big.
    # So k=6 gives 729 numbers all <= 4096.
    
    # Generate all 3^6 combinations of digits {0,1,2}^6
    skeleton = []
    for d0 in range(3):
        for d1 in range(3):
            for d2 in range(3):
                for d3 in range(3):
                    for d4 in range(3):
                        for d5 in range(3):
                            val = d0 + d1*4 + d2*16 + d3*64 + d4*256 + d5*1024
                            if 1 <= val <= 4096:
                                skeleton.append(val)
    
    # Remove 0 if present (we need positive integers)
    skeleton = [x for x in skeleton if x >= 1]
    skeleton.sort()
    
    selected = list(skeleton)
    selected_set = set(selected)
    
    # Build existing sums for the skeleton
    n = len(selected)
    sel = np.array(selected, dtype=np.int64)
    
    # All unordered triples with repetition: C(n+2,3) = (n)(n+1)(n+2)/6
    # For n=728, that's ~63 million sums. Too many to store in a set efficiently.
    # Instead, we'll use a different approach: for greedy augmentation, we only
    # need to check new candidate against existing sums.
    # 
    # Alternative: Use the structure. Since skeleton is base-4 separated,
    # we can check if a new number x would collide by checking its "digit pattern".
    # Actually, let's just do greedy but with a time limit and accept what we get.
    
    # For efficiency with large n, use numpy-based collision check
    existing_sums = set()
    
    # Build existing sums incrementally - this is expensive for n=728
    # C(728,3) with repetition = 728*729*730/6 = 63,170,440 sums
    # That's a lot. Let's use a smarter approach.
    
    # Actually, let's reconsider. The task says max 80 integers.
    # So we only need up to 80. Let's use a smaller skeleton.
    
    # With k=4 digits: 3^4 = 81 numbers, max value = 2*(64-1)/3 = 42
    # With k=5: 3^5 = 243, max = 2*(256-1)/3 = 170
    # 
    # Since we only need up to 80, let's use k=4 (81 numbers, all <= 42)
    # Then greedily add more numbers up to 4096.
    
    # Actually wait - we want to MAXIMIZE cardinality. So we want as many as possible.
    # The base-4 construction gives us a guaranteed valid set.
    # Let's use the largest base-4 set that fits, then try to add more.
    
    # Reset: use k=5 for 243 numbers (all <= 170), well within 4096
    # But we can only return up to 80. So let's just take 80 from the base-4 set.
    # Actually no - the task says maximize cardinality, up to 80.
    # So if our base-4 set has 80+ valid numbers, we're done with 80.
    
    # Let me re-read: "1..80 distinct positive integers <=n=4096"
    # So maximum is 80. If we can get 80, that's optimal.
    
    # Base-4 with k=4: 81 numbers in [1,42]. Take 80 of them.
    # But are all 81 valid? Yes, by construction.
    # So we can return any 80 of them.
    
    # Actually, let's be more careful. The base-4 construction:
    # Each number has digits d_i in {0,1,2} for i=0,1,2,3 (4 digits).
    # A triple sum x+y+z has digit i = d_i(x)+d_i(y)+d_i(z) which is in {0,1,...,6}.
    # Since each digit is at most 6 < 8 = 2*4, there's no carry between digits.
    # Wait, that's not quite right. The issue is that different triples could
    # produce the same digit sequence.
    #
    # Actually the key property: if x,y,z each have digits in {0,1,2},
    # then x+y+z has digits in {0,...,6} with NO carry (since 6 < 4? No, 6 >= 4).
    # So there IS carry. This doesn't work directly.
    
    # Correct approach for B_3[1]: use digits in {0,1} (binary-like) with base 4.
    # If each number has digits in {0,1} in base 4, then triple sum has digits in {0,1,2,3}.
    # Max digit is 3 < 4, so no carry. And the triple sum uniquely determines
    # the multiset of three numbers' digit vectors? Not necessarily.
    
    # Let me think differently. The standard construction for B_3[1]:
    # Use numbers of the form sum(a_i * b^i) where a_i in {0, 1, ..., k}
    # and b > k*(number of terms - 1) to avoid carries.
    # For triples: b > 3*2 = 6, so b >= 7.
    # With b=7 and digits in {0,1,2}: max value with k digits = 2*(7^k-1)/6.
    # k=4: 2*(2401-1)/6 = 800. 3^4=81 numbers. All <= 800 < 4096. 
    # k=5: 2*(16807-1)/6 = 5602 > 4096. Too big.
    
    # So with base 7, digits {0,1,2}, k=4: 81 numbers, all <= 800.
    # Triple sums: each digit of sum is in {0,...,6} < 7, no carry.
    # Uniqueness: the triple sum's base-7 digits determine the sum of digits
    # at each position. But different multisets of three digit-vectors can
    # give the same per-position sums.
    #
    # Hmm, this is the fundamental issue. B_3[1] requires ALL unordered triples
    # (with repetition) to have distinct sums. This is a very strong condition.
    #
    # Let me reconsider. The greedy approach in the original code is actually
    # the right idea, but it's slow because it scans from 1 upward and does
    # expensive checks. Let me optimize the greedy approach.
    
    # New plan: Greedy from 1 to 4096, but with fast rejection.
    # Key insight: for a candidate x, the new sums are:
    #   3x, 2x+y, x+2y, x+y+z for all y<=z in selected
    # We can check collisions using a set.
    
    # The bottleneck is building the set of existing sums.
    # For n selected numbers, total sums = n(n+1)(n+2)/6.
    # For n=80: 80*81*82/6 = 88,560 sums. Very manageable.
    
    # So the greedy should work fine for up to 80 numbers.
    # The issue with the original code: it was scanning ALL of 1..4096
    # and doing numpy operations each time. Let's optimize.
    
    selected = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(x)
            existing_sums.add(3 * x)
            continue
        
        n = len(selected)
        sel = np.array(selected, dtype=np.int64)
        
        # New sums involving x:
        # 3x
        # 2x + y for each y
        # x + 2y for each y  
        # x + y + z for each pair y <= z
        
        # Build as numpy array
        parts = [np.array([3 * x], dtype=np.int64)]
        parts.append(2 * x + sel)
        parts.append(x + 2 * sel)
        
        ii, jj = np.triu_indices(n)
        parts.append(x + sel[ii] + sel[jj])
        
        new_sums_arr = np.concatenate(parts)
        
        # Internal uniqueness check
        num_expected = 1 + 2 * n + n * (n + 1) // 2
        if len(np.unique(new_sums_arr)) != num_expected:
            continue
        
        # Check against existing
        new_sums_list = new_sums_arr.tolist()
        collision = False
        for s in new_sums_list:
            if s in existing_sums:
                collision = True
                break
        
        if collision:
            continue
        
        # Valid
        selected.append(x)
        existing_sums.update(new_sums_list)
        
        if len(selected) >= 80:
            break
    
    return selected
