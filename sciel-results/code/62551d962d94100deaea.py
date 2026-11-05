import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    best_A = []
    best_B = []
    best_size = 0
    
    def check_sidon_and_diffs(S, x, used_diffs_S, other_diffs):
        """
        Check if x can be added to S.
        S is a list of ints.
        used_diffs_S is a set of differences in S.
        other_diffs is a set of differences in the other set.
        Returns True if valid, else False.
        """
        # 1. Check Sidon property for S U {x}
        # New sums: x + s for s in S, and x + x
        # Existing sums are unique.
        # We need to ensure x+s is not equal to any existing sum, and x+s1 != x+s2 (which is true if s1!=s2)
        # Also x+x must not be equal to any existing sum or x+s.
        
        # Efficient check:
        # A set S is Sidon if all pair sums are unique.
        # Adding x:
        # New pair sums involving x: x+x, x+s for s in S.
        # We need these to be distinct from each other and from existing sums.
        # Distinct from each other: x+x is unique. x+s1 == x+s2 implies s1==s2. So new sums are distinct among themselves.
        # Distinct from existing:
        # We need to know the set of existing sums. 
        # Since S is Sidon, we can compute sums on the fly or maintain them.
        # For small S, computing sums on the fly is fine.
        
        sums_S = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums_S.add(S[i] + S[j])
                
        # New sums to check
        new_sums = [x + x] + [x + s for s in S]
        
        for ns in new_sums:
            if ns in sums_S:
                return False
        
        # 2. Check Difference Disjointness
        # New differences introduced by adding x to S:
        # d = x - s for s in S
        # d = s - x for s in S
        # These must not be in other_diffs.
        # Also, since S is Sidon, these new differences are distinct from existing differences in S 
        # (except 0 which is ignored). 
        # Wait, do we need to check if new diffs conflict with existing diffs in S?
        # If x - s1 = s2 - s3, then x + s3 = s2 + s1. This is a sum collision in S U {x}.
        # So if the Sidon check passes, the new differences are distinct from old differences in S.
        # We only need to check against other_diffs.
        
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 in other_diffs:
                return False
            if d2 in other_diffs:
                return False
                
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                diffs.add(S[i] - S[j])
                diffs.add(S[j] - S[i])
        return diffs

    # Main search loop
    while time.time() - start_time < timeout:
        # Randomize order
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        # Optional: Start with a random small seed to break symmetry?
        # Or just greedy from empty.
        
        for x in candidates:
            # Try adding to A
            can_add_A = check_sidon_and_diffs(A, x, diffs_A, diffs_B)
            # Try adding to B
            can_add_B = check_sidon_and_diffs(B, x, diffs_B, diffs_A)
            
            if can_add_A and can_add_B:
                # Heuristic: Add to the smaller set to balance? 
                # Or random?
                # Balancing might be better to keep difference sets sparse.
                if len(A) <= len(B):
                    A.append(x)
                    # Update diffs_A
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                else:
                    B.append(x)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            elif can_add_A:
                A.append(x)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif can_add_B:
                B.append(x)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
            else:
                pass # Cannot add to either
            
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
