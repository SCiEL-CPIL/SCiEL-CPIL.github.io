import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    def try_add(A, sums_A, diff_A, diff_B, x):
        """
        Check if x can be added to A given current A, sums_A, diff_A and diff_B.
        Returns (valid, new_sums, new_diffs).
        """
        # 1. Check Sidon property for A
        # New sums: 2x, x+a for a in A
        if 2 * x in sums_A:
            return False, None, None
        for a in A:
            if x + a in sums_A:
                return False, None, None
        
        # 2. Check Disjoint Differences
        # New diffs: x-a, a-x for a in A
        for a in A:
            d1 = x - a
            d2 = a - x
            if d1 in diff_B:
                return False, None, None
            if d2 in diff_B:
                return False, None, None
            
        # If valid, construct new sets
        # Using set union is efficient for small sets
        new_sums = sums_A | {2 * x}
        new_diffs = diff_A.copy()
        for a in A:
            new_sums.add(x + a)
            new_diffs.add(x - a)
            new_diffs.add(a - x)
            
        return True, new_sums, new_diffs

    def run_greedy(seed):
        random.seed(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diff_A = set()
        diff_B = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            valid_A, new_sums_A, new_diff_A = try_add(A, sums_A, diff_A, diff_B, x)
            # Try adding to B
            valid_B, new_sums_B, new_diff_B = try_add(B, sums_B, diff_B, diff_A, x)
            
            if valid_A and valid_B:
                # Randomly choose
                if random.random() < 0.5:
                    A.append(x)
                    sums_A = new_sums_A
                    diff_A = new_diff_A
                else:
                    B.append(x)
                    sums_B = new_sums_B
                    diff_B = new_diff_B
            elif valid_A:
                A.append(x)
                sums_A = new_sums_A
                diff_A = new_diff_A
            elif valid_B:
                B.append(x)
                sums_B = new_sums_B
                diff_B = new_diff_B
            else:
                pass
                
        return A, B

    seed = 0
    while time.time() < deadline:
        A, B = run_greedy(seed)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        seed += 1
        
    # Ensure non-empty if possible? 
    # If best_size == 0, we might have failed to find any valid pair.
    # But with n=256, it's very likely to find something.
    # The problem states "two nonempty Sidon subsets".
    # If we return empty, it might be invalid.
    # Let's add a fallback if best_size is 0.
    if best_size == 0:
         # Try a deterministic simple case
         A = [1, 2]
         B = [4, 8]
         # Check if valid
         # A: sums 2,3,4. Diffs 1, -1.
         # B: sums 8,12,16. Diffs 4, -4.
         # Disjoint diffs? {1,-1} and {4,-4}. Yes.
         # Sidon? A: 1+1=2, 1+2=3, 2+2=4. Unique. B: 4+4=8, 4+8=12, 8+8=16. Unique.
         # So [1,2] and [4,8] is a valid solution of size 4.
         return [A, B]

    return [best_A, best_B]
