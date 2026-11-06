# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A| + |B|.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds safety margin within 90s limit
    
    n = 256
    
    def check_sidon(S):
        """Check if S is a Sidon set (pair sums with replacement are unique)."""
        seen = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    def get_diff_set(S):
        """Return the set of nonzero absolute differences for S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def try_add(S, x, used_diffs):
        """
        Try to add x to Sidon set S.
        Returns (new_S, new_diffs) if successful, else (None, None).
        """
        # Check Sidon condition for S + [x]
        seen_sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in seen_sums:
                    return None, None
                seen_sums.add(s)
        
        # Add sums involving x
        for i in range(len(S)):
            s = S[i] + x
            if s in seen_sums:
                return None, None
            seen_sums.add(s)
        # x + x
        if x + x in seen_sums:
            return None, None
        
        # Check difference disjointness
        new_diffs = set()
        for i in range(len(S)):
            d = abs(S[i] - x)
            if d > 0:
                new_diffs.add(d)
        
        if new_diffs & used_diffs:
            return None, None
        
        return S + [x], new_diffs
    
    def greedy_construct(order, max_time):
        """
        Greedily construct two Sidon sets with disjoint difference sets.
        order: list of candidate numbers to consider.
        Returns (A, B).
        """
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > max_time:
                break
            
            # Try to add to A
            new_A, new_diffs = try_add(A, x, diffs_B)
            if new_A is not None:
                A = new_A
                diffs_A |= new_diffs
                continue
            
            # Try to add to B
            new_B, new_diffs = try_add(B, x, diffs_A)
            if new_B is not None:
                B = new_B
                diffs_B |= new_diffs
        
        return A, B
    
    # Strategy: Use different random orderings and pick the best result
    best_total = 0
    best_A = []
    best_B = []
    
    # First, try a deterministic greedy with natural order
    order = list(range(1, 257))
    A, B = greedy_construct(order, deadline)
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_A = A[:]
        best_B = B[:]
    
    # Then, try random permutations
    random.seed(42)
    num_tries = 50
    
    for trial in range(num_tries):
        if time.time() > deadline:
            break
        
        order = list(range(1, 257))
        random.shuffle(order)
        
        A, B = greedy_construct(order, deadline)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
