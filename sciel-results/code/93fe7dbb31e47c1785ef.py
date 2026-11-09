"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 256
    max_size = 64
    
    def is_sidon(s):
        """Check if s is a Sidon set (all pair sums including a+a are unique)."""
        if len(s) <= 1:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diffs(s):
        """Get set of nonzero differences from set s."""
        d = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d.add(s[j] - s[i])
        return d
    
    def try_add_greedy(A, B, x):
        """Try to add x to A or B, return which one it fits in or None."""
        # Try adding to A
        A_test = A + [x]
        if is_sidon(A_test):
            dA = get_diffs(A_test)
            dB = get_diffs(B)
            if not dA & dB:
                return 0
        # Try adding to B
        B_test = B + [x]
        if is_sidon(B_test):
            dA = get_diffs(A)
            dB = get_diffs(B_test)
            if not dA & dB:
                return 1
        return None
    
    # Greedy construction with multiple random orderings
    best_A, best_B = [], []
    best_score = 0
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random permutation of [1, 256]
        order = list(range(1, n+1))
        random.shuffle(order)
        
        A, B = [], []
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            side = try_add_greedy(A, B, x)
            if side == 0:
                A.append(x)
            elif side == 1:
                B.append(x)
        
        # Local search: try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                for j in range(len(B)):
                    if time.time() > deadline:
                        break
                    # Try swapping A[i] and B[j]
                    A_new = A[:i] + A[i+1:] + [B[j]]
                    B_new = B[:j] + B[j+1:] + [A[i]]
                    if is_sidon(A_new) and is_sidon(B_new):
                        dA = get_diffs(A_new)
                        dB = get_diffs(B_new)
                        if not dA & dB:
                            A, B = A_new, B_new
                            improved = True
                            break
                if improved:
                    break
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
