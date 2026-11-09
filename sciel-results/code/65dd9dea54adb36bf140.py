import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit
    
    def compute_scores(A):
        """Compute |A+A| and |A*A|"""
        A = sorted(A)
        sum_set = set()
        prod_set = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))
    
    def is_valid(A):
        """Check if A is valid: 20 distinct positive integers <= 10000"""
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x <= 0 or x > 10000 for x in A):
            return False
        return True
    
    best_score = float('inf')
    best_set = set(range(1, 21))
    
    # Strategy: union of arithmetic progression (AP) and geometric progression (GP)
    # AP: {a + i*d for i in range(n_ap)}
    # GP: {b * r^i for i in range(n_gp)}
    # Total: n_ap + n_gp = 20 (with possible overlap)
    
    # First, try pure AP and pure GP
    # Pure AP: {a, a+d, ..., a+19d}
    # A+A = {2a, 2a+d, ..., 2a+19d} -> 20 elements
    # A*A = {a², a(a+d), ..., (a+19d)²} -> up to 210 elements
    # Need to minimize A*A
    
    # Try small a and d to get small products
    for d in range(1, 20):
        for a in range(1, 50):
            A = [a + i*d for i in range(20)]
            if A[-1] > 10000:
                break
            if is_valid(A):
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best_set = set(A)
            if time.time() > deadline:
                return sorted(best_set)
    
    # Try pure GP: {b, b*r, b*r², ..., b*r¹⁹} with integer r
    # A+A = {b(1+r^i + r^j)} -> many distinct
    # A*A = {b²*r^(i+j)} -> 20 elements (i+j from 0 to 38)
    # Need to minimize A+A
    
    for r in range(2, 10):
        for b in range(1, 100):
            A = [b * (r ** i) for i in range(20)]
            if A[-1] > 10000:
                break
            if is_valid(A):
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best_set = set(A)
            if time.time() > deadline:
                return sorted(best_set)
    
    # Try union of AP and GP
    # AP of length n_ap, GP of length n_gp, n_ap + n_gp = 20 (allowing overlap)
    for n_ap in range(10, 16):  # AP length
        n_gp = 20 - n_ap  # GP length (some may overlap)
        
        for d in range(1, 15):
            for a in range(1, 30):
                ap_set = set(a + i*d for i in range(n_ap))
                if max(ap_set) > 10000:
                    break
                for r in range(2, 8):
                    for b in range(1, 50):
                        gp_set = set()
                        val = b
                        for i in range(n_gp):
                            gp_set.add(val)
                            val *= r
                        if max(gp_set) > 10000:
                            break
                        combined = ap_set | gp_set
                        if len(combined) != 20:
                            continue
                        if is_valid(list(combined)):
                            score = compute_scores(list(combined))
                            if score < best_score:
                                best_score = score
                                best_set = combined
                        if time.time() > deadline:
                            return sorted(best_set)
    
    # Local search: try to improve by replacing elements
    A = sorted(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_score = best_score
        for i in range(20):
            for new_val in range(1, 10001):
                if new_val in A:
                    continue
                A_new = A[:i] + [new_val] + A[i+1:]
                if len(set(A_new)) != 20:
                    continue
                score = compute_scores(A_new)
                if score < current_score:
                    A = A_new
                    best_set = set(A)
                    best_score = score
                    current_score = score
                    improved = True
                    break
            if time.time() > deadline:
                return sorted(best_set)
    
    return sorted(best_set)
