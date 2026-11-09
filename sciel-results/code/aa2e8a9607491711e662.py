import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = 256
    max_size = 64
    
    def check_add(s, x, sum_set, diff_set, other_diff_set):
        """Check if x can be added to set s. Returns (valid, new_sum_set, new_diff_set) or (False, None, None)."""
        # Check pair sums: x + s_i for each s_i in s, and x + x
        for si in s:
            if x + si in sum_set:
                return False, None, None
        if x + x in sum_set:
            return False, None, None
        # Check differences against other set
        for si in s:
            d1 = x - si
            d2 = si - x
            if d1 in other_diff_set:
                return False, None, None
            if d2 in other_diff_set:
                return False, None, None
        return True, None, None
    
    def add_element(s, x, sum_set, diff_set):
        """Add x to set s, update sum_set and diff_set in place."""
        for si in s:
            sum_set.add(x + si)
            diff_set.add(x - si)
            diff_set.add(si - x)
        sum_set.add(x + x)
        s.append(x)
    
    def greedy_build(order_a, order_b, strategy="interleave"):
        """Build two Sidon sets with disjoint diffs using given element orders."""
        A = []
        B = []
        sum_a = set()
        sum_b = set()
        diff_a = set()
        diff_b = set()
        
        if strategy == "interleave":
            ia, ib = 0, 0
            while (ia < len(order_a) or ib < len(order_b)) and time.time() < deadline:
                if ia < len(order_a) and len(A) < max_size:
                    x = order_a[ia]
                    ok, _, _ = check_add(A, x, sum_a, diff_a, diff_b)
                    if ok:
                        add_element(A, x, sum_a, diff_a)
                    ia += 1
                if ib < len(order_b) and len(B) < max_size:
                    x = order_b[ib]
                    ok, _, _ = check_add(B, x, sum_b, diff_b, diff_a)
                    if ok:
                        add_element(B, x, sum_b, diff_b)
                    ib += 1
        elif strategy == "a_first":
            for x in order_a:
                if len(A) >= max_size:
                    break
                if time.time() >= deadline:
                    break
                ok, _, _ = check_add(A, x, sum_a, diff_a, diff_b)
                if ok:
                    add_element(A, x, sum_a, diff_a)
            for x in order_b:
                if len(B) >= max_size:
                    break
                if time.time() >= deadline:
                    break
                ok, _, _ = check_add(B, x, sum_b, diff_b, diff_a)
                if ok:
                    add_element(B, x, sum_b, diff_b)
        elif strategy == "b_first":
            for x in order_b:
                if len(B) >= max_size:
                    break
                if time.time() >= deadline:
                    break
                ok, _, _ = check_add(B, x, sum_b, diff_b, diff_a)
                if ok:
                    add_element(B, x, sum_b, diff_b)
            for x in order_a:
                if len(A) >= max_size:
                    break
                if time.time() >= deadline:
                    break
                ok, _, _ = check_add(A, x, sum_a, diff_a, diff_b)
                if ok:
                    add_element(A, x, sum_a, diff_a)
        return A, B
    
    # Try multiple strategies and restarts
    best_total = 0
    best_result = ([1], [2])  # fallback
    
    all_elements = list(range(1, n + 1))
    
    # Strategy 1: Ascending order, interleave
    A, B = greedy_build(all_elements, all_elements, "interleave")
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A, B)
    
    # Strategy 2: Descending order, interleave
    desc = list(range(n, 0, -1))
    A, B = greedy_build(desc, desc, "interleave")
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A, B)
    
    # Strategy 3: Ascending, a_first
    A, B = greedy_build(all_elements, all_elements, "a_first")
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A, B)
    
    # Strategy 4: Descending, a_first
    A, B = greedy_build(desc, desc, "a_first")
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A, B)
    
    # Random restarts with different orderings
    random.seed(42)
    for restart in range(200):
        if time.time() >= deadline - 5.0:
            break
        
        # Random permutation
        order_a = all_elements[:]
        random.shuffle(order_a)
        order_b = all_elements[:]
        random.shuffle(order_b)
        
        strategy = random.choice(["interleave", "a_first", "b_first"])
        A, B = greedy_build(order_a, order_b, strategy)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A, B)
    
    # Try to improve best result by adding more elements
    if time.time() < deadline - 2.0:
        A, B = best_result
        sum_a = set()
        sum_b = set()
        diff_a = set()
        diff_b = set()
        for a in A:
            for b in A:
                if a <= b:
                    sum_a.add(a + b)
        for a in A:
            for b in A:
                if a != b:
                    diff_a.add(a - b)
        for a in B:
            for b in B:
                if a <= b:
                    sum_b.add(a + b)
        for a in B:
            for b in B:
                if a != b:
                    diff_b.add(a - b)
        
        # Try adding more to A
        for x in range(1, n + 1):
            if time.time() >= deadline - 1.0:
                break
            if x in A:
                continue
            if len(A) >= max_size:
                break
            ok, _, _ = check_add(A, x, sum_a, diff_a, diff_b)
            if ok:
                add_element(A, x, sum_a, diff_a)
        
        # Try adding more to B
        for x in range(1, n + 1):
            if time.time() >= deadline - 1.0:
                break
            if x in B:
                continue
            if len(B) >= max_size:
                break
            ok, _, _ = check_add(B, x, sum_b, diff_b, diff_a)
            if ok:
                add_element(B, x, sum_b, diff_b)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A, B)
    
    return list(best_result[0]), list(best_result[1])
