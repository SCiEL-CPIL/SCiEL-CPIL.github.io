# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds, safety margin for 90s limit

def _pair_sums_ok(lst):
    """Check if all pair sums (a+b for a<=b) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a!=b."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - a)
    return d

def _can_add(lst, x, other_diffs, used_sums):
    """Check if x can be added to lst maintaining Sidon and no diff conflict with other_diffs."""
    n = len(lst)
    # Check pair sums: x+x and x+a for a in lst
    if x + x in used_sums:
        return False
    for a in lst:
        if x + a in used_sums:
            return False
    # Check differences: x-a and a-x for a in lst must not be in other_diffs
    for a in lst:
        d1 = x - a
        d2 = a - x
        if d1 in other_diffs or d2 in other_diffs:
            return False
    return True

def _add_to(lst, x, other_diffs, used_sums, diff_set):
    """Add x to lst, update used_sums and diff_set in place."""
    lst.append(x)
    used_sums.add(x + x)
    diff_set.add(0)  # not needed but keep consistent
    for a in lst[:-1]:
        used_sums.add(x + a)
        diff_set.add(x - a)
        diff_set.add(a - x)

def _greedy_build(start_order, other_diffs, used_sums, diff_set, max_size=64):
    """Greedy build a Sidon set from start_order, avoiding other_diffs."""
    lst = []
    for x in start_order:
        if len(lst) >= max_size:
            break
        if _can_add(lst, x, other_diffs, used_sums):
            _add_to(lst, x, other_diffs, used_sums, diff_set)
    return lst

def _greedy_pair(order_a, order_b, max_size=64):
    """Build pair (A,B) greedily, alternating or sequential."""
    A = []
    B = []
    sums_a = set()
    sums_b = set()
    diffs_a = set()
    diffs_b = set()
    
    for x in order_a:
        if len(A) >= max_size:
            break
        if _can_add(A, x, diffs_b, sums_a):
            _add_to(A, x, diffs_b, sums_a, diffs_a)
    
    for x in order_b:
        if len(B) >= max_size:
            break
        if _can_add(B, x, diffs_a, sums_b):
            _add_to(B, x, diffs_a, sums_b, diffs_b)
    
    return A, B

def _greedy_interleaved(order, max_size=64):
    """Build A and B interleaved from a single order."""
    A = []
    B = []
    sums_a = set()
    sums_b = set()
    diffs_a = set()
    diffs_b = set()
    
    for x in order:
        if len(A) < max_size and _can_add(A, x, diffs_b, sums_a):
            _add_to(A, x, diffs_b, sums_a, diffs_a)
        elif len(B) < max_size and _can_add(B, x, diffs_a, sums_b):
            _add_to(B, x, diffs_a, sums_b, diffs_b)
    
    return A, B

def _sidon_greedy(order, max_size=64):
    """Build a single Sidon set greedily."""
    lst = []
    sums = set()
    for x in order:
        if len(lst) >= max_size:
            break
        if _can_add(lst, x, set(), sums):
            _add_to(lst, x, set(), sums, set())
    return lst

def construct():
    start_time = time.time()
    best_total = 0
    best_A = []
    best_B = []
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Strategy 1: Sequential greedy - build A first, then B
    # Try different orderings
    rng = random.Random(42)
    
    # Natural order
    A, B = _greedy_pair(list(range(1, N+1)), list(range(1, N+1)))
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_A = A[:]
        best_B = B[:]
    
    if elapsed() > DEADLINE - 2:
        return [best_A if best_A else [1], best_B if best_B else [2]]
    
    # Reverse order
    A, B = _greedy_pair(list(range(N, 0, -1)), list(range(N, 0, -1)))
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_A = A[:]
        best_B = B[:]
    
    if elapsed() > DEADLINE - 2:
        return [best_A if best_A else [1], best_B if best_B else [2]]
    
    # Interleaved natural
    A, B = _greedy_interleaved(list(range(1, N+1)))
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_A = A[:]
        best_B = B[:]
    
    if elapsed() > DEADLINE - 2:
        return [best_A if best_A else [1], best_B if best_B else [2]]
    
    # Randomized restarts
    for trial in range(50):
        if time_left() < 1.0:
            break
        
        # Random permutation
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        # Try interleaved
        A, B = _greedy_interleaved(order)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        # Try sequential with random orders
        order_a = order[:]
        order_b = order[:]
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        A, B = _greedy_pair(order_a, order_b)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        # Try with different max sizes
        for ms in [12, 14, 16, 18, 20]:
            if time_left() < 0.5:
                break
            A, B = _greedy_pair(order_a, order_b, max_size=ms)
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A[:]
                best_B = B[:]
    
    # Try structured orders: multiples, primes, etc.
    if time_left() > 2.0:
        # Multiples of 2
        order_even = [x for x in range(1, N+1) if x % 2 == 0]
        order_odd = [x for x in range(1, N+1) if x % 2 == 1]
        A, B = _greedy_pair(order_even, order_odd)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        # Multiples of 3
        order_3 = [x for x in range(1, N+1) if x % 3 == 0]
        order_not3 = [x for x in range(1, N+1) if x % 3 != 0]
        A, B = _greedy_pair(order_3, order_not3)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Try building one set first with various orders, then the other
    if time_left() > 3.0:
        for base_order in [list(range(1, N+1)), list(range(N, 0, -1))]:
            if time_left() < 2.0:
                break
            # Build A greedily
            A = []
            sums_a = set()
            diffs_a = set()
            for x in base_order:
                if len(A) >= 20:
                    break
                if _can_add(A, x, set(), sums_a):
                    _add_to(A, x, set(), sums_a, diffs_a)
            
            # Now build B avoiding diffs_a
            B = []
            sums_b = set()
            diffs_b = set()
            for x in base_order:
                if len(B) >= 20:
                    break
                if _can_add(B, x, diffs_a, sums_b):
                    _add_to(B, x, diffs_a, sums_b, diffs_b)
            
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
