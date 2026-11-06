# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _pair_sums_unique(S):
    """Check if all pair sums (with replacement) in S are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diff_set(S):
    """Return set of nonzero absolute differences in S."""
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _try_add(S, x, used_diffs):
    """Try to add x to Sidon set S. Returns (new_S, new_diffs) or (None, None)."""
    new_S = S + [x]
    # Check Sidon property
    sums = set()
    n = len(new_S)
    for i in range(n):
        for j in range(i, n):
            s = new_S[i] + new_S[j]
            if s in sums:
                return None, None
            sums.add(s)
    # Compute new differences
    new_diffs = set()
    for i in range(n - 1):  # differences involving x (last element)
        d = abs(x - new_S[i])
        if d > 0:
            if d in used_diffs:
                return None, None
            new_diffs.add(d)
    # Also need all existing diffs (but they're already in used_diffs)
    # Actually we just need to check new diffs don't conflict
    return new_S, new_diffs

def _greedy_construct(order_a, order_b, deadline):
    """Greedy construction with given orders for A and B."""
    A = []
    B = []
    diffs_A = set()
    diffs_B = set()
    
    # Build A first
    for x in order_a:
        if time.time() > deadline:
            break
        new_A, new_diffs = _try_add(A, x, diffs_B)
        if new_A is not None:
            A = new_A
            diffs_A.update(new_diffs)
    
    # Build B
    for x in order_b:
        if time.time() > deadline:
            break
        new_B, new_diffs = _try_add(B, x, diffs_A)
        if new_B is not None:
            B = new_B
            diffs_B.update(new_diffs)
    
    return A, B

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    
    best_total = 0
    best_A = []
    best_B = []
    
    # Multiple random restarts with different orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Create random orderings
        universe = list(range(1, 257))
        random.shuffle(universe)
        
        # Try different splits
        # Strategy: interleave or split the shuffled universe
        half = len(universe) // 2
        order_a = universe[:half]
        order_b = universe[half:]
        
        A, B = _greedy_construct(order_a, order_b, deadline)
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = A
            best_B = B
    
    # Also try the original greedy as a baseline
    A_orig = []
    B_orig = []
    diffs_A = set()
    diffs_B = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Try A
        new_A, new_diffs = _try_add(A_orig, x, diffs_B)
        if new_A is not None:
            A_orig = new_A
            diffs_A.update(new_diffs)
        # Try B
        new_B, new_diffs = _try_add(B_orig, x, diffs_A)
        if new_B is not None:
            B_orig = new_B
            diffs_B.update(new_diffs)
    
    total_orig = len(A_orig) + len(B_orig)
    if total_orig > best_total:
        best_total = total_orig
        best_A = A_orig
        best_B = B_orig
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
