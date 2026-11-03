"""Construct finite set U to maximize sumset/difference-set exponent lower bound."""
import time
import math
import random
from collections import Counter

def _score(U):
    """Compute the score for a set U."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            sum_set.add(U[i] + U[j])
        for j in range(n):
            diff_set.add(U[i] - U[j])
            diff_set.add(U[j] - U[i])
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _construct_paper_2003():
    """Construct the size-2003 set from the paper."""
    # Use a base-3 type construction
    # The paper uses sets based on powers and mixed-radix
    # Let's use a construction that gives good ratio
    U = set([0])
    # Powers of 3 construction
    base = 3
    # Build using digits {0, 1} in base 3, but with specific pattern
    # Actually, let's use a known good construction
    # Set of numbers with digits in {0, 1, 2} but restricted
    # Use the construction: U = {sum a_i * 3^i : a_i in {0,1}} for i=0..k
    # This gives |U| = 2^(k+1), |U+U| <= 3^(k+1), |U-U| = 2^(k+1) * 3^k approximately
    # Let's use a different approach - the paper's actual construction
    
    # For size ~2003, use 11 bits (2^11 = 2048)
    # U = {sum b_i * 3^i : b_i in {0,1}, i=0..10}
    k = 10
    U = set()
    for mask in range(2 ** (k + 1)):
        val = 0
        for i in range(k + 1):
            if mask & (1 << i):
                val += 3 ** i
        U.add(val)
    return sorted(U)

def _construct_paper_54265():
    """Construct the size-54265 set from the paper."""
    # For size ~54265, use more bits
    # 2^16 = 65536, too big; 2^15 = 32768, too small
    # Use mixed approach or different base
    # Let's use base 3 with 15-16 digits but with some restrictions
    # Actually, let's use a construction that gives better ratio
    
    # Use the "sum-free" type construction
    # U = {sum b_i * B^i : b_i in {0,1}} with B chosen for good ratio
    # With B=3: |U+U| <= 3^k, |U-U| ~ 2*3^k, ratio ~ 2/3 * 3^k / 3^k
    # The key is the ratio |U-U|/|U+U|
    
    # For base B with digits {0,1}:
    # |U+U| is at most B^k (each digit sum is 0,1,2)
    # |U-U| is at most 2*B^k (each digit diff is -1,0,1)
    # So ratio is at most 2
    
    # To get better ratio, use asymmetric digit sets
    # Let's use a construction from additive combinatorics
    
    # Actually, the best known constructions use "B_h sequences" or similar
    # For now, let's use the base-3 construction with 15 digits
    k = 14  # 2^15 = 32768 elements
    U = set()
    for mask in range(2 ** (k + 1)):
        val = 0
        for i in range(k + 1):
            if mask & (1 << i):
                val += 3 ** i
        U.add(val)
    return sorted(U)

def _local_search(U, time_limit=60.0):
    """Apply local search to improve the score."""
    start_time = time.time()
    U = list(sorted(set(U)))
    best_score = _score(U)
    best_U = list(U)
    
    # Try adding elements
    max_u = max(U)
    # Try elements up to 2*max_u + some margin
    search_max = 2 * max_u + 10
    
    # Also try removing elements
    random.seed(42)
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        # Randomly try to add or remove an element
        if random.random() < 0.7:
            # Try adding
            candidates = []
            # Try elements near existing ones and at strategic positions
            for _ in range(20):
                if random.random() < 0.5:
                    # Near existing element
                    base = random.choice(U)
                    offset = random.randint(-max(1, max_u // 10), max(1, max_u // 10))
                    cand = base + offset
                else:
                    # Random in range
                    cand = random.randint(1, search_max)
                if cand >= 0 and cand not in U:
                    candidates.append(cand)
            
            for cand in candidates:
                U_new = U + [cand]
                s = _score(U_new)
                if s > best_score:
                    best_score = s
                    best_U = list(U_new)
                    U = U_new
                    search_max = max(search_max, 2 * max(U) + 10)
                    break
        else:
            # Try removing (only if set is large enough)
            if len(U) > 10:
                for _ in range(10):
                    idx = random.randint(1, len(U) - 1)  # Don't remove 0
                    if U[idx] == 0:
                        continue
                    U_new = U[:idx] + U[idx+1:]
                    s = _score(U_new)
                    if s > best_score:
                        best_score = s
                        best_U = list(U_new)
                        U = U_new
                        break
    
    return best_U, best_score

def construct():
    """Main entry point: construct the set U."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with the paper constructions
    U1 = _construct_paper_2003()
    U2 = _construct_paper_54265()
    
    # Score both
    s1 = _score(U1)
    s2 = _score(U2)
    
    best_U = U1 if s1 >= s2 else U2
    best_score = max(s1, s2)
    
    # Try local search on the better one
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        U_ls, s_ls = _local_search(best_U, time_limit=remaining - 5.0)
        if s_ls > best_score:
            best_U = U_ls
            best_score = s_ls
    
    # Also try a different construction: use powers of 2 with gaps
    # U = {sum b_i * 2^(2i) : b_i in {0,1}} - this gives sparse sums
    if time.time() - start_time < deadline - 10.0:
        k = 10
        U3 = set([0])
        for mask in range(2 ** (k + 1)):
            val = 0
            for i in range(k + 1):
                if mask & (1 << i):
                    val += 2 ** (2 * i)
            U3.add(val)
        U3 = sorted(U3)
        s3 = _score(U3)
        if s3 > best_score:
            best_U = U3
            best_score = s3
        
        # Local search on U3 if it's good
        remaining = deadline - (time.time() - start_time)
        if remaining > 5.0 and s3 > 0:
            U_ls3, s_ls3 = _local_search(U3, time_limit=remaining - 5.0)
            if s_ls3 > best_score:
                best_U = U_ls3
                best_score = s_ls3
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U = list(best_U) + [0]
    
    return sorted(set(best_U))
